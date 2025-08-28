import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Any, Tuple
class BucketedDDP(nn.Module):
    """
    用桶（bucket）实现 overlapped gradient communication 的 DDP wrapper。
    接口:
      BucketedDDP(module: nn.Module, bucket_size_mb: float)
      forward(*args, **kwargs)
      finish_gradient_synchronization()
    说明:
      - 按 module.parameters() 的逆序将参数分配到桶中，每个桶容量不超过 bucket_size_mb（MB）
      - 当桶内所有成员参数的梯度就绪后，异步发起一次 all_reduce(flat_bucket_tensor, async_op=True)
      - finish_gradient_synchronization 等待所有 handle 完成，并把平均梯度恢复到对应参数上
    """
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.params: List[torch.nn.Parameter] = [p for p in self.module.parameters() if p.requires_grad]
        self.buckets: List[dict] = []
        self._build_buckets()
        self.bucket_flat_tensors: List[torch.Tensor] = [None] * len(self.buckets)
        self.bucket_remaining: List[int] = [len(b["params"]) for b in self.buckets]
        self.communication_handles: List[Tuple[Any, int]] = []
        self._register_hooks()
        if dist.is_available() and dist.is_initialized():
            self._broadcast_parameters()
    def _build_buckets(self):
        cur_bucket = {"params": [], "sizes": [], "offsets": [], "total_bytes": 0}
        # 从后往前遍历参数，符合反向传播的顺序
        for idx, p in reversed(list(enumerate(self.params))):
            bytes_p = p.numel() * p.element_size()
            
            # 如果当前参数自己就超过桶大小，或者加入后会超，则先关闭当前桶
            if cur_bucket["params"] and (cur_bucket["total_bytes"] + bytes_p > self.bucket_size_bytes or bytes_p > self.bucket_size_bytes):
                self.buckets.append(cur_bucket)
                cur_bucket = {"params": [], "sizes": [], "offsets": [], "total_bytes": 0}
            
            offset = cur_bucket["total_bytes"]
            cur_bucket["params"].append(idx)
            cur_bucket["sizes"].append(p.numel())
            cur_bucket["offsets"].append(offset)
            cur_bucket["total_bytes"] += bytes_p
        if cur_bucket["params"]:
            self.buckets.append(cur_bucket)
    def _register_hooks(self):
        for idx, p in enumerate(self.params):
            if p.requires_grad:
                p.register_hook(self._make_param_hook(idx))
    def _make_param_hook(self, idx: int):
        def hook(grad_tensor: torch.Tensor):
            # 调用我们的处理逻辑
            self._on_param_grad_ready(idx, grad_tensor)
            # 返回零张量来“消费”梯度，防止 autograd 将其累积到 .grad 属性
            return torch.zeros_like(grad_tensor)
        return hook
    def _broadcast_parameters(self):
        for p in self.module.parameters():
            # 也可以同步buffers
            dist.broadcast(p.data, src=0)
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    def _ensure_bucket_buffer(self, bucket_idx: int, device: torch.device, dtype: torch.dtype):
        if self.bucket_flat_tensors[bucket_idx] is None:
            total_numels = sum(self.buckets[bucket_idx]["sizes"])
            self.bucket_flat_tensors[bucket_idx] = torch.zeros(total_numels, dtype=dtype, device=device)
    def _on_param_grad_ready(self, param_idx: int, grad_tensor: torch.Tensor):
        if grad_tensor is None:
            return
        found = False
        for b_idx, b in enumerate(self.buckets):
            try:
                pos = b["params"].index(param_idx)
                found = True
            except ValueError:
                continue
            offset = b["offsets"][pos]
            size = b["sizes"][pos]
            device = grad_tensor.device
            dtype = grad_tensor.dtype
            self._ensure_bucket_buffer(b_idx, device, dtype)
            flat = self.bucket_flat_tensors[b_idx]
            
            # 将新计算出的梯度累加到我们的 bucket buffer 中
            flat[offset:offset + size].add_(grad_tensor.view(-1))
            
            # (已移除) self.params[param_idx].grad = None
            
            self.bucket_remaining[b_idx] -= 1
            if self.bucket_remaining[b_idx] == 0:
                if dist.is_available() and dist.is_initialized():
                    handle = dist.all_reduce(self.bucket_flat_tensors[b_idx], op=dist.ReduceOp.SUM, async_op=True)
                    self.communication_handles.append((handle, b_idx))
                else:
                    self.communication_handles.append((None, b_idx))
            
            if found:
                break
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        for handle, b_idx in self.communication_handles:
            if handle is not None:
                handle.wait()
            
            flat = self.bucket_flat_tensors[b_idx]
            if flat is None:
                continue
            
            # 平均梯度
            flat.div_(world_size)
            # 将平均后的梯度复制回参数的 .grad 属性
            # 注意这里是 ptr，不是 offset
            ptr = 0
            for pos, p_idx in enumerate(self.buckets[b_idx]["params"]):
                numel = self.buckets[b_idx]["sizes"][pos]
                param = self.params[p_idx]
                grad_view = flat[ptr:ptr + numel].view(param.shape)
                
                # 为参数设置 .grad 属性，以便 optimizer 使用
                # 如果原来的 grad 不是 None (比如 optimizer.zero_grad(set_to_none=False))
                # 最好先清零再加，或者直接覆盖
                if param.grad is None:
                    param.grad = grad_view.clone()
                else:
                    # 如果 .grad 已存在，直接覆盖
                    param.grad.copy_(grad_view)
                
                ptr += numel
        self._reset_bucket_state()
    def _reset_bucket_state(self):
        for b_idx, b in enumerate(self.buckets):
            self.bucket_remaining[b_idx] = len(b["params"])
            if self.bucket_flat_tensors[b_idx] is not None:
                self.bucket_flat_tensors[b_idx].zero_()
        self.communication_handles.clear()
        
    # --- 代理方法，确保 DDP wrapper 的行为和原始 module 一致 ---
    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)
    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)