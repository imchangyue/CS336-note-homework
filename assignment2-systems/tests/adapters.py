from __future__ import annotations



import torch
import triton
import triton.language as tl
from typing import Type
from einops import einsum
from .ddp_overlap_individual_parameters import OverlappedDDP, ToyModel, ToyDataset
from .ddp_overlap_bucketed import BucketedDDP
def _flash_backward_kernel(Q, K, V, O, L, dO, is_causal):
    if Q.dim() == 4:
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        Q_proc, K_proc, V_proc, dO_proc, O_proc = Q, K, V, dO, O
        einsum_str = 'b h'
    else: # Q.dim() == 3
        B, N_q, d = Q.shape
        N_k = K.shape[1]
        H = 1
        Q_proc, K_proc, V_proc, dO_proc, O_proc = Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1), dO.unsqueeze(1), O.unsqueeze(1)
        einsum_str = 'b h'

    scale = d ** -0.5
    
    S = einsum(Q_proc, K_proc, f'{einsum_str} q d, {einsum_str} k d -> {einsum_str} q k') * scale

    if is_causal:
        causal_mask = torch.triu(torch.full_like(S, -torch.inf, device=S.device), diagonal=1)
        S = S + causal_mask

    P = torch.softmax(S, dim=-1)

    dV = einsum(P, dO_proc, f'{einsum_str} q k, {einsum_str} q d -> {einsum_str} k d')

    D = torch.sum(dO_proc * O_proc, dim=-1, keepdim=True)
    
    dOV = einsum(dO_proc, V_proc, f'{einsum_str} q d, {einsum_str} k d -> {einsum_str} q k')
    dS = P * (dOV - D)
    
    dQ = einsum(dS, K_proc, f'{einsum_str} q k, {einsum_str} k d -> {einsum_str} q d') * scale
    dK = einsum(dS.transpose(-2, -1), Q_proc, f'{einsum_str} k q, {einsum_str} q d -> {einsum_str} k d') * scale

    return dQ, dK, dV

# 确保在 CPU 和 GPU 上都能工作
try:
    flash_backward_kernel_compiled = torch.compile(_flash_backward_kernel)
except RuntimeError:
    flash_backward_kernel_compiled = _flash_backward_kernel

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        q_shape = Q.shape
        was_3d = len(q_shape) == 3

        if was_3d:
            Q, K, V = [x.unsqueeze(1) for x in (Q, K, V)]
        
        B, H, N_q, d = Q.shape
        _, _, N_k, _ = K.shape
        
        B_q = 16 
        B_k = 16
        
        O = torch.zeros_like(Q)
        L = torch.zeros(B * H, N_q, device=Q.device)
        scale = d ** -0.5
        
        for i in range(0, N_q, B_q):
            q_start = i
            q_end = min((i + B_q), N_q)
            Q_i = Q[:, :, q_start:q_end, :]

            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(B, H, q_end - q_start, 1, device=Q.device)
            m_i = torch.full((B, H, q_end - q_start, 1), -torch.inf, device=Q.device)
            
            for j in range(0, N_k, B_k):
                k_start = j
                k_end = min((j + B_k), N_k)
                K_j = K[:, :, k_start:k_end, :]
                V_j = V[:, :, k_start:k_end, :]
                
                S_ij = (Q_i @ K_j.transpose(-2, -1)) * scale

                if is_causal and (k_end > q_start):
                    current_q_indices = torch.arange(q_start, q_end, device=S_ij.device)
                    current_k_indices = torch.arange(k_start, k_end, device=S_ij.device)
                    causal_mask_block = current_q_indices[:, None] >= current_k_indices[None, :]
                    S_ij = torch.where(causal_mask_block[None, None, :, :], S_ij, -torch.inf)

                m_i_new = torch.maximum(m_i, S_ij.max(dim=-1, keepdim=True)[0])
                P_tilde_ij = torch.exp(S_ij - m_i_new)
                l_i_new = torch.exp(m_i - m_i_new) * l_i + P_tilde_ij.sum(dim=-1, keepdim=True)
                O_i = (torch.exp(m_i - m_i_new)) * O_i + (P_tilde_ij @ V_j)
                l_i = l_i_new
                m_i = m_i_new

            O_i_final = O_i / l_i
            L_i_final = (m_i + torch.log(l_i)).squeeze(-1)
            O[:, :, q_start:q_end, :] = O_i_final
            L_reshaped_slice = L_i_final.reshape(B * H, -1)
            L[:, q_start:q_end] = L_reshaped_slice

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.was_3d = was_3d
        
        if was_3d:
            O = O.squeeze(1)
            
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        was_3d = ctx.was_3d

        if Q.dim() == 4 and dO.dim() == 3:
            dO = dO.unsqueeze(1)
        
        dQ, dK, dV = flash_backward_kernel_compiled(Q, K, V, O, L, dO, is_causal)
        
        # 💡 SOLUTION: ensure the gradients are squeezed back to 3D if the original input was 3D
        if was_3d:
            dQ = dQ.squeeze(1)
            dK = dK.squeeze(1)
            dV = dV.squeeze(1)
        
        return dQ, dK, dV, None

def get_flashattention_autograd_function_pytorch() -> Type:
    return FlashAttentionPyTorch

def get_flashattention_autograd_function_triton() -> Type:
    return FlashAttentionTriton

@triton.jit
def flash_fwd_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    # Stride information for each tensor
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    # Other parameters
    N_QUERIES, N_KEYS,
    scale,
    # Compile-time constants
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Triton Kernel for the FlashAttention-2 forward pass.
    
    Each program instance computes one tile of the output O for one batch item.
    """
    # 1. Get Program IDs to identify the current work item
    # This program computes the `query_tile_index`-th tile of Q for the `batch_index`-th batch item
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 2. Create block pointers for Q, O, and L for the current program instance
    # These pointers are offset to the specific tile and batch this program is responsible for.
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L is a vector, so its pointer setup is simpler
    L_ptr_tile = L_ptr + batch_index * stride_lb + (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))

    # 3. Load the Q tile from HBM into SRAM once
    q_i = tl.load(Q_block_ptr)

    # 4. Initialize on-chip accumulators (in SRAM)
    # These must be float32 for precision during accumulation
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # 5. Main loop over blocks of K and V
    # We iterate through the key/value sequence dimension
    for k_start_offset in range(0, N_KEYS, K_TILE_SIZE):
        # Create block pointers for the current K and V tiles
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_start_offset, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_start_offset, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        # Load K_j and V_j tiles from HBM into SRAM
        k_j = tl.load(K_block_ptr)
        v_j = tl.load(V_block_ptr)

        # --- Core Attention Computation ---
        # Compute S_ij = Q_i * K_j^T * scale
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        
        # (c) Apply causal mask if enabled
        if is_causal:
            # Get the indices for the current query and key blocks
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = k_start_offset + tl.arange(0, K_TILE_SIZE)
            # Create a mask where q_index >= k_index
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            # Apply the mask to S_ij, setting masked elements to a large negative number
            s_ij = tl.where(causal_mask, s_ij, -1e6)

        # --- Online Softmax Update ---
        # m_i_new = max(m_i, rowmax(S_ij))
        m_ij = tl.max(s_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # P_tilde_ij = exp(S_ij - m_i_new)
        p_tilde_ij = tl.exp(s_ij - m_i_new[:, None])
        
        # alpha = exp(m_i - m_i_new)
        alpha = tl.exp(m_i - m_i_new)

        # l_i_new = alpha * l_i + rowsum(P_tilde_ij)
        l_i_new = alpha * l_i + tl.sum(p_tilde_ij, axis=1)

        # acc_new = alpha * acc + P_tilde_ij @ V_j
        # Rescale the old accumulator
        acc = acc * alpha[:, None]
        # Cast P_tilde to V's dtype before dot product for performance
        p_tilde_ij = p_tilde_ij.to(v_j.dtype)
        # Add the contribution from the current V tile
        acc = tl.dot(p_tilde_ij, v_j, acc)
        
        # Update running stats for the next iteration
        l_i = l_i_new
        m_i = m_i_new

    # 6. Post-processing and storing results to HBM
    # Final normalization of the accumulator
    o_i = acc / l_i[:, None]
    
    # Compute L_i = m_i + log(l_i)
    l_final = m_i + tl.log(l_i)

    # Write the final output tile O_i and L_i back to HBM
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))
    tl.store(L_ptr_tile, l_final)


# ------------------------------------------------------------------
# (b) & (c) PyTorch Autograd Function Wrapper
# ------------------------------------------------------------------
def _flash_backward_kernel_triton(Q, K, V, O, L, dO, is_causal):
    # 根据方程式 13-19 进行计算
    # 获取张量维度
    B, N_q, d = Q.shape
    N_k = K.shape[1]
    scale = d ** -0.5
    
    # 重新计算注意力分数矩阵 S
    # S = Q @ K.T * scale
    S = einsum(Q, K, 'b q d, b k d -> b q k') * scale

    # 应用因果掩码 (如果需要)
    if is_causal:
        # 创建一个掩码，使得上三角部分（未来 token）为 -inf
        causal_mask = torch.triu(torch.full_like(S, -torch.inf), diagonal=1)
        S = S + causal_mask

    # 重新计算 P (在反向传播中避免了显式保存它)
    P = torch.softmax(S, dim=-1)

    # 1. 计算 dV
    # dV = P.T @ dO
    # 这里的 einsum 对应公式 14
    dV = einsum(P, dO, 'b q k, b q d -> b k d')

    # 2. 计算 D (向量)
    # D = sum(dO * O, dim=-1)
    # 这里的 sum 对应公式 16
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    
    # 3. 计算 dS
    # dS = P * (dO @ V.T - D)
    # 这里的 einsum 和元素相乘对应公式 17 和 18
    # 首先计算 dO @ V.T
    dOV = einsum(dO, V, 'b q d, b k d -> b q k')
    dS = P * (dOV - D)
    
    # 4. 计算 dQ 和 dK
    # dQ = dS @ K * scale
    # dK = dS.T @ Q * scale
    # 这里的 einsum 对应公式 19
    dQ = einsum(dS, K, 'b q k, b k d -> b q d') * scale
    dK = einsum(dS.transpose(-2, -1), Q, 'b k q, b q d -> b k d') * scale

    return dQ, dK, dV
flash_backward_kernel_compiled1 = torch.compile(_flash_backward_kernel_triton, fullgraph=True)
class FlashAttentionTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # The test case _make_attn_inputs() generates 3D tensors of shape (batch_size, seq_len, dim)
        # We need to correctly handle this case, which is a single head.
        if Q.dim() == 3:
            B, N_q, d = Q.shape
            # For a single head, we can treat the 'batch_size' as 'batch_size * num_heads'
            # The test function will pass a 3D tensor where B=4, N_q=128, d=64.
            # So, B_H = B * H = 4 * 1 = 4.
            # The shapes will be (4, 128, 64).
            B_H = B
            N_k = K.shape[1]
            Q_reshaped = Q
            K_reshaped = K
            V_reshaped = V

        # elif Q.dim() == 4:
        #     B, H, N_q, d = Q.shape
        #     B_H = B * H
        #     N_k = K.shape[2]
        #     # Reshape to (B*H, N_q, d) for the kernel
        #     Q_reshaped = Q.reshape(B_H, N_q, d)
        #     K_reshaped = K.reshape(B_H, N_k, d)
        #     V_reshaped = V.reshape(B_H, N_k, d)
        else:
             raise ValueError(f"Input Q must be 3D or 4D, but got {Q.dim()}D")
        
        # Define tile sizes. These can be tuned for performance.
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # Allocate output tensors
        O_reshaped = torch.empty_like(Q_reshaped)
        # We need to allocate L with the same batch dimension as the reshaped inputs.
        L = torch.empty(B_H, N_q, device=Q.device, dtype=torch.float32)

        # Define the launch grid
        # Each program instance handles one query tile for one batch/head combination
        grid = (triton.cdiv(N_q, Q_TILE_SIZE), B_H)
        
        scale = d ** -0.5

        # Launch the Triton kernel
        flash_fwd_kernel[grid](
            # Pointers
            Q_reshaped, K_reshaped, V_reshaped, O_reshaped, L,
            # Strides
            Q_reshaped.stride(0), Q_reshaped.stride(1), Q_reshaped.stride(2),
            K_reshaped.stride(0), K_reshaped.stride(1), K_reshaped.stride(2),
            V_reshaped.stride(0), V_reshaped.stride(1), V_reshaped.stride(2),
            O_reshaped.stride(0), O_reshaped.stride(1), O_reshaped.stride(2),
            L.stride(0), L.stride(1),
            # Other parameters
            N_q, N_k,
            scale,
            # Constexpr
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        # The output O should have the original shape (B, N_q, d)
        O = O_reshaped
        
        # The test expects L to have the shape (batch_size, n_queries), which is (4, 128) in this case.
        # Our `L` tensor already has this shape (B_H, N_q), where B_H = B.
        # We save it for the backward pass.
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        # 调用用 torch.compile 编译的反向传播函数
        # 传入所有需要的张量
        dQ, dK, dV = flash_backward_kernel_compiled1(Q, K, V, O, L, dO, is_causal)
        
        return dQ, dK, dV, None



def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    return OverlappedDDP(module)



def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    return BucketedDDP(module=module, bucket_size_mb=bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # ddp_model.reset_buckets()


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
