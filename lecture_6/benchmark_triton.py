import torch
import triton
import triton.language as tl
import time

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 0.5 * x * (1 + tl.math.erf(x / tl.math.sqrt(2.0)))
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n_elements, block_size)
    gelu_kernel[(num_blocks,)](x, y, n_elements, BLOCK_SIZE=block_size)
    return y

if __name__ == "__main__":
    size = 2**24
    x = torch.randn(size, device='cuda', dtype=torch.float32)

    # 预热 Triton 核函数，完成编译
    print("Warming up Triton kernel...")
    triton_gelu(x) 

    # 再次测量 Triton 核函数的运行时间（不包含编译）
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    y_triton = triton_gelu(x)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_triton_ms = start_event.elapsed_time(end_event)
    print(f"Correct Triton GELU kernel time (after warmup): {elapsed_triton_ms:.4f} ms")

    # 测量 PyTorch 内核时间
    start_event.record()
    y_torch = torch.nn.functional.gelu(x)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_torch_ms = start_event.elapsed_time(end_event)
    print(f"PyTorch built-in GELU took {elapsed_torch_ms:.4f} ms")

    assert torch.allclose(y_triton, y_torch, atol=1e-5), "Outputs do not match!"
    print("Verification successful: Triton and PyTorch outputs match.")