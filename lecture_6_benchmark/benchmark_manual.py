import time
import os
from typing import Callable
from statistics import mean
import torch
from torch.utils.cpp_extension import load_inline

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times
    mean_time = mean(times) # @inspect mean_time
    print(f"{description} mean time: {mean_time:.2f} ms")
    return mean_time

def run_cuda_operation(dim: int, operation: Callable):
    """Create a CUDA tensor of the given dimension and apply the operation."""
    x = torch.randn(dim).cuda()
    return lambda: operation(x)

def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")

def create_cuda_gelu():
    """Create and compile the CUDA GELU implementation."""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 设置 CUDA 错误阻塞
    
    # 检查是否有可用的 CUDA
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping CUDA GELU implementation.")
        return None
    
    # CUDA kernel代码
    cuda_gelu_src = """
#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

// CUDA kernel for GELU
__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  //当前块的位置和块的偏移量
    if (i < num_elements) {
        out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
    }
}

// Helper function to compute ceiling of division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// CPU调用GPU的核函数
torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda());  // 检查张量是否在 CUDA 设备上
    TORCH_CHECK(x.is_contiguous());     // 检查张量是否是连续的
    
    torch::Tensor y = torch::empty_like(x);  // 创建一个与输入张量相同大小的输出张量
    int num_elements = x.numel();  // 获取输入张量的元素数量
    int block_size = 1024;  // 每个线程块的线程数
    int num_blocks = cdiv(num_elements, block_size);  // 计算需要的线程块数量
    
    // 启动内核
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();  // 错误检查
    
    return y;  // 返回结果张量
}
"""
    
    # C++ 接口声明
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"
    
    try:
        # 编译 CUDA 代码并加载为 Python 模块
        module = load_inline(
            cuda_sources=[cuda_gelu_src],
            cpp_sources=[cpp_gelu_src],
            functions=["gelu"],  # 绑定 gelu 函数
            extra_cflags=["-O2"],
            verbose=True,
            name="inline_gelu",
            build_directory="./cuda_gelu_build",  # 临时目录
        )
        
        cuda_gelu = getattr(module, "gelu")  # 获取编译后的函数
        print("CUDA GELU compiled successfully!")
        return cuda_gelu  # 返回 CUDA 实现的 gelu 函数
        
    except Exception as e:
        print(f"Failed to compile CUDA GELU: {e}")
        return None

def main():
    """Main benchmarking function."""
    print("Starting CUDA GELU benchmark comparison...")
    print("=" * 50)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Exiting.")
        return
    
    # 设置测试维度
    dim = 16384
    
    print("CUDA Benchmarks:")
    
    # PyTorch CUDA版本
    pytorch_cuda_time = benchmark("pytorch_gelu (CUDA)", 
                                run_cuda_operation(dim=dim, operation=pytorch_gelu))
    
    # 自定义CUDA版本
    cuda_gelu_func = create_cuda_gelu()
    if cuda_gelu_func is not None:
        cuda_time = benchmark("cuda_gelu (Custom CUDA)", 
                            run_cuda_operation(dim=dim, operation=cuda_gelu_func))
    else:
        cuda_time = None
        print("Custom CUDA GELU is not available.")
    
    # 结果比较
    print("\n" + "=" * 50)
    print("Performance Comparison:")
    print("=" * 50)
    
    if pytorch_cuda_time is not None and cuda_time is not None:
        speedup = pytorch_cuda_time / cuda_time
        print(f"PyTorch CUDA: {pytorch_cuda_time:.2f} ms")
        print(f"Custom CUDA: {cuda_time:.2f} ms")
        print(f"Custom CUDA is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch CUDA")
    else:
        print("Could not compare times - one of the implementations failed")

if __name__ == "__main__":
    main()