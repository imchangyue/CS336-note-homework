#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
