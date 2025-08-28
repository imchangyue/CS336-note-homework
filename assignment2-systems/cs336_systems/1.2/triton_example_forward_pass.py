import torch
import triton
import triton.language as tl
from einops import rearrange # 示例中使用了einops, 这里补充导入以便代码完整运行


def cdiv(a, b):
    """向上取整除法"""
    return -(-a // b)

# ------------------------------------------------------------------
# Part 1: Triton Kernel - 真正的GPU计算核心
# ------------------------------------------------------------------

@triton.jit # Triton的JIT（即时编译）装饰器，会将这个Python函数编译成高效的GPU Kernel
def weighted_sum_fwd(
    # --- 输入/输出张量的指针 ---
    x_ptr,           # 指向输入张量 x 第一个元素的指针
    weight_ptr,      # 指向权重张量 weight 第一个元素的指针
    output_ptr,      # 指向输出张量 output 第一个元素的指针

    # --- 张量的步长 (Strides) ---
    # 步长告诉Kernel如何在内存中移动以到达下一个元素。
    # 例如，对于一个2D矩阵，x_stride_row是移动到下一行时指针需要跳过的元素数量。
    x_stride_row,  
    # 例如，对于一个2D矩阵，x_stride_dim是移动到下一列时指针需要跳过的元素数量。
    x_stride_dim,  
    weight_stride_dim, # 对于1D向量，步长通常是1  
    output_stride_row, # 对于1D向量，步长通常是1 

    # --- 张量的维度信息 ---
    ROWS, # 输入张量 x 的总行数 (经过flatten之后)
    D,    # x 的特征维度，也是 weight 的长度

    # --- 分块大小 (Tile Sizes) ---
    # 这些必须是编译时常量（tl.constexpr），以便Triton进行编译时优化。
    # 它们定义了每个线程块（Thread Block）处理的数据块的大小。
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    # 1. --- 并行化策略：确定当前线程块负责哪一部分数据 ---
    # 获取当前程序实例（program instance / thread block）的ID。
    # 这是一个1D的启动网格，所以我们只关心第0维的ID。
    row_tile_idx = tl.program_id(0)

    # 2. --- Block Pointer 设置：简化内存访问 ---
    # Block Pointer 是Triton的强大抽象，它让我们像操作N维数组一样安全地访问内存，
    # 而无需手动进行复杂的指针运算。

    # 为输入张量 x 创建一个块指针
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,                                     # 基础指针：指向张量的起始位置
        shape=(ROWS, D),                                # 全局形状：整个张量的形状
        strides=(x_stride_row, x_stride_dim),           # 全局步长：在每个维度上移动的步长
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),     # 初始偏移：根据线程块ID，计算出当前块的起始行
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),      # 块形状：定义一次load/store操作的数据块大小
        order=(1, 0)                                    # 内存布局顺序：(1, 0)表示第1维(D)是内存连续的，有助于优化
    )

    # 为权重向量 weight 创建一个块指针 (1D情况)
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    # 为输出向量 output 创建一个块指针 (1D情况)
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    # 3. --- 计算过程 ---
    # 初始化一个累加器（accumulator），用于存储当前块的计算结果。
    # 这通常是在GPU核心的高速SRAM/寄存器中，读写速度极快。
    output_accumulator = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # 对D维度（列）进行分块遍历。如果D维度太大，一个块装不下，就需要多次迭代来完成一行数据的计算。
    for i in range(tl.cdiv(D, D_TILE_SIZE)): # tl.cdiv 是Triton内置的向上取整除法
        # 从全局内存加载数据块到SRAM/寄存器
        # boundary_check告诉Triton在访问可能越界的维度时要小心处理。
        # padding_option="zero"表示如果访问越界，则用0来填充，确保计算正确性。
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)

        # 执行核心计算：广播(broadcast)权重，然后进行元素乘法，最后按行求和
        output_accumulator += tl.sum(row * weight[None, :], axis=1)

        # 移动块指针到下一个D维度上的块，为下一次迭代做准备。
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE)) # 在第1维（列）上前进 D_TILE_SIZE
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,)) # 前进 D_TILE_SIZE

    # 4. --- 写回结果 ---
    # 将累加器中的最终计算结果从SRAM写回到全局内存的输出张量中。
    # 同样需要边界检查，因为总行数可能不是ROWS_TILE_SIZE的整数倍。
    tl.store(output_block_ptr, output_accumulator, boundary_check=(0,))


# ------------------------------------------------------------------
# Part 2: PyTorch Wrapper - 连接Triton Kernel和PyTorch的桥梁
# ------------------------------------------------------------------

class WeightedSumFunc(torch.autograd.Function):
    # 继承自torch.autograd.Function，这是将自定义操作接入PyTorch自动求导机制的标准方法。
    
    @staticmethod
    def forward(ctx, x, weight):
        """
        定义前向传播逻辑
        ctx: 上下文对象，用于在反向传播时传递信息或缓存张量
        x, weight: 输入的PyTorch张量
        """
        # --- 1. 数据准备和校验 ---
        D = x.shape[-1]
        
        # 保存原始输入形状，以便在最后恢复输出形状
        input_shape = x.shape
        # 将任意维度的输入x重塑为一个2D矩阵 `(总行数, D)`。
        # 这大大简化了Triton Kernel的设计，让它只需处理通用的2D情况。
        x = rearrange(x, "... d -> (...) d")

        # ctx.save_for_backward是autograd的关键！
        # 它会缓存张量x和weight，以便在反向传播计算梯度时使用。
        ctx.save_for_backward(x, weight)
        
        # 进行一系列断言，确保输入满足Kernel的要求
        assert len(weight.shape) == 1 and weight.shape[0] == D, "维度不匹配"
        assert x.is_cuda and weight.is_cuda, "期望输入为CUDA张量"
        assert x.is_contiguous(), "我们的指针运算假设x是内存连续的"

        # --- 2. 配置Kernel启动参数 ---
        # 定义Triton Kernel中使用的分块大小。这些是重要的性能调优参数。
        # 不同的GPU架构、数据类型和问题规模，最优的分块大小也不同。
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # 举例：让D维度的循环大致执行16次
        ctx.ROWS_TILE_SIZE = 16 # 举例：每个线程块一次处理16行
        ctx.input_shape = input_shape # 缓存原始形状

        # 初始化一个空的PyTorch输出张量，用于接收Kernel的计算结果。
        n_rows = x.shape[0]
        y = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

        # --- 3. 启动Triton Kernel ---
        # 定义启动网格（Launch Grid）。它告诉GPU要启动多少个程序实例（线程块）。
        # 这里的元组 `(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)` 指定了网格的维度和大小。
        # 我们需要 `cdiv(n_rows, ctx.ROWS_TILE_SIZE)` 个线程块来覆盖所有的行。
        grid = (cdiv(n_rows, ctx.ROWS_TILE_SIZE),)

        # 使用 `kernel[grid](...)` 语法启动Kernel
        weighted_sum_fwd[grid](
            x, weight, y,                                       # 张量（会被自动转为指针）
            x.stride(0), x.stride(1),                            # x的步长
            weight.stride(0),                                    # weight的步长
            y.stride(0),                                         # y的步长
            n_rows, D,                                           # 维度大小
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,                   # 分块大小 (作为编译时常量传入)
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        # --- 4. 返回结果 ---
        # 将计算结果从扁平的形状恢复为原始期望的输出形状并返回。
        return y.view(input_shape[:-1])

def main():
    # 测试Triton实现的前向传播
    torch.manual_seed(0)
    x = torch.randn(32, 128, device='cuda') # 示例输入
    weight = torch.randn(128, device='cuda') # 示例权重

    # 使用自定义的Triton操作
    y_triton = WeightedSumFunc.apply(x, weight)

    # 使用PyTorch的内置操作进行对比
    y_torch = (x * weight).sum(dim=-1)

    # 验证两者结果是否接近
    print("结果是否接近:", torch.allclose(y_triton, y_torch, atol=1e-5))
    print("Triton 结果:", y_triton)
    print("PyTorch 结果:", y_torch)

if __name__ == "__main__":

    main()