import torch
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,  # 输入张量 x 和 weight 的指针
    grad_output_ptr,  # 损失对输出的梯度（反向传播的输入）的指针
    grad_x_ptr, partial_grad_weight_ptr,  # 损失对输入 x 和 weight 的梯度（反向传播的输出）的指针
    stride_xr, stride_xd,  # 输入 x 的行步长和维度步长
    stride_wd,  # 输入 weight 的维度步长
    stride_gr,  # 输出梯度 grad_output 的行步长
    stride_gxr, stride_gxd,  # 输出梯度 grad_x 的行步长和维度步长
    stride_gwb, stride_gwd,  # 部分梯度 partial_grad_weight 的块步长和维度步长
    NUM_ROWS, D,  # 整个张量的行数和维度
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # 行和维度方向的块大小，编译时常量
):
    # 每个程序（或线程块）处理一个行块（tile）
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # ------------------- 输入/输出张量块指针的创建 -------------------
    # 创建 grad_output 的块指针，它是一个一维向量
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), 
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 创建输入 x 的块指针，它是一个二维矩阵
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), 
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 创建输入 weight 的块指针，它是一个一维向量
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), 
        strides=(stride_wd,),
        offsets=(0,), 
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 创建 grad_x 的块指针，它是一个二维矩阵
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), 
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 创建 partial_grad_weight 的块指针，用于存储每个线程块计算的部分结果
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), 
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # ------------------- 计算循环 -------------------
    # 遍历维度 D，每次处理一个 D_TILE_SIZE 大小的块
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载 grad_output，尺寸为 (ROWS_TILE_SIZE,)
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") 

        # ------------------- 计算 grad_x -------------------
        # 加载 weight，尺寸为 (D_TILE_SIZE,)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") 
        # 外积操作：(ROWS_TILE_SIZE, 1) * (1, D_TILE_SIZE) -> (ROWS_TILE_SIZE, D_TILE_SIZE)
        # 这对应于 Eq 2，即 wj · (∇f(x,w)L)i
        grad_x_row = grad_output[:, None] * weight[None, :]
        # 将计算出的梯度写回 grad_x
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # ------------------- 计算 grad_w 的部分和 -------------------
        # 加载输入 x 的当前块，尺寸为 (ROWS_TILE_SIZE, D_TILE_SIZE)
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") 
        # 矩阵乘法和求和操作，对应于 Eq 3
        # `row * grad_output[:, None]` 是逐元素相乘
        # `tl.sum(..., axis=0)` 沿着行方向求和，得到 (1, D_TILE_SIZE) 的结果
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        # 将部分和结果存储到 partial_grad_weight
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) 

        # ------------------- 移动指针到下一个块 -------------------
        # 沿着维度 D 的方向前进一个块大小
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # forward pass 部分（此处未展示）
        pass

    @staticmethod
    def backward(ctx, grad_out):
        # 从 ctx 中恢复保存的张量和常量
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # 创建用于存储部分梯度的中间缓冲区
        # 缓冲区的行数等于线程块的数量
        partial_grad_weight = torch.empty((tl.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        # 创建用于存储 grad_x 的空张量
        grad_x = torch.empty_like(x)

        # 启动 Triton 内核
        # 启动配置：(网格维度) = (行块的数量,)
        weighted_sum_backward[(tl.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        # 最终的 grad_weight 梯度需要对所有线程块的部分结果求和
        grad_weight = partial_grad_weight.sum(axis=0)
        
        # 返回对 x 和 weight 的梯度
        return grad_x, grad_weight

# 使用 .apply 方法创建可微分的函数
f_weightedsum = WeightedSumFunc.apply