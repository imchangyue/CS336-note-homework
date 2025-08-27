import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from tabulate import tabulate
from typing import Type
import math

# 用户提供的 Triton FlashAttention-2 前向传递代码
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Triton Kernel for the FlashAttention-2 forward pass.
    每个程序实例为每个批次项计算一个输出 O 块。
    """
    # 1. 获取程序 ID 以识别当前工作项
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 2. 为当前程序实例创建 Q, O, L 的块指针
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
    L_ptr_tile = L_ptr + batch_index * stride_lb + (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))

    # 3. 将 Q 块从 HBM 加载到 SRAM 一次
    q_i = tl.load(Q_block_ptr)

    # 4. 初始化片上累加器（在 SRAM 中）
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # 5. 主循环遍历 K 和 V 块
    for k_start_offset in range(0, N_KEYS, K_TILE_SIZE):
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
        
        k_j = tl.load(K_block_ptr)
        v_j = tl.load(V_block_ptr)

        # --- 核心注意力计算 ---
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        
        # 如果启用，应用因果掩码
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = k_start_offset + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            s_ij = tl.where(causal_mask, s_ij, -1e6)

        # --- 在线 Softmax 更新 ---
        m_ij = tl.max(s_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p_tilde_ij = tl.exp(s_ij - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p_tilde_ij, axis=1)
        acc = acc * alpha[:, None]
        p_tilde_ij = p_tilde_ij.to(v_j.dtype)
        acc = tl.dot(p_tilde_ij, v_j, acc)
        
        # 更新运行状态以备下一次迭代
        l_i = l_i_new
        m_i = m_i_new

    # 6. 后处理并将结果存储到 HBM
    o_i = acc / l_i[:, None]
    l_final = m_i + tl.log(l_i)

    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))
    tl.store(L_ptr_tile, l_final)


# 用户提供的用于反向传递的 PyTorch 函数
# 这不是一个 Triton kernel
def _flash_backward_kernel_triton(Q, K, V, O, L, dO, is_causal):
    scale = Q.shape[-1] ** -0.5
    S = torch.einsum('b h q d, b h k d -> b h q k', Q, K) * scale
    if is_causal:
        causal_mask = torch.triu(torch.full_like(S, -torch.inf, device=S.device), diagonal=1)
        S = S + causal_mask
    P = torch.softmax(S, dim=-1)
    dV = torch.einsum('b h q k, b h q d -> b h k d', P.transpose(-2, -1), dO)
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    dOV = torch.einsum('b h q d, b h k d -> b h q k', dO, V)
    dS = P * (dOV - D)
    dQ = torch.einsum('b h q k, b h k d -> b h q d', dS, K) * scale
    dK = torch.einsum('b h k q, b h q d -> b h k d', dS.transpose(-2, -1), Q) * scale
    return dQ, dK, dV

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        # 处理 3D (batch, seq, dim) 和 4D (batch, head, seq, dim) 输入
        was_3d = len(Q.shape) == 3
        if was_3d:
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        
        B, H, N_q, d = Q.shape
        N_k = K.shape[2]
        
        Q_reshaped = Q.reshape(B * H, N_q, d)
        K_reshaped = K.reshape(B * H, N_k, d)
        V_reshaped = V.reshape(B * H, N_k, d)
        
        # 根据序列长度动态调整块大小
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        if N_q > 4096:
             Q_TILE_SIZE = 128
        if N_k > 4096:
             K_TILE_SIZE = 128
        
        O_reshaped = torch.empty_like(Q_reshaped)
        L = torch.empty(B * H, N_q, device=Q.device, dtype=torch.float32)
        
        grid = (triton.cdiv(N_q, Q_TILE_SIZE), B * H)
        scale = d ** -0.5
        
        flash_fwd_kernel[grid](
            Q_reshaped, K_reshaped, V_reshaped, O_reshaped, L,
            Q_reshaped.stride(0), Q_reshaped.stride(1), Q_reshaped.stride(2),
            K_reshaped.stride(0), K_reshaped.stride(1), K_reshaped.stride(2),
            V_reshaped.stride(0), V_reshaped.stride(1), V_reshaped.stride(2),
            O_reshaped.stride(0), O_reshaped.stride(1), O_reshaped.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        O = O_reshaped.reshape(B, H, N_q, d)
        if was_3d:
            O = O.squeeze(1)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.was_3d = was_3d
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        was_3d = ctx.was_3d
        
        if Q.dim() == 4 and dO.dim() == 3:
            dO = dO.unsqueeze(1)
        
        # 使用用户提供的基于 PyTorch 的反向传递
        dQ, dK, dV = _flash_backward_kernel_triton(Q, K, V, O, L, dO, is_causal)
        
        if was_3d:
            dQ = dQ.squeeze(1)
            dK = dK.squeeze(1)
            dV = dV.squeeze(1)
            
        return dQ, dK, dV, None

# 用于比较的常规 PyTorch 注意力实现
def pytorch_attention_fwd(Q, K, V, is_causal=False):
    """
    此函数使用标准的 PyTorch 缩放点积注意力，该函数经过高度优化，
    可作为强大的基线。
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

def run_benchmark():
    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this script on a machine with a CUDA-enabled GPU.")
        return
    
    # 定义扫描参数
    #seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    seq_lengths = [128,256,512,1024, 2048, 4096, 8192]
    #dims = [16, 32, 64, 128]
    dims = [16, 32, 64]
    data_types = [torch.bfloat16, torch.float32]
    
    results = []
    
    # 固定参数，按照问题描述
    batch_size = 1
    num_heads = 1
    is_causal = True
    
    print("Starting CUDA warm-up run...")
    # 热身运行以初始化 CUDA 上下文并防止第一次测量变慢
    warmup_Q = torch.randn(1, 1, 1024, 64, device=device, dtype=torch.float32, requires_grad=True)
    warmup_K = torch.randn_like(warmup_Q)
    warmup_V = torch.randn_like(warmup_Q)
    FlashAttentionTriton.apply(warmup_Q, warmup_K, warmup_V, is_causal)
    pytorch_attention_fwd(warmup_Q, warmup_K, warmup_V, is_causal=is_causal)
    print("Warm-up complete. Starting benchmark...")

    for dtype in data_types:
        for seq_len in seq_lengths:
            for dim in dims:
                print(f"Benchmarking: Seq Len={seq_len}, Dim={dim}, Dtype={str(dtype).split('.')[-1]}...")
                
                # 生成带有 `requires_grad=True` 的输入以进行反向传递
                try:
                    Q = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
                    K = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
                    V = torch.randn(batch_size, num_heads, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
                except Exception as e:
                    print(f"Skipping seq_len={seq_len} dim={dim} dtype={dtype} due to memory allocation failure: {e}")
                    continue
                
                # 基于 Triton 的 FlashAttention 基准测试
                try:
                    ms_fwd_triton = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(Q, K, V, is_causal))
                    # 在后向传递基准测试中，每次都重新执行前向传递以创建新图
                    ms_bwd_triton = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(Q, K, V, is_causal).backward(torch.ones_like(Q)), grad_to_none=[Q, K, V])
                    ms_e2e_triton = ms_fwd_triton + ms_bwd_triton
                except RuntimeError as e:
                    print(f"Triton kernel failed for this config: {e}")
                    ms_fwd_triton, ms_bwd_triton, ms_e2e_triton = float('inf'), float('inf'), float('inf')

                # PyTorch 常规注意力基准测试
                try:
                    ms_fwd_pt = triton.testing.do_bench(lambda: pytorch_attention_fwd(Q, K, V, is_causal=is_causal))
                    # 在后向传递基准测试中，每次都重新执行前向传递以创建新图
                    ms_bwd_pt = triton.testing.do_bench(lambda: pytorch_attention_fwd(Q, K, V, is_causal=is_causal).backward(torch.ones_like(Q)), grad_to_none=[Q, K, V])
                    ms_e2e_pt = ms_fwd_pt + ms_bwd_pt
                except RuntimeError as e:
                    print(f"PyTorch attention failed for this config: {e}")
                    ms_fwd_pt, ms_bwd_pt, ms_e2e_pt = float('inf'), float('inf'), float('inf')
                
                results.append({
                    'Seq Len': seq_len,
                    'Dim': dim,
                    'Dtype': str(dtype).split('.')[-1],
                    'Triton Fwd (ms)': f"{ms_fwd_triton:.3f}",
                    'Triton Bwd (ms)': f"{ms_bwd_triton:.3f}",
                    'Triton E2E (ms)': f"{ms_e2e_triton:.3f}",
                    'PyTorch Fwd (ms)': f"{ms_fwd_pt:.3f}",
                    'PyTorch Bwd (ms)': f"{ms_bwd_pt:.3f}",
                    'PyTorch E2E (ms)': f"{ms_e2e_pt:.3f}",
                })
    
    # 将结果打印到表格中
    if results:
        headers = results[0].keys()
        data = [list(res.values()) for res in results]
        print("\n--- Performance Benchmarking Results ---")
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        print("\nNo results to display. All configurations may have failed due to memory constraints.")

if __name__ == '__main__':
    run_benchmark()
