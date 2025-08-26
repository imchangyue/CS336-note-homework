import torch
import time
import pandas as pd
import gc

# 检查CUDA是否可用，如果不可用则退出
if not torch.cuda.is_available():
    print("CUDA is not available. This script requires a CUDA-enabled GPU.")
    exit()

# 固定批量大小
BATCH_SIZE = 8

# 定义要测试的维度
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]

# 创建一个DataFrame来存储结果
results = []

print("开始进行PyTorch注意力机制的基准测试...")
print("-" * 30)

# 确保在开始前释放所有未使用的显存
torch.cuda.empty_cache()

# 定义标准的注意力函数
def naive_attention(q, k, v):
    """
    一个简单的注意力机制实现，用于基准测试。
    q, k, v 的形状都为 [batch_size, seq_len, d_model]
    """
    # 计算注意力分数 Q*K^T
    # 结果形状为 [batch_size, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # 缩放
    d_k = q.size(-1)
    scores = scores / (d_k ** 0.5)
    
    # 应用softmax
    p_attn = torch.nn.functional.softmax(scores, dim=-1)
    
    # 与V相乘
    output = torch.matmul(p_attn, v)
    
    return output

# 迭代所有组合
for d_model in d_models:
    for seq_len in seq_lens:
        print(f"测试 d_model={d_model}, seq_len={seq_len}")
        
        # 将结果存储在字典中
        current_config = {
            'd_model': d_model,
            'seq_len': seq_len,
            'forward_time': 'N/A',
            'backward_time': 'N/A',
            'memory_usage_mb': 'N/A',
            'status': 'OK'
        }

        try:
            # 创建随机输入张量，并移动到GPU
            q = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", requires_grad=True)
            k = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", requires_grad=True)
            v = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", requires_grad=True)

            # --- 热身阶段 ---
            for _ in range(10):
                output = naive_attention(q, k, v)
                output.sum().backward()

            # --- 前向传播计时 ---
            torch.cuda.synchronize()
            start_fwd = time.time()
            for _ in range(100):
                output = naive_attention(q, k, v)
            torch.cuda.synchronize()
            end_fwd = time.time()
            
            # 计时
            current_config['forward_time'] = (end_fwd - start_fwd) / 100

            # 测量后向传播前的显存使用
            # 在这里，我们特意不释放显存，因为我们需要测量注意力分数矩阵的存储
            current_config['memory_usage_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # 梯度清零
            q.grad = None
            k.grad = None
            v.grad = None

            # --- 后向传播计时 ---
            torch.cuda.synchronize()
            start_bwd = time.time()
            for _ in range(100):
                # 再次执行前向传播，因为反向传播需要前向传播的计算图
                output = naive_attention(q, k, v)
                output.sum().backward()
            torch.cuda.synchronize()
            end_bwd = time.time()
            
            # 计时
            current_config['backward_time'] = (end_bwd - start_bwd) / 100

        except RuntimeError as e:
            if "out of memory" in str(e):
                current_config['status'] = 'OOM'
                print(f"    --> 显存不足 (OOM)")
                # 清理显存以继续下一个循环
                torch.cuda.empty_cache()
            else:
                current_config['status'] = f'Error: {e}'
                print(f"    --> 发生其他错误: {e}")
        
        # 释放张量，为下一个循环做准备
        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()
        
        results.append(current_config)

# 将结果转换为DataFrame并打印
df = pd.DataFrame(results)
print("\n基准测试结果表:")
print("-" * 30)
print(df.to_string())

# 获取OOM时最大的d_model和seq_len组合
oom_row = df[df['status'] == 'OOM'].iloc[0] if 'OOM' in df['status'].values else None
if oom_row is not None:
    oom_d_model = oom_row['d_model']
    oom_seq_len = oom_row['seq_len']
    
    print(f"\n显存不足发生在 d_model={oom_d_model}, seq_len={oom_seq_len} 的配置。")

    # 计算该配置下的内存使用情况
    B, T, D = BATCH_SIZE, oom_seq_len, oom_d_model
    
    # Q, K, V 的大小
    qkv_size = 3 * B * T * D * 4 # 4字节/float32
    # 注意力分数矩阵 S 的大小
    s_size = B * T * T * 4
    # 注意力输出矩阵 A 的大小
    a_size = B * T * D * 4
    
    # 前向传播需要的显存: Q, K, V, S, A
    fwd_memory = (qkv_size + s_size + a_size) / (1024 * 1024)
    # 后向传播额外需要的显存（用于保存S以计算梯度）
    bwd_memory = (s_size) / (1024 * 1024)
    
    print("\n对于这个 OOM 配置的内存使用估算:")
    print(f"  Q, K, V 张量所需内存: {qkv_size / (1024 * 1024):.2f} MB")
    print(f"  注意力分数矩阵 (S) 所需内存: {s_size / (1024 * 1024):.2f} MB")
    print(f"  注意力输出 (A) 所需内存: {a_size / (1024 * 1024):.2f} MB")
    print(f"  总前向传播所需显存: 约 {fwd_memory:.2f} MB")
    print(f"  总后向传播所需显存 (额外): 约 {bwd_memory:.2f} MB")

