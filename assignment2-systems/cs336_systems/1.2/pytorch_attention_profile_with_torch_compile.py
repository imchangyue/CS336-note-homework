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

print("开始进行PyTorch注意力机制的基准测试 (包含 torch.compile 版本)...")
print("-" * 50)

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

# 使用 torch.compile 编译注意力函数
try:
    compiled_attention = torch.compile(naive_attention)
except Exception as e:
    print(f"警告: 无法编译注意力函数。可能需要更高版本的PyTorch或不同的环境。错误信息: {e}")
    compiled_attention = None
    
# 迭代所有组合
for d_model in d_models:
    for seq_len in seq_lens:
        print(f"测试 d_model={d_model}, seq_len={seq_len}")
        
        # 将结果存储在字典中
        current_config = {
            'd_model': d_model,
            'seq_len': seq_len,
            'uncompiled_forward_time_ms': 'N/A',
            'uncompiled_backward_time_ms': 'N/A',
            'compiled_forward_time_ms': 'N/A',
            'compiled_backward_time_ms': 'N/A',
            'status': 'OK'
        }

        try:
            # 创建随机输入张量，并移动到GPU
            q = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", requires_grad=True)
            k = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", requires_grad=True)
            v = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", requires_grad=True)

            # --- 热身阶段 ---
            # 对未编译和已编译版本都进行热身
            for _ in range(10):
                output_uncompiled = naive_attention(q, k, v)
                output_uncompiled.sum().backward()
                
                if compiled_attention:
                    output_compiled = compiled_attention(q, k, v)
                    output_compiled.sum().backward()
            
            # 释放梯度，准备计时
            q.grad = None
            k.grad = None
            v.grad = None

            # --- 未编译版本计时 ---
            torch.cuda.synchronize()
            start_fwd_uc = time.time()
            for _ in range(100):
                output_uncompiled = naive_attention(q, k, v)
            torch.cuda.synchronize()
            end_fwd_uc = time.time()
            current_config['uncompiled_forward_time_ms'] = (end_fwd_uc - start_fwd_uc) * 10

            start_bwd_uc = time.time()
            for _ in range(100):
                output_uncompiled = naive_attention(q, k, v)
                output_uncompiled.sum().backward()
            torch.cuda.synchronize()
            end_bwd_uc = time.time()
            current_config['uncompiled_backward_time_ms'] = (end_bwd_uc - start_bwd_uc) * 10
            
            # --- 已编译版本计时 ---
            if compiled_attention:
                q.grad = None
                k.grad = None
                v.grad = None
                
                torch.cuda.synchronize()
                start_fwd_c = time.time()
                for _ in range(100):
                    output_compiled = compiled_attention(q, k, v)
                torch.cuda.synchronize()
                end_fwd_c = time.time()
                current_config['compiled_forward_time_ms'] = (end_fwd_c - start_fwd_c) * 10
                
                start_bwd_c = time.time()
                for _ in range(100):
                    output_compiled = compiled_attention(q, k, v)
                    output_compiled.sum().backward()
                torch.cuda.synchronize()
                end_bwd_c = time.time()
                current_config['compiled_backward_time_ms'] = (end_bwd_c - start_bwd_c) * 10

        except RuntimeError as e:
            if "out of memory" in str(e):
                current_config['status'] = 'OOM'
                print(f"    --> 显存不足 (OOM)")
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
print("-" * 50)
print(df.to_string())

