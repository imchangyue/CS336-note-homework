import argparse
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os

# 新增：导入 NVTX
try:
    import torch.cuda.nvtx as nvtx
except ImportError:
    nvtx = None

# --- 0. 模型配置 ---
MODEL_CONFIGS = {
    "small": {"n_layers": 12, "n_heads": 12, "embed_dim": 768},
    "medium": {"n_layers": 24, "n_heads": 16, "embed_dim": 1024},
    "large": {"n_layers": 36, "n_heads": 20, "embed_dim": 1280},
    "xl": {"n_layers": 48, "n_heads": 25, "embed_dim": 1600},
    "2.7B": {"n_layers": 32, "n_heads": 32, "embed_dim": 2560},
}


# --- 1. 定义一个简单的 Transformer 模型 ---
class CausalSelfAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, block_size):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

    def forward(self, x):
        if nvtx: nvtx.range_push("CausalSelfAttention")
        B, T, C = x.size()
        
        if nvtx: nvtx.range_push("query_key_value_projection")
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        if nvtx: nvtx.range_pop()
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        if nvtx: nvtx.range_push("QK_multiplication")
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        if nvtx: nvtx.range_pop()
        
        if nvtx: nvtx.range_push("softmax_masking")
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if nvtx: nvtx.range_pop()
        
        if nvtx: nvtx.range_push("Attention_V_multiplication")
        y = att @ v
        if nvtx: nvtx.range_pop()
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        if nvtx: nvtx.range_push("c_proj_projection")
        y = self.c_proj(y)
        if nvtx: nvtx.range_pop()

        if nvtx: nvtx.range_pop()
        return y

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_heads, embed_dim, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(n_heads, embed_dim, block_size)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, embed_dim, block_size, vocab_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, embed_dim),
            wpe=nn.Embedding(block_size, embed_dim),
            h=nn.ModuleList([Block(n_heads, embed_dim, block_size) for _ in range(n_layers)]),
            ln_f=nn.LayerNorm(embed_dim),
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        if nvtx: nvtx.range_push("Transformer_Forward_Pass")
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if nvtx: nvtx.range_pop()

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None


def run_benchmark_and_return_results(args):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_config = MODEL_CONFIGS[args.model_size]
    try:
        model = Transformer(
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            embed_dim=model_config['embed_dim'],
            block_size=args.block_size,
            vocab_size=args.vocab_size
        ).to(device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return {
                "model_size": args.model_size,
                "mode": args.mode,
                "block_size": args.block_size,
                "batch_size": args.batch_size,
                "avg_time_ms": "OOM",
                "std_dev_ms": "OOM"
            }
        else:
            raise e
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    if device == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)

    input_data = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    def forward_only_pass():
        model.zero_grad()
        logits, _ = model(input_data)
    
    def full_training_step():
        optimizer.zero_grad()
        logits, loss = model(input_data, targets)
        loss.backward()
        optimizer.step()

    # 预热步骤
    for _ in range(5):
        if args.mode == 'forward':
            forward_only_pass()
        elif args.mode == 'full_training_step':
            full_training_step()
    
    if device == 'cuda':
        torch.cuda.synchronize()

    # 测量步骤
    timings = []
    for _ in range(10):
        start_time = timeit.default_timer()
        if args.mode == 'forward':
            forward_only_pass()
        elif args.mode == 'full_training_step':
            full_training_step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    avg_time_ms = np.mean(timings) * 1000
    std_dev_ms = np.std(timings) * 1000
    
    return {
        "model_size": args.model_size,
        "mode": args.mode,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "avg_time_ms": f"{avg_time_ms:.4f}",
        "std_dev_ms": f"{std_dev_ms:.4f}"
    }


def write_results_to_csv(results, output_file="results.csv"):
    """将结果写入CSV文件，支持追加模式"""
    if not results:
        return
    
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:  # 使用追加模式 'a'
        fieldnames = results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 如果文件不存在或为空，写入表头
        if not file_exists or os.path.getsize(output_file) == 0:
            writer.writeheader()
        
        # 写入所有结果
        writer.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer 模型端到端性能评测脚本")
    
    parser.add_argument('--model_size', type=str, choices=MODEL_CONFIGS.keys(),
                        help='要评测的模型尺寸 (可选，如果未指定则遍历所有)')
    parser.add_argument('--mode', type=str, default='full_training_step', 
                        choices=['forward', 'full_training_step'],
                        help='评测模式: 仅前向或完整训练步骤')
    
    # 评测超参数
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表大小')
    parser.add_argument('--block_size', type=int, default=1024, help='上下文长度 (序列长度) (可选，如果未指定则遍历所有)')
    
    # 新增参数：输出文件名和写入模式
    parser.add_argument('--output_file', type=str, default='results.csv', 
                        help='输出CSV文件名')
    parser.add_argument('--append', action='store_true', 
                        help='追加到现有CSV文件而不是覆盖')
    parser.add_argument('--write_per_result', action='store_true',
                        help='每完成一个测试就写入CSV文件（实时保存）')
    
    args = parser.parse_args()

    all_results = []
    model_sizes_to_run = [args.model_size] if args.model_size else MODEL_CONFIGS.keys()
    block_sizes_to_run = [args.block_size] if args.block_size else [128, 256, 512, 1024]
    
    # 如果不是追加模式且文件存在，创建备份或清空文件
    if not args.append and os.path.exists(args.output_file):
        if args.write_per_result:
            # 实时写入模式下，先清空文件
            open(args.output_file, 'w').close()
    
    print("正在开始评测...")
    for model_size in model_sizes_to_run:
        for block_size in block_sizes_to_run:
            # 确定批量大小
            batch_size = 4
            if model_size in ["xl", "2.7B"]:
                batch_size = 1
            
            print(f"正在评测: 模型={model_size}, 模式={args.mode}, 上下文长度={block_size}, 批量大小={batch_size}...")
            
            temp_args = argparse.Namespace(
                model_size=model_size,
                mode=args.mode,
                batch_size=batch_size,
                vocab_size=args.vocab_size,
                block_size=block_size
            )
            
            result = run_benchmark_and_return_results(temp_args)
            all_results.append(result)
            
            if result['avg_time_ms'] == 'OOM':
                print("由于内存不足，跳过该组合。")
            else:
                print(f"完成。平均耗时: {result['avg_time_ms']} ms")
            
            # 如果启用了实时写入，每完成一个测试就写入
            if args.write_per_result:
                write_results_to_csv([result], args.output_file)
    
    # 如果没有使用实时写入，则在最后一次性写入所有结果
    if not args.write_per_result:
        # 根据append参数决定写入模式
        if args.append:
            write_results_to_csv(all_results, args.output_file)
        else:
            # 覆盖模式
            if all_results:
                with open(args.output_file, 'w', newline='') as f:
                    fieldnames = all_results[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
    
    print(f"\n评测完成，所有结果已保存到 {args.output_file} 文件中。")