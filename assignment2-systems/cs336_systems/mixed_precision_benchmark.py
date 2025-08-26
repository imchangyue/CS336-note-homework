import argparse
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

# --- 0. 模型配置 ---
# 根据题目要求定义的模型尺寸
MODEL_CONFIGS = {
    "small": {"n_layers": 12, "n_heads": 12, "embed_dim": 768},
    "medium": {"n_layers": 24, "n_heads": 16, "embed_dim": 1024},
    "large": {"n_layers": 36, "n_heads": 20, "embed_dim": 1280},
    "xl": {"n_layers": 48, "n_heads": 25, "embed_dim": 1600},
    "2.7B": {"n_layers": 32, "n_heads": 32, "embed_dim": 2560},
}


# --- 1. 定义一个简单的 Transformer 模型 ---
# 模型结构与之前相同

class CausalSelfAttention(nn.Module):
    """一个简单的多头因果自注意力模块"""
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
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """一个简单的 MLP 模块 (d_ff = 4 * d_model)"""
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc    = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    """一个 Transformer Block"""
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
    """完整的 Transformer 模型"""
    def __init__(self, n_layers, n_heads, embed_dim, block_size, vocab_size):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(block_size, embed_dim),
            h = nn.ModuleList([Block(n_heads, embed_dim, block_size) for _ in range(n_layers)]),
            ln_f = nn.LayerNorm(embed_dim),
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
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
        return logits

# --- 2. 基准测试主函数 ---

def run_single_precision_benchmark(model, input_data, targets, args, device, precision_name, autocast_context, use_scaler=False):
    """运行单个精度配置的基准测试"""
    print(f"\n--- {precision_name} ---")
    
    # 创建梯度缩放器（用于混合精度训练）
    scaler = torch.amp.GradScaler('cuda') if use_scaler and device == 'cuda' else None
    
    # 创建临时优化器（仅在需要scaler时使用）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if scaler else None
    
    def step():
        if args.mode == 'forward':
            with autocast_context:
                outputs = model(input_data)
                # 打印一些张量的数据类型以展示混合精度（仅第一次）
                if precision_name == "BF16 (Mixed Precision)" and hasattr(outputs, 'dtype'):
                    print(f"输出张量类型: {outputs.dtype}")
        elif args.mode == 'forward_backward':
            with autocast_context:
                outputs = model(input_data)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                # 展示混合精度：输出和loss的数据类型（仅第一次）
                if precision_name == "BF16 (Mixed Precision)" and hasattr(outputs, 'dtype'):
                    print(f"模型输出类型: {outputs.dtype}, Loss类型: {loss.dtype}")
            
            if scaler:
                # 使用梯度缩放（混合精度训练的标准做法）
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
        else:
            raise ValueError(f"未知的模式: {args.mode}")

    # 预热
    print(f"正在进行预热 ({precision_name})...")
    for _ in range(args.warmup_steps):
        step()
        if device == 'cuda':
            torch.cuda.synchronize()
        # 清除梯度，避免累积
        if not scaler:
            model.zero_grad()

    # 测量
    print(f"开始正式测量 ({precision_name})...")
    timings = []
    for _ in range(args.measure_steps):
        start_time = timeit.default_timer()
        step()
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
        # 清除梯度，避免累积
        if not scaler:
            model.zero_grad()

    timings_ms = [t * 1000 for t in timings]
    mean_time = np.mean(timings_ms)
    std_dev = np.std(timings_ms)

    print(f"平均每步耗时: {mean_time:.2f} ms")
    print(f"标准差: {std_dev:.2f} ms")
    
    return mean_time, std_dev

def run_benchmark(args):
    """执行基准测试"""
    if torch.cuda.is_available():
        device = 'cuda'
        print("使用 GPU 进行评测。")
    else:
        device = 'cpu'
        print("警告：未找到 CUDA 设备，将使用 CPU 进行评测。")

    # 从 args 获取模型配置
    model_config = MODEL_CONFIGS[args.model_size]
    model = Transformer(
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        embed_dim=model_config['embed_dim'],
        block_size=args.block_size,
        vocab_size=args.vocab_size
    ).to(device)

    print(f"\n模型: {args.model_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"评测模式: {args.mode}, 上下文长度: {args.block_size}")
    print(f"预热步数: {args.warmup_steps}, 测量步数: {args.measure_steps}")

    input_data = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    # 检查是否支持 BF16
    use_bf16 = device == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not use_bf16:
        print("警告：当前设备不支持 BF16，将跳过混合精度测试。")

    # 运行 FP32 (Full Precision) 基准测试
    print("\n=== 开始 FP32 基准测试 ===")
    fp32_context = nullcontext()
    fp32_mean, fp32_std = run_single_precision_benchmark(
        model, input_data, targets, args, device, "FP32 (Full Precision)", fp32_context, use_scaler=False
    )

    # 运行 BF16 Mixed Precision 基准测试 (如果支持)
    if use_bf16:
        print("\n=== 开始 BF16 混合精度基准测试 ===")
        bf16_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
        bf16_mean, bf16_std = run_single_precision_benchmark(
            model, input_data, targets, args, device, "BF16 (Mixed Precision)", bf16_context, use_scaler=True
        )

        # 计算加速比
        speedup = fp32_mean / bf16_mean
        print(f"\n=== 比较结果 ===")
        print(f"FP32 平均耗时: {fp32_mean:.2f} ms")
        print(f"BF16 平均耗时: {bf16_mean:.2f} ms")
        print(f"混合精度加速比: {speedup:.2f}x")
        print("=================")
    else:
        print("\n=== 评测结果 ===")
        print(f"FP32 平均每步耗时: {fp32_mean:.2f} ms")
        print(f"标准差: {fp32_std:.2f} ms")
        print("================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer 模型端到端性能评测脚本（支持混合精度）")
    
    parser.add_argument('--model_size', type=str, required=True, choices=MODEL_CONFIGS.keys(),
                        help='要评测的模型尺寸')
    parser.add_argument('--mode', type=str, default='forward_backward', choices=['forward', 'forward_backward'],
                        help='评测模式: 仅前向或前向+后向')
    
    # 评测超参数
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表大小')
    parser.add_argument('--block_size', type=int, default=1024, help='上下文长度 (序列长度)')
    parser.add_argument('--warmup_steps', type=int, default=5, help='预热步数')
    parser.add_argument('--measure_steps', type=int, default=10, help='测量步数')

    args = parser.parse_args()
    run_benchmark(args)