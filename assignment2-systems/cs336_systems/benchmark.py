import argparse
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    if device == 'cuda' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("使用 bfloat16 精度。")
        model.to(torch.bfloat16)

    print(f"\n模型: {args.model_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"评测模式: {args.mode}, 上下文长度: {args.block_size}")
    print(f"预热步数: {args.warmup_steps}, 测量步数: {args.measure_steps}")

    input_data = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)

    def step():
        if args.mode == 'forward':
            _ = model(input_data)
        elif args.mode == 'forward_backward':
            outputs = model(input_data)
            targets = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device=device)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
        else:
            raise ValueError(f"未知的模式: {args.mode}")

    print("\n正在进行预热...")
    for _ in range(args.warmup_steps):
        step()
        if device == 'cuda':
            torch.cuda.synchronize()
    print("预热完成。")

    print("\n开始正式测量...")
    timings = []
    for _ in range(args.measure_steps):
        start_time = timeit.default_timer()
        step()
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
    print("测量完成。")

    timings_ms = [t * 1000 for t in timings]
    mean_time = np.mean(timings_ms)
    std_dev = np.std(timings_ms)

    print("\n--- 评测结果 ---")
    print(f"平均每步耗时: {mean_time:.2f} ms")
    print(f"标准差: {std_dev:.2f} ms")
    print("----------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer 模型端到端性能评测脚本")
    
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
