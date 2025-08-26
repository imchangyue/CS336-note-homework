import argparse
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


# --- 2. 内存分析主函数 ---
def run_memory_profiling(args):
    """执行内存分析"""
    if not torch.cuda.is_available():
        raise RuntimeError("未找到 CUDA 设备，内存分析需要 GPU。")
    
    device = 'cuda'
    print("使用 GPU 进行内存分析。")

    # 从 args 获取模型配置
    model_config = MODEL_CONFIGS[args.model_size]
    model = Transformer(
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        embed_dim=model_config['embed_dim'],
        block_size=args.block_size,
        vocab_size=args.vocab_size
    ).to(device)

    # 此时，我们不再强制转换模型精度，它将保持为默认的 float32
    print("模型将使用默认的 float32 精度进行分析。")


    print(f"\n模型: {args.model_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"分析模式: {args.mode}, 上下文长度: {args.block_size}")

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

    # --- 内存分析核心代码 ---
    print("预热中...")
    for _ in range(args.warmup_steps):
        step()
        torch.cuda.synchronize()
    print("预热完成。")

    print("\n开始记录内存...")
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    step()

    output_filename = f"memory_snapshot/fp32/memory_snapshot_{args.model_size}_{args.mode}_{args.block_size}_fp32.pickle"
    torch.cuda.memory._dump_snapshot(output_filename)
    
    torch.cuda.memory._record_memory_history(enabled=None)
    
    print("内存记录完成。")
    print(f"内存快照已保存到文件: {output_filename}")
    print("请使用 https://pytorch.org/memory_viz/ 工具进行可视化分析。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer 模型内存分析脚本")
    
    parser.add_argument('--model_size', type=str, required=True, choices=MODEL_CONFIGS.keys(),
                        help='要分析的模型尺寸')
    parser.add_argument('--mode', type=str, default='forward_backward', choices=['forward', 'forward_backward'],
                        help='分析模式: 仅前向 (forward) 或前向+后向 (forward_backward)')

    # 评测超参数
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表大小')
    parser.add_argument('--block_size', type=int, default=1024, help='上下文长度 (序列长度)')
    parser.add_argument('--warmup_steps', type=int, default=5, help='预热步数')
    
    args = parser.parse_args()
    
    run_memory_profiling(args)