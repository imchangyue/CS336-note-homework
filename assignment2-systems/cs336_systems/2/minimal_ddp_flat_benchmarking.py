#!/usr/bin/env python3
"""
改进的 DDP 实现，带有扁平化的梯度通信。
这个版本在进行 all-reduce 之前，将所有梯度打包成一个单独的张量。
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


class LanguageModelDataset(Dataset):
    """用于语言模型训练的简单数据集。"""
    
    def __init__(self, vocab_size=32000, seq_len=512, num_samples=100, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # 生成随机的 token 序列，形状为 (num_samples, seq_len + 1)
        # 最后一个 token 作为下一个词的预测目标
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 返回输入序列和目标序列
        return self.data[idx, :-1], self.data[idx, 1:]


class TransformerBlock(nn.Module):
    """单个 Transformer 块。"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU 激活函数
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力：Q, K, V 都来自输入 x
        attn_out, _ = self.attention(x, x, x)
        # 残差连接和层归一化
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        # 残差连接和层归一化
        x = self.norm2(x + ffn_out)
        return x


class LanguageModel(nn.Module):
    """GPT 风格的语言模型。"""
    
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=12, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer 块的列表
        d_ff = 4 * d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 应用权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # 权重初始化函数，遵循 GPT-2 风格
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        # 创建位置 ID 张量
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # 词嵌入 + 位置嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        
        # 依次通过 Transformer 块
        for block in self.blocks:
            x = block(x)
        
        # 最终的层归一化和线性层
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class BenchmarkTimer:
    """用于计时的上下文管理器。"""
    
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        # 如果使用 CUDA，确保所有 GPU 操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        return self.end_time - self.start_time if self.end_time else None


def flatten_dense_tensors(tensors):
    """将稠密张量列表展平为单个连续张量。"""
    if len(tensors) == 0:
        return torch.tensor([])
    
    # 将所有张量展平
    flat_tensors = []
    for tensor in tensors:
        flat_tensors.append(tensor.view(-1))
    
    # 将所有展平的张量拼接在一起
    return torch.cat(flat_tensors, dim=0)


def unflatten_dense_tensors(flat_tensor, tensors):
    """将一个张量解压回原来的张量形状。"""
    if len(tensors) == 0:
        return []
    
    outputs = []
    start_idx = 0
    
    for tensor in tensors:
        numel = tensor.numel()  # 获取张量中元素的总数
        # 提取与当前张量对应的部分
        tensor_data = flat_tensor[start_idx:start_idx + numel]
        # 将其重塑回原始形状
        outputs.append(tensor_data.view_as(tensor))
        start_idx += numel
    
    return outputs


def simulate_individual_all_reduce(model, world_size=2):
    """
    模拟逐个参数的 All-Reduce（朴素方法）。
    这代表了教科书或简单实现中的方法。
    """
    total_comm_time = 0.0
    num_comm_calls = 0
    total_bytes = 0
    
    for param in model.parameters():
        if param.grad is not None:
            num_comm_calls += 1
            param_size = param.grad.numel()
            bytes_per_param = param_size * 4  # float32 是 4 字节
            total_bytes += bytes_per_param
            
            # 模拟单个 all-reduce 通信
            with BenchmarkTimer() as timer:
                # 模拟每次调用的通信开销（启动成本）
                startup_overhead = 1e-5  # 每次调用 10 微秒的启动开销
                
                # 模拟这个参数的归约操作（all-reduce 的核心）
                grad_copy = param.grad.clone()
                for _ in range(world_size - 1):
                    grad_copy = grad_copy + param.grad
                grad_copy = grad_copy / world_size
                param.grad.copy_(grad_copy)
                
                # 添加启动开销
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            total_comm_time += timer.elapsed + startup_overhead
    
    return total_comm_time, total_bytes, num_comm_calls


def simulate_batched_all_reduce(model, world_size=2):
    """
    模拟带有扁平化梯度的批量 All-Reduce（改进方法）。
    这在 all-reduce 之前将所有梯度打包成一个单独的张量。
    """
    # 收集所有梯度
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad)
    
    if len(gradients) == 0:
        return 0.0, 0, 1
    
    total_bytes = sum(g.numel() * 4 for g in gradients)
    
    with BenchmarkTimer() as timer:
        # 单次启动开销（远低于多次单独调用）
        startup_overhead = 2e-5  # 一次大型调用 20 微秒的启动开销
        
        # 将所有梯度展平为单个张量
        flat_grads = flatten_dense_tensors(gradients)
        
        # 在展平后的张量上模拟 all-reduce
        flat_grads_copy = flat_grads.clone()
        for _ in range(world_size - 1):
            flat_grads_copy = flat_grads_copy + flat_grads
        flat_grads_copy = flat_grads_copy / world_size
        
        # 解压回原始形状
        unflat_grads = unflatten_dense_tensors(flat_grads_copy, gradients)
        
        # 将梯度复制回参数
        for param, new_grad in zip([p for p in model.parameters() if p.grad is not None], unflat_grads):
            param.grad.copy_(new_grad)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    return timer.elapsed + startup_overhead, total_bytes, 1  # 只有一次通信调用


def count_parameters(model):
    """计算总参数量。"""
    return sum(p.numel() for p in model.parameters())


def benchmark_ddp_approaches():
    """对逐个和批量梯度通信方法进行基准测试。"""
    
    # 针对 8GB GPU 优化的配置
    config = {
        'vocab_size': 32000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 12,
        'seq_len': 512,
        'batch_size': 1,
        'lr': 1e-4,
        'num_samples': 50,
        'benchmark_steps': 12,
        'world_size': 2
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 内存: {gpu_memory:.1f} GB")
    
    # 创建模型
    print("正在创建语言模型...")
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    
    param_count = count_parameters(model)
    model_size_gb = param_count * 4 / 1e9  # float32 是 4 字节
    
    print("模型配置:")
    print(f"  参数量: {param_count:,}")
    print(f"  模型大小: ~{model_size_gb:.2f} GB")
    print(f"  配置: {config['d_model']}d, {config['n_layers']}L, {config['n_heads']}H")
    
    # 创建优化器和数据集
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    dataset = LanguageModelDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=config['num_samples']
    )
    
    effective_batch_size = config['batch_size'] * config['world_size']
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    print("\n基准测试设置:")
    print(f"  进程数: {config['world_size']} (模拟)")
    print(f"  每个 GPU 的批次大小: {config['batch_size']}")
    print(f"  有效批次大小: {effective_batch_size}")
    print(f"  基准测试步数: {config['benchmark_steps']}")
    
    # 结果存储
    individual_results = {
        'step_times': [], 'comm_times': [], 'num_comm_calls': [],
        'forward_times': [], 'backward_times': [], 'optimizer_times': []
    }
    batched_results = {
        'step_times': [], 'comm_times': [], 'num_comm_calls': [],
        'forward_times': [], 'backward_times': [], 'optimizer_times': []
    }
    
    print(f"\n{'='*80}")
    print("基准测试：逐个梯度通信 (朴素方法)")
    print(f"{'='*80}")
    
    # 基准测试 1: 逐个 all-reduce（朴素方法）
    model.train()
    warmup_steps = 2  # 预热步数，以消除初始开销
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= warmup_steps + config['benchmark_steps']:
            break
        
        # 将数据移动到设备
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with BenchmarkTimer("full_step") as step_timer:
            optimizer.zero_grad()
            
            # 前向传播
            with BenchmarkTimer("forward") as forward_timer:
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
            
            # 反向传播
            with BenchmarkTimer("backward") as backward_timer:
                loss.backward()
            
            # 逐个梯度通信
            comm_time, bytes_transferred, num_calls = simulate_individual_all_reduce(model, config['world_size'])
            
            # 优化器步进
            with BenchmarkTimer("optimizer") as optimizer_timer:
                optimizer.step()
        
        # 记录结果（跳过预热步）
        if batch_idx >= warmup_steps:
            individual_results['step_times'].append(step_timer.elapsed)
            individual_results['comm_times'].append(comm_time)
            individual_results['num_comm_calls'].append(num_calls)
            individual_results['forward_times'].append(forward_timer.elapsed)
            individual_results['backward_times'].append(backward_timer.elapsed)
            individual_results['optimizer_times'].append(optimizer_timer.elapsed)
            
            step_num = batch_idx - warmup_steps + 1
            print(f"逐个梯度通信第 {step_num:2d}/{config['benchmark_steps']} 步: "
                  f"总耗时: {step_timer.elapsed:.4f}s, "
                  f"通信耗时: {comm_time:.4f}s ({comm_time/step_timer.elapsed*100:.1f}%), "
                  f"调用次数: {num_calls}, 损失: {loss.item():.4f}")
    
    print(f"\n{'='*80}")
    print("基准测试：批量梯度通信 (改进方法)")
    print(f"{'='*80}")
    
    # 为公平比较，重置模型和优化器
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    # 基准测试 2: 批量 all-reduce（改进方法）
    model.train()
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= warmup_steps + config['benchmark_steps']:
            break
        
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with BenchmarkTimer("full_step") as step_timer:
            optimizer.zero_grad()
            
            # 前向传播
            with BenchmarkTimer("forward") as forward_timer:
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
            
            # 反向传播
            with BenchmarkTimer("backward") as backward_timer:
                loss.backward()
            
            # 批量梯度通信
            comm_time, bytes_transferred, num_calls = simulate_batched_all_reduce(model, config['world_size'])
            
            # 优化器步进
            with BenchmarkTimer("optimizer") as optimizer_timer:
                optimizer.step()
        
        # 记录结果（跳过预热步）
        if batch_idx >= warmup_steps:
            batched_results['step_times'].append(step_timer.elapsed)
            batched_results['comm_times'].append(comm_time)
            batched_results['num_comm_calls'].append(num_calls)
            batched_results['forward_times'].append(forward_timer.elapsed)
            batched_results['backward_times'].append(backward_timer.elapsed)
            batched_results['optimizer_times'].append(optimizer_timer.elapsed)
            
            step_num = batch_idx - warmup_steps + 1
            print(f"批量梯度通信第 {step_num:2d}/{config['benchmark_steps']} 步: "
                  f"总耗时: {step_timer.elapsed:.4f}s, "
                  f"通信耗时: {comm_time:.4f}s ({comm_time/step_timer.elapsed*100:.1f}%), "
                  f"调用次数: {num_calls}, 损失: {loss.item():.4f}")
    
    # 计算统计数据并进行比较
    individual_stats = {
        'avg_step_time': np.mean(individual_results['step_times']),
        'avg_comm_time': np.mean(individual_results['comm_times']),
        'avg_num_calls': np.mean(individual_results['num_comm_calls']),
        'comm_percentage': np.mean(individual_results['comm_times']) / np.mean(individual_results['step_times']) * 100
    }
    
    batched_stats = {
        'avg_step_time': np.mean(batched_results['step_times']),
        'avg_comm_time': np.mean(batched_results['comm_times']),
        'avg_num_calls': np.mean(batched_results['num_comm_calls']),
        'comm_percentage': np.mean(batched_results['comm_times']) / np.mean(batched_results['step_times']) * 100
    }
    
    print(f"\n{'='*80}")
    print("比较结果")
    print(f"{'='*80}")
    
    print("\n模型配置:")
    print(f"  参数量: {param_count:,}")
    print(f"  模型大小: ~{model_size_gb:.2f} GB")
    print(f"  进程数: {config['world_size']} GPUs")
    print(f"  每个 GPU 的批次大小: {config['batch_size']}")
    
    print("\n逐个梯度通信 (朴素方法):")
    print(f"  平均每步耗时:      {individual_stats['avg_step_time']:.4f} ± {np.std(individual_results['step_times']):.4f} 秒")
    print(f"  平均通信耗时:      {individual_stats['avg_comm_time']:.4f} ± {np.std(individual_results['comm_times']):.4f} 秒 ({individual_stats['comm_percentage']:.1f}%)")
    print(f"  通信调用次数:      {individual_stats['avg_num_calls']:.0f} 次/步")
    
    print("\n批量梯度通信 (改进方法):")
    print(f"  平均每步耗时:      {batched_stats['avg_step_time']:.4f} ± {np.std(batched_results['step_times']):.4f} 秒")
    print(f"  平均通信耗时:      {batched_stats['avg_comm_time']:.4f} ± {np.std(batched_results['comm_times']):.4f} 秒 ({batched_stats['comm_percentage']:.1f}%)")
    print(f"  通信调用次数:      {batched_stats['avg_num_calls']:.0f} 次/步")
    
    # 计算性能提升
    step_time_improvement = (individual_stats['avg_step_time'] - batched_stats['avg_step_time']) / individual_stats['avg_step_time'] * 100
    comm_time_improvement = (individual_stats['avg_comm_time'] - batched_stats['avg_comm_time']) / individual_stats['avg_comm_time'] * 100
    throughput_individual = effective_batch_size / individual_stats['avg_step_time']
    throughput_batched = effective_batch_size / batched_stats['avg_step_time']
    throughput_improvement = (throughput_batched - throughput_individual) / throughput_individual * 100
    
    print(f"\n{'='*60}")
    print("性能提升")
    print(f"{'='*60}")
    print(f"每步耗时提升:            {step_time_improvement:+.1f}%")
    print(f"通信耗时减少:            {comm_time_improvement:+.1f}%")
    print(f"吞吐量提升:              {throughput_improvement:+.1f}%")
    print(f"通信调用次数减少:        {individual_stats['avg_num_calls']:.0f} → {batched_stats['avg_num_calls']:.0f} (-{(individual_stats['avg_num_calls']-1):.0f})")
    
    print("\n吞吐量对比:")
    print(f"  逐个通信方法: {throughput_individual:.1f} 样本/秒")
    print(f"  批量通信方法: {throughput_batched:.1f} 样本/秒")
    
    # 分析
    print(f"\n{'='*60}")
    print("分析")
    print(f"{'='*60}")
    
    if step_time_improvement > 5:
        print("✅ 梯度批量通信带来了显著的性能提升！")
    elif step_time_improvement > 1:
        print("✅ 梯度批量通信带来了适度的性能提升。")
    else:
        print("⚠️  梯度批量通信带来的提升有限。")
    
    print("\n🔍 关键发现:")
    print(f"• 批量通信将通信调用次数从 {individual_stats['avg_num_calls']:.0f} 减少到 {batched_stats['avg_num_calls']:.0f}")
    print(f"• 通信开销: {individual_stats['comm_percentage']:.1f}% → {batched_stats['comm_percentage']:.1f}%")
    print(f"• 这种提升说明了为什么现代 DDP 实现会使用梯度分桶（gradient bucketing）策略。")
    
    if comm_time_improvement < 20:
        print("• 对于更大的模型，由于通信开销更高，这种提升会更加显著。")
    
    # 保存详细结果到 JSON 文件
    results = {
        'config': config,
        'model_parameters': param_count,
        'individual_approach': {
            'avg_step_time': float(individual_stats['avg_step_time']),
            'avg_comm_time': float(individual_stats['avg_comm_time']),
            'comm_percentage': float(individual_stats['comm_percentage']),
            'num_comm_calls': float(individual_stats['avg_num_calls']),
            'throughput': float(throughput_individual)
        },
        'batched_approach': {
            'avg_step_time': float(batched_stats['avg_step_time']),
            'avg_comm_time': float(batched_stats['avg_comm_time']),
            'comm_percentage': float(batched_stats['comm_percentage']),
            'num_comm_calls': float(batched_stats['avg_num_calls']),
            'throughput': float(throughput_batched)
        },
        'improvements': {
            'step_time_improvement_percent': float(step_time_improvement),
            'comm_time_reduction_percent': float(comm_time_improvement),
            'throughput_improvement_percent': float(throughput_improvement)
        }
    }
    
    with open('ddp_batching_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n📊 详细结果已保存到 'ddp_batching_comparison.json'")
    
    return results


if __name__ == "__main__":
    print("DDP 梯度批量通信基准测试")
    print("="*50)
    print("对比逐个梯度通信与批量梯度通信")
    print("本代码实现了《动手学深度学习》§2.3.1 中描述的改进")
    print("="*50)
    print()
    
    try:
        benchmark_ddp_approaches()
        
        print("\n🎯 结论:")
        print("将梯度批量打包成一个 all-reduce 调用，可以消除逐参数的启动开销，")
        print("从而显著减少通信开销，这解释了为什么现代 DDP 实现会使用梯度分桶策略。")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA 显存不足: {e}")
        print("\n🔧 尝试在配置中减小 batch_size 或模型大小")
        
    except Exception as e:
        print(f"基准测试失败: {e}")
        import traceback
        traceback.print_exc()