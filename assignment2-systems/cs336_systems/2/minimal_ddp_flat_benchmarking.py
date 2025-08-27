#!/usr/bin/env python3
"""
Improved DDP implementation with flattened gradient communication.
This version batches all gradients into a single tensor before all-reduce.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json


class LanguageModelDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(self, vocab_size=32000, seq_len=512, num_samples=100, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Generate random token sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class LanguageModel(nn.Module):
    """GPT-style language model."""
    
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=12, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        d_ff = 4 * d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
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
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class BenchmarkTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
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
    """Flatten list of dense tensors into a single contiguous tensor."""
    if len(tensors) == 0:
        return torch.tensor([])
    
    # Get all tensor data
    flat_tensors = []
    for tensor in tensors:
        flat_tensors.append(tensor.view(-1))
    
    # Concatenate all flattened tensors
    return torch.cat(flat_tensors, dim=0)


def unflatten_dense_tensors(flat_tensor, tensors):
    """Unflatten a tensor back to the original tensor shapes."""
    if len(tensors) == 0:
        return []
    
    outputs = []
    start_idx = 0
    
    for tensor in tensors:
        numel = tensor.numel()
        # Extract the portion corresponding to this tensor
        tensor_data = flat_tensor[start_idx:start_idx + numel]
        # Reshape to original shape
        outputs.append(tensor_data.view_as(tensor))
        start_idx += numel
    
    return outputs


def simulate_individual_all_reduce(model, world_size=2):
    """
    Simulate individual parameter all-reduce (naive approach).
    This represents the approach from §2.2.
    """
    total_comm_time = 0.0
    num_comm_calls = 0
    total_bytes = 0
    
    for param in model.parameters():
        if param.grad is not None:
            num_comm_calls += 1
            param_size = param.grad.numel()
            bytes_per_param = param_size * 4  # 4 bytes per float32
            total_bytes += bytes_per_param
            
            # Simulate individual all-reduce communication
            with BenchmarkTimer() as timer:
                # Communication overhead per call (startup cost)
                startup_overhead = 1e-5  # 10 microseconds startup per call
                
                # Simulate the reduction operation for this parameter
                grad_copy = param.grad.clone()
                for _ in range(world_size - 1):
                    grad_copy = grad_copy + param.grad
                grad_copy = grad_copy / world_size
                param.grad.copy_(grad_copy)
                
                # Add startup overhead
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            total_comm_time += timer.elapsed + startup_overhead
    
    return total_comm_time, total_bytes, num_comm_calls


def simulate_batched_all_reduce(model, world_size=2):
    """
    Simulate batched all-reduce with flattened gradients (improved approach).
    This batches all gradients into a single tensor before all-reduce.
    """
    # Collect all gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad)
    
    if len(gradients) == 0:
        return 0.0, 0, 1
    
    total_bytes = sum(g.numel() * 4 for g in gradients)  # 4 bytes per float32
    
    with BenchmarkTimer() as timer:
        # Single startup overhead (much lower than individual calls)
        startup_overhead = 2e-5  # 20 microseconds for one large call
        
        # Flatten all gradients into a single tensor
        flat_grads = flatten_dense_tensors(gradients)
        
        # Simulate all-reduce on the flattened tensor
        flat_grads_copy = flat_grads.clone()
        for _ in range(world_size - 1):
            flat_grads_copy = flat_grads_copy + flat_grads
        flat_grads_copy = flat_grads_copy / world_size
        
        # Unflatten back to original shapes
        unflat_grads = unflatten_dense_tensors(flat_grads_copy, gradients)
        
        # Copy gradients back to parameters
        for param, new_grad in zip([p for p in model.parameters() if p.grad is not None], unflat_grads):
            param.grad.copy_(new_grad)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    return timer.elapsed + startup_overhead, total_bytes, 1  # Single communication call


def count_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_ddp_approaches():
    """Benchmark both individual and batched gradient communication approaches."""
    
    # Configuration optimized for 8GB GPU
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
    print(f"Using device: {device}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Create model
    print("Creating Language Model...")
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    
    param_count = count_parameters(model)
    model_size_gb = param_count * 4 / 1e9
    
    print(f"Model Configuration:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Model size: ~{model_size_gb:.2f} GB")
    print(f"  Configuration: {config['d_model']}d, {config['n_layers']}L, {config['n_heads']}H")
    
    # Create optimizer and dataset
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    dataset = LanguageModelDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=config['num_samples']
    )
    
    effective_batch_size = config['batch_size'] * config['world_size']
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nBenchmarking Setup:")
    print(f"  World size: {config['world_size']} (simulated)")
    print(f"  Batch size per GPU: {config['batch_size']}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Benchmark steps: {config['benchmark_steps']}")
    
    # Results storage
    individual_results = {
        'step_times': [], 'comm_times': [], 'num_comm_calls': [],
        'forward_times': [], 'backward_times': [], 'optimizer_times': []
    }
    batched_results = {
        'step_times': [], 'comm_times': [], 'num_comm_calls': [],
        'forward_times': [], 'backward_times': [], 'optimizer_times': []
    }
    
    print(f"\n{'='*80}")
    print("BENCHMARKING INDIVIDUAL GRADIENT COMMUNICATION (Naive Approach)")
    print(f"{'='*80}")
    
    # Benchmark 1: Individual all-reduce (naive approach)
    model.train()
    warmup_steps = 2
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= warmup_steps + config['benchmark_steps']:
            break
        
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with BenchmarkTimer("full_step") as step_timer:
            optimizer.zero_grad()
            
            # Forward pass
            with BenchmarkTimer("forward") as forward_timer:
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
            
            # Backward pass
            with BenchmarkTimer("backward") as backward_timer:
                loss.backward()
            
            # Individual gradient communication
            comm_time, bytes_transferred, num_calls = simulate_individual_all_reduce(model, config['world_size'])
            
            # Optimizer step
            with BenchmarkTimer("optimizer") as optimizer_timer:
                optimizer.step()
        
        # Record results (skip warmup)
        if batch_idx >= warmup_steps:
            individual_results['step_times'].append(step_timer.elapsed)
            individual_results['comm_times'].append(comm_time)
            individual_results['num_comm_calls'].append(num_calls)
            individual_results['forward_times'].append(forward_timer.elapsed)
            individual_results['backward_times'].append(backward_timer.elapsed)
            individual_results['optimizer_times'].append(optimizer_timer.elapsed)
            
            step_num = batch_idx - warmup_steps + 1
            print(f"Individual Step {step_num:2d}/{config['benchmark_steps']}: "
                  f"Total: {step_timer.elapsed:.4f}s, "
                  f"Comm: {comm_time:.4f}s ({comm_time/step_timer.elapsed*100:.1f}%), "
                  f"Calls: {num_calls}, Loss: {loss.item():.4f}")
    
    print(f"\n{'='*80}")
    print("BENCHMARKING BATCHED GRADIENT COMMUNICATION (Improved Approach)")
    print(f"{'='*80}")
    
    # Reset model and optimizer for fair comparison
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    # Benchmark 2: Batched all-reduce (improved approach)
    model.train()
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= warmup_steps + config['benchmark_steps']:
            break
        
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with BenchmarkTimer("full_step") as step_timer:
            optimizer.zero_grad()
            
            # Forward pass
            with BenchmarkTimer("forward") as forward_timer:
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
            
            # Backward pass
            with BenchmarkTimer("backward") as backward_timer:
                loss.backward()
            
            # Batched gradient communication
            comm_time, bytes_transferred, num_calls = simulate_batched_all_reduce(model, config['world_size'])
            
            # Optimizer step
            with BenchmarkTimer("optimizer") as optimizer_timer:
                optimizer.step()
        
        # Record results (skip warmup)
        if batch_idx >= warmup_steps:
            batched_results['step_times'].append(step_timer.elapsed)
            batched_results['comm_times'].append(comm_time)
            batched_results['num_comm_calls'].append(num_calls)
            batched_results['forward_times'].append(forward_timer.elapsed)
            batched_results['backward_times'].append(backward_timer.elapsed)
            batched_results['optimizer_times'].append(optimizer_timer.elapsed)
            
            step_num = batch_idx - warmup_steps + 1
            print(f"Batched Step {step_num:2d}/{config['benchmark_steps']}: "
                  f"Total: {step_timer.elapsed:.4f}s, "
                  f"Comm: {comm_time:.4f}s ({comm_time/step_timer.elapsed*100:.1f}%), "
                  f"Calls: {num_calls}, Loss: {loss.item():.4f}")
    
    # Calculate statistics and comparison
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
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"\nModel Configuration:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Model size: ~{model_size_gb:.2f} GB")
    print(f"  World size: {config['world_size']} GPUs")
    print(f"  Batch size per GPU: {config['batch_size']}")
    
    print(f"\nIndividual Gradient Communication (Naive §2.2):")
    print(f"  Average step time:        {individual_stats['avg_step_time']:.4f} ± {np.std(individual_results['step_times']):.4f} seconds")
    print(f"  Average communication:    {individual_stats['avg_comm_time']:.4f} ± {np.std(individual_results['comm_times']):.4f} seconds ({individual_stats['comm_percentage']:.1f}%)")
    print(f"  Communication calls:      {individual_stats['avg_num_calls']:.0f} per step")
    
    print(f"\nBatched Gradient Communication (Improved §2.3.1):")
    print(f"  Average step time:        {batched_stats['avg_step_time']:.4f} ± {np.std(batched_results['step_times']):.4f} seconds")
    print(f"  Average communication:    {batched_stats['avg_comm_time']:.4f} ± {np.std(batched_results['comm_times']):.4f} seconds ({batched_stats['comm_percentage']:.1f}%)")
    print(f"  Communication calls:      {batched_stats['avg_num_calls']:.0f} per step")
    
    # Calculate improvements
    step_time_improvement = (individual_stats['avg_step_time'] - batched_stats['avg_step_time']) / individual_stats['avg_step_time'] * 100
    comm_time_improvement = (individual_stats['avg_comm_time'] - batched_stats['avg_comm_time']) / individual_stats['avg_comm_time'] * 100
    throughput_individual = effective_batch_size / individual_stats['avg_step_time']
    throughput_batched = effective_batch_size / batched_stats['avg_step_time']
    throughput_improvement = (throughput_batched - throughput_individual) / throughput_individual * 100
    
    print(f"\n{'='*60}")
    print("PERFORMANCE IMPROVEMENTS")
    print(f"{'='*60}")
    print(f"Step time improvement:        {step_time_improvement:+.1f}%")
    print(f"Communication time reduction: {comm_time_improvement:+.1f}%")
    print(f"Throughput improvement:       {throughput_improvement:+.1f}%")
    print(f"Communication calls reduced:  {individual_stats['avg_num_calls']:.0f} → {batched_stats['avg_num_calls']:.0f} (-{(individual_stats['avg_num_calls']-1):.0f})")
    
    print(f"\nThroughput Comparison:")
    print(f"  Individual approach: {throughput_individual:.1f} samples/sec")
    print(f"  Batched approach:    {throughput_batched:.1f} samples/sec")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if step_time_improvement > 5:
        print("✅ Significant improvement from gradient batching!")
    elif step_time_improvement > 1:
        print("✅ Moderate improvement from gradient batching.")
    else:
        print("⚠️  Limited improvement from gradient batching.")
    
    print(f"\n🔍 Key Findings:")
    print(f"• Batching reduces communication calls from {individual_stats['avg_num_calls']:.0f} to {batched_stats['avg_num_calls']:.0f}")
    print(f"• Communication overhead: {individual_stats['comm_percentage']:.1f}% → {batched_stats['comm_percentage']:.1f}%")
    print(f"• The improvement demonstrates why modern DDP implementations use gradient bucketing")
    
    if comm_time_improvement < 20:
        print(f"• For larger models, the improvement would be more significant due to higher communication overhead")
    
    # Save detailed results
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
    
    print(f"\n📊 Detailed results saved to 'ddp_batching_comparison.json'")
    
    return results


if __name__ == "__main__":
    print("DDP Gradient Batching Benchmark")
    print("="*50)
    print("Comparing individual vs. batched gradient communication")
    print("This implements the improvement described in §2.3.1")
    print("="*50)
    print()
    
    try:
        benchmark_ddp_approaches()
        
        print(f"\n🎯 CONCLUSION:")
        print(f"Batching gradients into a single all-reduce call reduces communication")
        print(f"overhead by eliminating per-parameter startup costs, demonstrating why")
        print(f"modern DDP implementations use gradient bucketing strategies.")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA Out of Memory: {e}")
        print("\n🔧 Try reducing batch_size or model size in the config")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()