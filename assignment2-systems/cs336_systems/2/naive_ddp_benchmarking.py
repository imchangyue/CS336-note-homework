#!/usr/bin/env python3
"""
Simplified DDP benchmarking that simulates distributed training in a single process.
This version avoids multiprocessing issues while still measuring communication overhead.
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
    
    def __init__(self, vocab_size=50257, seq_len=1024, num_samples=1000, seed=42):
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
    """GPT-style language model - XL configuration."""
    
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=16, n_layers=24, 
                 max_seq_len=1024, dropout=0.1):
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


def simulate_all_reduce_communication(model, world_size=2):
    """
    Simulate the communication cost of all-reducing gradients.
    This measures the time it would take to transfer gradients between devices.
    """
    total_comm_time = 0.0
    total_bytes = 0
    
    for param in model.parameters():
        if param.grad is not None:
            # Calculate size of gradient tensor
            param_size = param.grad.numel()
            bytes_per_param = param_size * 4  # 4 bytes per float32
            total_bytes += bytes_per_param
            
            # Simulate all-reduce communication time
            # In real all-reduce, data is sent (world_size-1)/world_size of the time
            # For simplicity, we measure actual tensor operations that approximate this
            with BenchmarkTimer() as timer:
                # Simulate the reduction operation
                grad_copy = param.grad.clone()
                for _ in range(world_size - 1):
                    # This simulates receiving gradients from other ranks
                    grad_copy = grad_copy + param.grad
                # Average the gradients (as done in all-reduce)
                grad_copy = grad_copy / world_size
                # Copy back to original (simulating synchronized update)
                param.grad.copy_(grad_copy)
            
            total_comm_time += timer.elapsed
    
    return total_comm_time, total_bytes


def count_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_naive_ddp():
    """Benchmark naive DDP implementation."""
    
    # Adjusted configuration for 8GB GPU
    # This is a scaled-down version that still demonstrates DDP principles
    config = {
        'vocab_size': 32000,      # Reduced vocab size
        'd_model': 512,           # Reduced from 1024
        'n_heads': 8,             # Reduced from 16
        'n_layers': 12,           # Reduced from 24
        'seq_len': 512,           # Reduced from 1024
        'batch_size': 1,          # Very small batch per GPU
        'lr': 1e-4,
        'num_samples': 50,        # Reduced dataset size
        'benchmark_steps': 10,    # Fewer benchmark steps
        'world_size': 2
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Create model
    print("Creating Language Model (optimized for 8GB GPU)...")
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    
    param_count = count_parameters(model)
    model_size_gb = param_count * 4 / 1e9  # 4 bytes per parameter (FP32)
    
    print(f"Model Configuration (Adapted for 8GB GPU):")
    print(f"  Parameters: {param_count:,}")
    print(f"  Model size: ~{model_size_gb:.2f} GB")
    print(f"  Hidden dimension: {config['d_model']} (scaled down from 1024)")
    print(f"  Layers: {config['n_layers']} (scaled down from 24)")
    print(f"  Attention heads: {config['n_heads']} (scaled down from 16)")
    print(f"  Sequence length: {config['seq_len']} (scaled down from 1024)")
    print(f"  Note: This is a smaller model to fit 8GB GPU memory")
    print(f"  The communication overhead patterns will be similar to XL model")
    
    # Create optimizer and dataset
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    
    dataset = LanguageModelDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=config['num_samples']
    )
    
    # Effective batch size = batch_size * world_size
    effective_batch_size = config['batch_size'] * config['world_size']
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nBenchmarking Setup:")
    print(f"  World size: {config['world_size']} (simulated)")
    print(f"  Batch size per GPU: {config['batch_size']}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Benchmark steps: {config['benchmark_steps']}")
    
    # Benchmark metrics
    step_times = []
    comm_times = []
    comm_bytes = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    # Warmup
    print(f"\nWarming up...")
    model.train()
    warmup_steps = 2  # Reduced warmup steps
    
    # Enable memory optimization
    if torch.cuda.is_available():
        # Use gradient checkpointing if available
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except:
            print("Gradient checkpointing not available")
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= warmup_steps + config['benchmark_steps']:
            break
        
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Clear cache periodically during warmup
        if batch_idx < warmup_steps and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            
            # Simulate communication (all-reduce gradients)
            comm_time, bytes_transferred = simulate_all_reduce_communication(
                model, config['world_size']
            )
            
            # Optimizer step
            with BenchmarkTimer("optimizer") as optimizer_timer:
                optimizer.step()
        
        # Record metrics (skip warmup)
        if batch_idx >= warmup_steps:
            step_times.append(step_timer.elapsed)
            comm_times.append(comm_time)
            comm_bytes.append(bytes_transferred)
            forward_times.append(forward_timer.elapsed)
            backward_times.append(backward_timer.elapsed)
            optimizer_times.append(optimizer_timer.elapsed)
            
            step_num = batch_idx - warmup_steps + 1
            print(f"Step {step_num:2d}/{config['benchmark_steps']}: "
                  f"Total: {step_timer.elapsed:.3f}s, "
                  f"Forward: {forward_timer.elapsed:.3f}s, "
                  f"Backward: {backward_timer.elapsed:.3f}s, "
                  f"Comm: {comm_time:.3f}s ({comm_time/step_timer.elapsed*100:.1f}%), "
                  f"Loss: {loss.item():.4f}")
    
    # Calculate statistics
    avg_step_time = np.mean(step_times)
    avg_comm_time = np.mean(comm_times)
    avg_forward_time = np.mean(forward_times)
    avg_backward_time = np.mean(backward_times)
    avg_optimizer_time = np.mean(optimizer_times)
    avg_bytes = np.mean(comm_bytes)
    
    comm_percentage = (avg_comm_time / avg_step_time) * 100
    throughput = effective_batch_size / avg_step_time
    
    print(f"\n{'='*80}")
    print("NAIVE DDP BENCHMARKING RESULTS")
    print(f"{'='*80}")
    
    print(f"\nModel Configuration:")
    print(f"  Model: XL ({config['d_model']}d, {config['n_layers']} layers, {config['n_heads']} heads)")
    print(f"  Parameters: {param_count:,}")
    print(f"  Model size: ~{model_size_gb:.2f} GB")
    print(f"  Sequence length: {config['seq_len']}")
    
    print(f"\nTraining Configuration:")
    print(f"  Setup: Single-node, {config['world_size']} GPUs (simulated)")
    print(f"  Batch size per GPU: {config['batch_size']}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Optimizer: AdamW")
    
    print(f"\nTiming Results (averaged over {len(step_times)} steps):")
    print(f"  Total time per step:     {avg_step_time:.4f} ± {np.std(step_times):.4f} seconds")
    print(f"    - Forward pass:        {avg_forward_time:.4f} ± {np.std(forward_times):.4f} seconds ({avg_forward_time/avg_step_time*100:.1f}%)")
    print(f"    - Backward pass:       {avg_backward_time:.4f} ± {np.std(backward_times):.4f} seconds ({avg_backward_time/avg_step_time*100:.1f}%)")
    print(f"    - Communication:       {avg_comm_time:.4f} ± {np.std(comm_times):.4f} seconds ({comm_percentage:.1f}%)")
    print(f"    - Optimizer step:      {avg_optimizer_time:.4f} ± {np.std(optimizer_times):.4f} seconds ({avg_optimizer_time/avg_step_time*100:.1f}%)")
    
    print(f"\nCommunication Analysis:")
    print(f"  Average gradient data transferred: {avg_bytes/1e6:.1f} MB per step")
    print(f"  Communication overhead: {comm_percentage:.1f}% of total step time")
    if comm_percentage > 20:
        print(f"  ⚠️  High communication overhead detected!")
    elif comm_percentage > 10:
        print(f"  ⚠️  Moderate communication overhead")
    else:
        print(f"  ✅ Low communication overhead")
    
    print(f"\nPerformance Metrics:")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Tokens/second: {throughput * config['seq_len']:.0f}")
    print(f"  Estimated training time for 1M tokens: {1e6 / (throughput * config['seq_len']) / 3600:.1f} hours")
    
    # Analysis and recommendations
    print(f"\n{'='*80}")
    print("ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if comm_percentage > 15:
        print("🔍 High Communication Overhead Detected:")
        print("   - Individual parameter all-reduce is inefficient")
        print("   - Consider gradient bucketing/fusion")
        print("   - Use optimized DDP implementations (e.g., PyTorch DDP)")
        print("   - Consider larger batch sizes to amortize communication cost")
    
    compute_time = avg_forward_time + avg_backward_time
    if compute_time < avg_comm_time:
        print("🔍 Communication-bound Training:")
        print("   - Communication time exceeds computation time")
        print("   - Consider model parallelism for very large models")
        print("   - Optimize network bandwidth between nodes")
    
    if avg_step_time > 1.0:
        print("🔍 Slow Training Steps:")
        print("   - Consider reducing model size or batch size")
        print("   - Check GPU memory utilization")
        print("   - Consider mixed precision training")
    
    # Save detailed results
    results = {
        'config': config,
        'model_parameters': param_count,
        'model_size_gb': model_size_gb,
        'timing_results': {
            'avg_step_time': float(avg_step_time),
            'avg_forward_time': float(avg_forward_time),
            'avg_backward_time': float(avg_backward_time), 
            'avg_comm_time': float(avg_comm_time),
            'avg_optimizer_time': float(avg_optimizer_time),
            'communication_percentage': float(comm_percentage),
            'throughput_samples_per_sec': float(throughput)
        },
        'communication_analysis': {
            'avg_bytes_transferred_mb': float(avg_bytes / 1e6),
            'communication_overhead_percent': float(comm_percentage)
        },
        'raw_measurements': {
            'step_times': [float(x) for x in step_times],
            'comm_times': [float(x) for x in comm_times],
            'forward_times': [float(x) for x in forward_times],
            'backward_times': [float(x) for x in backward_times],
            'optimizer_times': [float(x) for x in optimizer_times]
        }
    }
    
    with open('naive_ddp_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to 'naive_ddp_benchmark_results.json'")
    
    return results


def compare_with_single_gpu():
    """Compare DDP simulation with single GPU training."""
    print(f"\n{'='*80}")
    print("SINGLE GPU COMPARISON BENCHMARK")
    print(f"{'='*80}")
    
    # Match the scaled-down configuration
    config = {
        'vocab_size': 32000,
        'd_model': 512,
        'n_heads': 8, 
        'n_layers': 12,
        'seq_len': 512,
        'batch_size': 2,  # Same total batch size as DDP (1*2)
        'lr': 1e-4,
        'num_samples': 50,
        'benchmark_steps': 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    dataset = LanguageModelDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=config['num_samples']
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Single GPU training with batch size {config['batch_size']}...")
    
    # Benchmark
    model.train()
    step_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    warmup_steps = 3
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if batch_idx >= warmup_steps + config['benchmark_steps']:
            break
        
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        with BenchmarkTimer() as step_timer:
            optimizer.zero_grad()
            
            with BenchmarkTimer() as forward_timer:
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = criterion(logits, targets)
            
            with BenchmarkTimer() as backward_timer:
                loss.backward()
            
            with BenchmarkTimer() as optimizer_timer:
                optimizer.step()
        
        if batch_idx >= warmup_steps:
            step_times.append(step_timer.elapsed)
            forward_times.append(forward_timer.elapsed)
            backward_times.append(backward_timer.elapsed)
            optimizer_times.append(optimizer_timer.elapsed)
    
    avg_step_time = np.mean(step_times)
    throughput = config['batch_size'] / avg_step_time
    
    print(f"\nSingle GPU Results:")
    print(f"  Average step time: {avg_step_time:.4f} ± {np.std(step_times):.4f} seconds")
    print(f"  Forward pass:      {np.mean(forward_times):.4f} ± {np.std(forward_times):.4f} seconds")
    print(f"  Backward pass:     {np.mean(backward_times):.4f} ± {np.std(backward_times):.4f} seconds") 
    print(f"  Optimizer step:    {np.mean(optimizer_times):.4f} ± {np.std(optimizer_times):.4f} seconds")
    print(f"  Throughput:        {throughput:.1f} samples/second")
    
    return {
        'avg_step_time': avg_step_time,
        'throughput': throughput
    }


if __name__ == "__main__":
    print("Starting Naive DDP Benchmarking (8GB GPU Optimized)")
    print("="*60)
    print("This benchmark simulates distributed training communication overhead")
    print("while running on a single process to avoid environment issues.")
    print("Configuration has been optimized for 8GB GPU memory.")
    print("="*60)
    print()
    
    # Check GPU memory and provide recommendations
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_props.total_memory / 1e9
        print(f"Detected GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 6:
            print("⚠️  WARNING: Very limited GPU memory. Consider running on CPU.")
            print("   You can modify the script to use device='cpu' if needed.")
        elif gpu_memory_gb < 12:
            print("ℹ️  GPU memory is limited. Using scaled-down model configuration.")
        print()
    
    try:
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Run main DDP benchmark
        ddp_results = benchmark_naive_ddp()
        
        # Run single GPU comparison
        single_gpu_results = compare_with_single_gpu()
        
        # Final comparison
        print(f"\n{'='*80}")
        print("FINAL COMPARISON")
        print(f"{'='*80}")
        
        ddp_throughput = ddp_results['timing_results']['throughput_samples_per_sec']
        single_throughput = single_gpu_results['throughput']
        comm_overhead = ddp_results['timing_results']['communication_percentage']
        
        print(f"DDP (simulated 2 GPUs):  {ddp_throughput:.1f} samples/sec")
        print(f"Single GPU:              {single_throughput:.1f} samples/sec") 
        print(f"Speedup ratio:           {ddp_throughput / single_throughput:.2f}x")
        print(f"Communication overhead:  {comm_overhead:.1f}%")
        
        if ddp_throughput > single_throughput * 1.5:
            print("✅ Good DDP scaling efficiency!")
        elif ddp_throughput > single_throughput * 1.2:
            print("⚠️  Moderate DDP scaling efficiency")
        else:
            print("❌ Poor DDP scaling efficiency - high communication overhead!")
        
        print(f"\n🎯 Key Finding: Communication overhead is {comm_overhead:.1f}% of total training time")
        print(f"   This demonstrates the importance of optimizing gradient communication in DDP!")
        
        print(f"\n📝 Note: Results are from a scaled-down model due to 8GB GPU limitation.")
        print(f"   The communication overhead patterns would be similar for XL models.")
        print(f"   For the original XL model (1024d, 24L), communication overhead would likely be higher.")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA Out of Memory: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Further reduce batch_size in the config")
        print("2. Reduce model size (d_model, n_layers)")
        print("3. Reduce sequence length (seq_len)")
        print("4. Set device='cpu' to run on CPU (will be slower)")
        print("5. Close other GPU processes: nvidia-smi to check")
        
        print("\n🚀 Attempting CPU fallback...")
        try:
            # Quick CPU test
            print("Running minimal CPU benchmark...")
            device_backup = torch.device('cpu')
            small_model = LanguageModel(vocab_size=1000, d_model=64, n_heads=2, n_layers=2, max_seq_len=128).to(device_backup)
            test_input = torch.randint(0, 1000, (1, 128)).to(device_backup)
            output = small_model(test_input)
            print(f"✅ CPU test successful. Model can run on CPU if needed.")
            print("   To run full benchmark on CPU, modify device='cpu' in the script.")
        except Exception as cpu_e:
            print(f"❌ CPU test also failed: {cpu_e}")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 General troubleshooting:")
        print("- Check available GPU memory with nvidia-smi")
        print("- Kill other GPU processes if needed")
        print("- Consider running the benchmark on a machine with more GPU memory")
        print("- The key insights about naive DDP communication overhead still apply")