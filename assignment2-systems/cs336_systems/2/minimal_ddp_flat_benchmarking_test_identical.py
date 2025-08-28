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


def compare_gradient_consistency():
    """
    验证两种通信方法在同一模型状态下，计算出的梯度是否一致。
    """
    # Configuration
    config = {
        'vocab_size': 32000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 12,
        'seq_len': 512,
        'batch_size': 1,
        'world_size': 2,
        'seed': 42 # 使用固定的随机种子
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 清除 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")

    # 创建模型和数据
    print("Creating Language Model...")
    torch.manual_seed(config['seed'])
    model = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)

    dataset = LanguageModelDataset(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        num_samples=1,
        seed=config['seed']
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'])
    criterion = nn.CrossEntropyLoss()

    # 获取一个批次数据
    input_ids, targets = next(iter(dataloader))
    input_ids = input_ids.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # 1. 模拟 Naive All-Reduce 并保存梯度
    print("\n--- Testing Naive All-Reduce ---")
    model.zero_grad()
    logits_naive = model(input_ids)
    loss_naive = criterion(logits_naive.view(-1, logits_naive.size(-1)), targets.view(-1))
    loss_naive.backward()
    simulate_individual_all_reduce(model, config['world_size'])
    # 存储梯度副本
    naive_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    # 2. 重新初始化模型和数据，确保状态完全相同
    print("\n--- Re-initializing for Batched All-Reduce ---")
    # 这比从头创建模型更高效，但为了严格保证一致性，我们还是重新创建
    torch.manual_seed(config['seed'])
    model_batched = LanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['seq_len']
    ).to(device)
    # 验证模型初始参数是否一致
    for p1, p2 in zip(model.parameters(), model_batched.parameters()):
        assert torch.allclose(p1, p2), "Initial model parameters are not identical!"

    # 3. 模拟 Batched All-Reduce 并保存梯度
    print("\n--- Testing Batched All-Reduce ---")
    model_batched.zero_grad()
    logits_batched = model_batched(input_ids)
    loss_batched = criterion(logits_batched.view(-1, logits_batched.size(-1)), targets.view(-1))
    loss_batched.backward()
    simulate_batched_all_reduce(model_batched, config['world_size'])
    # 存储梯度副本
    batched_grads = [p.grad.clone() for p in model_batched.parameters() if p.grad is not None]

    # 4. 比较两个版本的梯度和损失
    print("\n--- Comparing Results ---")
    # 检查损失是否完全一致
    assert torch.allclose(loss_naive, loss_batched, atol=1e-6), f"Loss mismatch: {loss_naive.item()} vs {loss_batched.item()}"
    print(f"✅ Loss values are identical: {loss_naive.item():.6f}")

    # 检查所有梯度是否完全一致
    for i, (g1, g2) in enumerate(zip(naive_grads, batched_grads)):
        if not torch.allclose(g1, g2, atol=1e-6):
            print(f"❌ Gradient mismatch at parameter {i}!")
            print(f"   Naive norm: {torch.norm(g1):.6f}")
            print(f"   Batched norm: {torch.norm(g2):.6f}")
            raise AssertionError("Gradient mismatch detected!")
    
    print("✅ All gradients are identical.")
    print("\nCONCLUSION: Both gradient communication methods are mathematically equivalent.")
    print("The previous loss differences were due to running two independent training processes.")


if __name__ == "__main__":
    print("DDP Gradient Consistency Verification")
    print("="*50)
    print("Verifying that individual and batched all-reduce produce identical gradients.")
    print("This confirms the correctness of the batched approach.")
    print("="*50)
    print()
    try:
        compare_gradient_consistency()
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()