#!/usr/bin/env python3
"""
Benchmarking script for comparing Naive DDP and Flattened Gradient DDP.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# 导入用于打包/解包的工具函数
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np

# --- 数据集和模型定义 (与之前相同，但增加了XL尺寸) ---

class ToyDataset(Dataset):
    """Simple dataset with random data for testing."""
    def __init__(self, size=2048, input_dim=512):
        self.size = size
        self.input_dim = input_dim
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class ToyModel(nn.Module):
    """
    Simple neural network for testing. 
    'XL' size to better demonstrate communication overhead.
    """
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=1, num_layers=8):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# --- DDP 辅助函数 ---

def init_process(rank, world_size, backend='gloo'):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()

# --- 两种梯度同步的实现 ---

def all_reduce_gradients_naive(model):
    """[原始方法] 逐个 All-reduce 梯度。"""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

def all_reduce_gradients_flat(model):
    """[优化方法] 打包梯度后进行单次 All-reduce。"""
    world_size = dist.get_world_size()
    
    # 1. 收集所有非空的梯度
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    if not grads:
        return

    # 2. 打包梯度到一个大的、一维的张量
    flat_grads = _flatten_dense_tensors(grads)
    
    # 3. 执行单次 All-reduce
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    
    # 4. 求平均
    flat_grads /= world_size
    
    # 5. 解包并将平均后的梯度复制回原位
    unflattened_grads = _unflatten_dense_tensors(flat_grads, grads)
    for grad, unflattened_grad in zip(grads, unflattened_grads):
        grad.copy_(unflattened_grad)

# --- 训练函数 ---

def train(rank, world_size, all_reduce_func, epochs=1, batch_size=64):
    """
    通用训练函数，接收一个梯度同步函数作为参数。
    """
    init_process(rank, world_size, backend='gloo')
    device = torch.device('cpu')
    
    # 为保证两种方法初始权重相同，固定随机种子
    torch.manual_seed(42 + rank) 
    
    model = ToyModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    criterion = nn.MSELoss()
    
    # --- 基准测试变量 ---
    # 使用 torch.cuda.Event 来精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_iter_time = 0.0
    total_comm_time = 0.0
    # 热身 (Warm-up)
    warmup_iters = 5
    
    model.train()
    
    iters_to_run = len(dataloader)
    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):
            if i >= iters_to_run: break
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # --- 计时开始 ---
            if i >= warmup_iters:
                start_event.record()

            # 调用传入的梯度同步函数
            all_reduce_func(model)

            if i >= warmup_iters:
                end_event.record()
                torch.cuda.synchronize() # 等待 GPU 操作完成
                comm_time = start_event.elapsed_time(end_event) / 1000.0 # 转换为秒
                total_comm_time += comm_time
            
            optimizer.step()
        
    # --- 收集并打印结果 ---
    total_iters = max(1, iters_to_run - warmup_iters)
    avg_comm_time = total_comm_time / total_iters if total_iters > 0 else 0
    
    # 使用 all_reduce 汇总所有 rank 的平均通信时间
    avg_comm_time_tensor = torch.tensor([avg_comm_time], device=device)
    dist.all_reduce(avg_comm_time_tensor, op=dist.ReduceOp.SUM)
    avg_comm_time_all_ranks = (avg_comm_time_tensor.item() / world_size) * 1000 # 转换为毫秒

    if rank == 0:
        # 测量单次迭代总时间 (这里简化为只测量通信时间，因为其他部分几乎一样)
        # 在实际场景中，会记录整个 iteration 的时间
        # 为了演示，我们假设总时间 = 计算时间 + 通信时间
        # 这里只报告通信时间
        func_name = all_reduce_func.__name__
        print(f"--- Method: {func_name} ---")
        print(f"Average time spent communicating gradients per iteration: {avg_comm_time_all_ranks:.4f} ms")
        print("-" * 30)
    
    cleanup()

if __name__ == "__main__":
    world_size = 1 # 使用 2 个 GPU
    if torch.cuda.device_count() < world_size:
        print(f"Need at least {world_size} GPUs to run this benchmark.")
        exit(1)

    print("Starting DDP benchmark...")
    print(f"World Size: {world_size}, Model: XL")
    
    print("\nRunning benchmark for Naive DDP (per-parameter all-reduce)...")
    mp.spawn(train, args=(world_size, all_reduce_gradients_naive), nprocs=world_size, join=True)
    
    print("\nRunning benchmark for Flattened Gradient DDP (single all-reduce)...")
    mp.spawn(train, args=(world_size, all_reduce_gradients_flat), nprocs=world_size, join=True)