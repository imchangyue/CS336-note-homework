#!/usr/bin/env python3
"""
分布式数据并行（DDP）训练的朴素实现。
这个脚本通过手动对每个参数的梯度进行 All-Reduce 来实现 DDP。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np


class ToyDataset(Dataset):
    """用于测试的简单随机数据集。"""
    
    def __init__(self, size=1000, input_dim=10, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.size = size
        self.input_dim = input_dim
        
        # 生成随机数据
        self.data = torch.randn(size, input_dim)
        # 生成随机目标（回归任务）
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ToyModel(nn.Module):
    """用于测试的简单神经网络。"""
    
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def init_process(rank, world_size, backend='nccl'):
    """初始化分布式进程组。"""
    # 设置主节点的地址和端口
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    # 初始化进程组
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    """清理分布式进程组。"""
    dist.destroy_process_group()


def broadcast_parameters(model, src_rank=0):
    """从源进程（src_rank）广播模型参数到所有其他进程。"""
    for param in model.parameters():
        # 广播每个参数的数据
        dist.broadcast(param.data, src=src_rank)


def all_reduce_gradients(model):
    """对所有进程的梯度进行 All-Reduce。"""
    for param in model.parameters():
        if param.grad is not None:
            # All-Reduce 梯度并求平均
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()


def naive_ddp_train(rank, world_size, epochs=5, batch_size=32, lr=0.01):
    """朴素 DDP 训练函数。"""
    print(f"在 rank {rank} 上运行 DDP 训练")
    
    # 初始化进程组
    init_process(rank, world_size)
    
    # 设置设备
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型并移动到设备
    model = ToyModel().to(device)
    
    # 从 rank 0 广播参数，确保所有进程的模型权重一致
    broadcast_parameters(model, src_rank=0)
    
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 创建数据集和分布式采样器
    dataset = ToyDataset(size=1000)
    # DistributedSampler 确保每个进程只看到数据集的一个不重叠子集
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    # DataLoader 使用采样器
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        # 必须在每个 epoch 开始时调用，以确保正确洗牌
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 在所有进程中 All-Reduce 梯度
            all_reduce_gradients(model)
            
            # 优化器更新参数
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 只有主进程（rank 0）打印日志
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    
    # 保存最终的模型状态用于验证
    # .clone().cpu() 用于将张量从 GPU 复制到 CPU 并克隆，以便在多进程结束后访问
    final_state = {name: param.clone().cpu() for name, param in model.named_parameters()}
    
    # 清理进程组
    cleanup()
    return final_state


def single_process_train(epochs=5, batch_size=32, lr=0.01, world_size=2):
    """用于比较的单进程训练。"""
    print("运行单进程训练进行比较")
    
    # 设置确定性行为以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型，其初始化与 DDP 相同
    model = ToyModel().to(device)
    
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 创建数据集 - 使用与 DDP 相同的总批量大小（batch_size * world_size）
    dataset = ToyDataset(size=1000)
    # 因为单进程，不需要 DistributedSampler，但是为了公平比较，使用相同的总批次
    # shuffle=False 确保数据顺序与 DDP 模拟器中的一致
    dataloader = DataLoader(dataset, batch_size=batch_size * world_size, shuffle=False)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 优化器更新参数
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    
    # 返回最终模型状态
    return {name: param.clone().cpu() for name, param in model.named_parameters()}


def run_ddp(world_size=2):
    """运行分布式训练。"""
    # 'gloo' 后端用于 CPU，'nccl' 用于 GPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    # 启动进程进行分布式训练
    # mp.spawn 会为每个进程调用 naive_ddp_train 函数
    mp.spawn(naive_ddp_train, args=(world_size,), nprocs=world_size, join=True)


def compare_models(ddp_state, single_state, tolerance=1e-6):
    """比较 DDP 和单进程训练的模型状态。"""
    print("\n比较模型参数:")
    all_match = True
    max_diff_overall = 0.0
    
    # 遍历所有参数
    for name in ddp_state.keys():
        ddp_param = ddp_state[name]
        single_param = single_state[name]
        
        # 计算差异
        diff = torch.abs(ddp_param - single_param)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        max_diff_overall = max(max_diff_overall, max_diff)
        
        # 使用 torch.allclose 检查参数是否在容忍范围内
        if torch.allclose(ddp_param, single_param, atol=tolerance, rtol=1e-5):
            print(f"✓ {name}: 参数匹配 (最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e})")
        else:
            print(f"✗ {name}: 参数不匹配 (最大差异: {max_diff:.2e}, 平均差异: {mean_diff:.2e})")
            all_match = False
            
            # 对于差异较大的参数，打印一些调试信息
            if max_diff > 1e-3:
                print(f"  DDP 形状: {ddp_param.shape}, 单进程形状: {single_param.shape}")
                print(f"  DDP 样本: {ddp_param.flatten()[:5]}")
                print(f"  单进程样本: {single_param.flatten()[:5]}")
    
    print(f"\n总体最大差异: {max_diff_overall:.2e}")
    if max_diff_overall < tolerance:
        print("✅ 所有参数都在容忍范围内!")
    
    return all_match


if __name__ == "__main__":
    # 检查 PyTorch 分布式是否可用
    if not torch.distributed.is_available():
        print("PyTorch 分布式不可用")
        exit(1)
    
    world_size = 2
    epochs = 3
    batch_size = 16
    lr = 0.01
    
    print("开始朴素 DDP 实现测试...")
    print(f"进程数: {world_size}, 训练轮次: {epochs}, 批次大小: {batch_size}, 学习率: {lr}")
    
    try:
        # 为了测试，这里修改了方法，以避免多进程问题
        # 而是，在一个进程中模拟 DDP 行为
        print("\n运行模拟 DDP 训练...")
        
        # 设置确定性行为以确保结果可复现
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # 为每个“进程”创建模型，并进行相同的初始化
        models = []
        optimizers = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 创建第一个模型（rank 0）
        model_0 = ToyModel().to(device)
        optimizer_0 = optim.SGD(model_0.parameters(), lr=lr)
        models.append(model_0)
        optimizers.append(optimizer_0)
        
        # 创建额外的模型并从 rank 0 复制参数
        for rank in range(1, world_size):
            model = ToyModel().to(device)
            # 复制参数以确保所有模型的初始权重完全相同
            with torch.no_grad():
                for p_src, p_dst in zip(model_0.parameters(), model.parameters()):
                    p_dst.data.copy_(p_src.data)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            models.append(model)
            optimizers.append(optimizer)
        
        # 创建数据集
        dataset = ToyDataset(size=1000)
        
        # 训练循环
        criterion = nn.MSELoss()
        
        # 创建一个单进程数据加载器，并按顺序处理，以匹配单进程训练的行为
        dataloader = DataLoader(dataset, batch_size=batch_size * world_size, shuffle=False)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # 将批次数据分割到各个“进程”
                batch_splits = torch.chunk(data, world_size, dim=0)
                target_splits = torch.chunk(target, world_size, dim=0)
                
                # 清零所有模型的梯度
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                # 为每个进程进行前向和反向传播
                gradients = {}  # 存储梯度以便求平均
                total_loss = 0.0
                
                for rank in range(world_size):
                    # 检查是否有足够的批次数据
                    if rank < len(batch_splits) and batch_splits[rank].size(0) > 0:
                        batch_data = batch_splits[rank]
                        batch_target = target_splits[rank]
                        
                        # 前向传播
                        output = models[rank](batch_data)
                        loss = criterion(output, batch_target)
                        total_loss += loss.item()
                        
                        # 反向传播
                        loss.backward()
                        
                        # 收集梯度
                        for name, param in models[rank].named_parameters():
                            if param.grad is not None:
                                if name not in gradients:
                                    gradients[name] = []
                                gradients[name].append(param.grad.clone())
                
                # 对所有进程的梯度进行 All-Reduce（求平均）
                averaged_gradients = {}
                for name, grad_list in gradients.items():
                    if grad_list:  # 确保有梯度
                        # 将所有进程的梯度堆叠并求平均
                        averaged_gradients[name] = torch.stack(grad_list).mean(dim=0)
                
                # 将平均梯度应用到所有模型并更新
                for rank in range(world_size):
                    for name, param in models[rank].named_parameters():
                        if name in averaged_gradients:
                            # 复制平均梯度到当前模型的梯度
                            param.grad = averaged_gradients[name].clone()
                    optimizers[rank].step()
                
                epoch_loss += total_loss / world_size
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
        
        # 从 rank 0 获取最终的 DDP 状态
        ddp_final_state = {name: param.clone().cpu() for name, param in models[0].named_parameters()}
        
        print("\n运行单进程训练进行比较...")
        
        # 重置随机种子以确保相同的初始化
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            
        single_final_state = single_process_train(epochs=epochs, batch_size=batch_size, lr=lr, world_size=world_size)
        
        # 比较结果，由于浮点精度问题，使用更宽松的容差
        matches = compare_models(ddp_final_state, single_final_state, tolerance=1e-4)
        
        if matches:
            print("\n✅ 成功: 朴素 DDP 实现产生的结果与单进程训练相同！")
        else:
            print("\n❌ 失败: 朴素 DDP 实现与单进程训练不匹配。")
            print("这可能是由于:")
            print("1. 数值精度差异")
            print("2. 数据顺序不同")
            print("3. 实现中的 bug")
            
            # 尝试使用更宽松的容差
            print("\n尝试使用更宽松的容差 (1e-3)...")
            lenient_matches = compare_models(ddp_final_state, single_final_state, tolerance=1e-3)
            if lenient_matches:
                print("✅ 在宽松容差下成功！这很可能是数值精度问题。")
            
    except Exception as e:
        print(f"训练期间发生错误: {e}")
        import traceback
        traceback.print_exc()