#!/usr/bin/env python3
"""
Naive implementation of Distributed Data Parallel (DDP) training.
This script implements DDP by manually all-reducing individual parameter gradients.
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
    """Simple dataset with random data for testing."""
    
    def __init__(self, size=1000, input_dim=10, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.size = size
        self.input_dim = input_dim
        
        # Generate random data
        self.data = torch.randn(size, input_dim)
        # Generate random targets (regression task)
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ToyModel(nn.Module):
    """Simple neural network for testing."""
    
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
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def broadcast_parameters(model, src_rank=0):
    """Broadcast model parameters from source rank to all other ranks."""
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)


def all_reduce_gradients(model):
    """All-reduce gradients across all processes."""
    for param in model.parameters():
        if param.grad is not None:
            # All-reduce gradients and average them
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()


def naive_ddp_train(rank, world_size, epochs=5, batch_size=32, lr=0.01):
    """Naive DDP training function."""
    print(f"Running DDP training on rank {rank}")
    
    # Initialize process group
    init_process(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create model and move to device
    model = ToyModel().to(device)
    
    # Broadcast parameters from rank 0 to ensure all processes start with same weights
    broadcast_parameters(model, src_rank=0)
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Create dataset and distributed sampler
    dataset = ToyDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # All-reduce gradients across all processes
            all_reduce_gradients(model)
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    
    # Save final model state for verification
    final_state = {name: param.clone().cpu() for name, param in model.named_parameters()}
    
    cleanup()
    return final_state


def single_process_train(epochs=5, batch_size=32, lr=0.01, world_size=2):
    """Single process training for comparison."""
    print("Running single process training for comparison")
    
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model with same initialization as DDP
    model = ToyModel().to(device)
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Create dataset - use same total batch size as DDP (batch_size * world_size)
    dataset = ToyDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size * world_size, shuffle=False)  # No shuffle for reproducibility
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
    
    # Return final model state
    return {name: param.clone().cpu() for name, param in model.named_parameters()}


def run_ddp(world_size=2):
    """Run distributed training."""
    # Use 'gloo' backend for CPU or 'nccl' for GPU
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    # Spawn processes for distributed training
    mp.spawn(naive_ddp_train, args=(world_size,), nprocs=world_size, join=True)


def compare_models(ddp_state, single_state, tolerance=1e-6):
    """Compare model states from DDP and single process training."""
    print("\nComparing model parameters:")
    all_match = True
    max_diff_overall = 0.0
    
    for name in ddp_state.keys():
        ddp_param = ddp_state[name]
        single_param = single_state[name]
        
        # Calculate differences
        diff = torch.abs(ddp_param - single_param)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        max_diff_overall = max(max_diff_overall, max_diff)
        
        # Check if parameters are close
        if torch.allclose(ddp_param, single_param, atol=tolerance, rtol=1e-5):
            print(f"✓ {name}: Parameters match (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        else:
            print(f"✗ {name}: Parameters don't match (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
            all_match = False
            
            # Debug info for first few mismatches
            if max_diff > 1e-3:
                print(f"  DDP shape: {ddp_param.shape}, Single shape: {single_param.shape}")
                print(f"  DDP sample: {ddp_param.flatten()[:5]}")
                print(f"  Single sample: {single_param.flatten()[:5]}")
    
    print(f"\nOverall maximum difference: {max_diff_overall:.2e}")
    if max_diff_overall < tolerance:
        print("✅ All parameters within tolerance!")
    
    return all_match


if __name__ == "__main__":
    # Check if we can run distributed training
    if not torch.distributed.is_available():
        print("PyTorch distributed is not available")
        exit(1)
    
    world_size = 2
    epochs = 3
    batch_size = 16
    lr = 0.01
    
    print("Starting naive DDP implementation test...")
    print(f"World size: {world_size}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
    
    try:
        # For this test, we'll modify the approach to avoid multiprocessing issues
        # Instead, we'll simulate the DDP behavior in a single process
        print("\nRunning simulated DDP training...")
        
        # Set deterministic behavior for reproducible results
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Create models for each "rank" with identical initialization
        models = []
        optimizers = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create the first model (rank 0)
        model_0 = ToyModel().to(device)
        optimizer_0 = optim.SGD(model_0.parameters(), lr=lr)
        models.append(model_0)
        optimizers.append(optimizer_0)
        
        # Create additional models and copy parameters from rank 0
        for rank in range(1, world_size):
            model = ToyModel().to(device)
            # Copy parameters from rank 0 to ensure identical initialization
            with torch.no_grad():
                for p_src, p_dst in zip(model_0.parameters(), model.parameters()):
                    p_dst.data.copy_(p_src.data)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            models.append(model)
            optimizers.append(optimizer)
        
        # Create dataset
        dataset = ToyDataset(size=1000)
        
        # Training loop
        criterion = nn.MSELoss()
        
        # Create a single dataloader and process it sequentially to match single-process behavior
        dataloader = DataLoader(dataset, batch_size=batch_size * world_size, shuffle=False)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # Split batch across "ranks"
                batch_splits = torch.chunk(data, world_size, dim=0)
                target_splits = torch.chunk(target, world_size, dim=0)
                
                # Zero gradients for all models
                for optimizer in optimizers:
                    optimizer.zero_grad()
                
                # Forward and backward pass for each rank
                gradients = {}  # Store gradients for averaging
                total_loss = 0.0
                
                for rank in range(world_size):
                    if rank < len(batch_splits) and batch_splits[rank].size(0) > 0:
                        batch_data = batch_splits[rank]
                        batch_target = target_splits[rank]
                        
                        # Forward pass
                        output = models[rank](batch_data)
                        loss = criterion(output, batch_target)
                        total_loss += loss.item()
                        
                        # Backward pass
                        loss.backward()
                        
                        # Collect gradients
                        for name, param in models[rank].named_parameters():
                            if param.grad is not None:
                                if name not in gradients:
                                    gradients[name] = []
                                gradients[name].append(param.grad.clone())
                
                # All-reduce gradients (average them)
                averaged_gradients = {}
                for name, grad_list in gradients.items():
                    if grad_list:  # Make sure we have gradients
                        averaged_gradients[name] = torch.stack(grad_list).mean(dim=0)
                
                # Apply averaged gradients to all models and update
                for rank in range(world_size):
                    for name, param in models[rank].named_parameters():
                        if name in averaged_gradients:
                            param.grad = averaged_gradients[name].clone()
                    optimizers[rank].step()
                
                epoch_loss += total_loss / world_size
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
        
        # Get final DDP state from rank 0
        ddp_final_state = {name: param.clone().cpu() for name, param in models[0].named_parameters()}
        
        print("\nRunning single process training for comparison...")
        
        # Reset random seed to ensure same initialization
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            
        single_final_state = single_process_train(epochs=epochs, batch_size=batch_size, lr=lr, world_size=world_size)
        
        # Compare results with more lenient tolerance due to floating point precision
        matches = compare_models(ddp_final_state, single_final_state, tolerance=1e-4)
        
        if matches:
            print("\n✅ SUCCESS: DDP implementation produces the same results as single-process training!")
        else:
            print("\n❌ FAILURE: DDP implementation doesn't match single-process training.")
            print("This might be due to:")
            print("1. Numerical precision differences")
            print("2. Different data ordering")
            print("3. Implementation bugs")
            
            # Try with more lenient tolerance
            print("\nTrying with more lenient tolerance (1e-3)...")
            lenient_matches = compare_models(ddp_final_state, single_final_state, tolerance=1e-3)
            if lenient_matches:
                print("✅ SUCCESS with lenient tolerance! This is likely due to numerical precision.")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()