from multiprocessing import spawn
import torch
import time
import os
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
import torch.multiprocessing as mp

def get_device(rank: int):
    # 强制所有进程使用 cuda:0，或者在无GPU时使用CPU
    if torch.cuda.is_available():
        return torch.device("cuda:0") 
    else:
        return torch.device("cpu")

def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
# 示例: 验证 All-reduce 与 Reduce-scatter + All-gather 等价
def collective_operations_main(rank: int, world_size: int):
    setup(rank, world_size)

    # All-reduce
    dist.barrier()
    tensor = torch.tensor([0.,1,2,3], device=get_device(rank)) + rank
    print(f"Rank {rank} before all_reduce: {tensor}", flush=True)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} after all_reduce: {tensor}", flush=True)
    dist.barrier()
    
    # Reduce-scatter
    dist.barrier()
    # 示例: 假设每个进程有16个元素，世界大小为4，则每个进程有4个元素。
    # 每个进程将自己的4个元素发送出去，最后每个进程收到一个4个元素的向量。
    input_tensor = torch.arange(4, device=get_device(rank)) + rank * 4
    output_tensor = torch.empty(1, device=get_device(rank))

    print(f"Rank {rank} [before reduce-scatter]: input = {input_tensor}", flush=True)
    dist.reduce_scatter_tensor(output=output_tensor, input=input_tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: output = {output_tensor}", flush=True)
    
    # All-gather
    dist.barrier()
    input_gather = output_tensor
    output_gather = [torch.empty_like(input_gather) for _ in range(world_size)]

    print(f"Rank {rank} [before all-gather]: input = {input_gather}", flush=True)
    dist.all_gather(output_gather, input_gather, async_op=False)
    print(f"Rank {rank} [after all-gather]: output = {torch.cat(output_gather)}", flush=True)

    cleanup()

def render_duration(duration):
    if duration < 1e-6:
        return f"{duration*1e9:.2f} ns"
    if duration < 1e-3:
        return f"{duration*1e6:.2f} µs"
    if duration < 1:
        return f"{duration*1e3:.2f} ms"
    return f"{duration:.2f} s"

def all_reduce_benchmark(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    tensor = torch.randn(num_elements, device=get_device(rank))
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
    end_time = time.time()
    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)
    cleanup()

def reduce_scatter_benchmark(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    # 输入张量维度修正，以便每个进程有自己的分片数据
    input = torch.randn(num_elements * world_size, device=get_device(rank)) 
    output = torch.empty(num_elements, device=get_device(rank))

    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
    end_time = time.time()
    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)
    cleanup()

if __name__ == "__main__":
    # 示例1：运行集体操作验证
    print("--- 验证 All-reduce 与 Reduce-scatter + All-gather ---")
    mp.spawn(collective_operations_main, args=(2,), nprocs=2, join=True)
    
    # 示例2：运行性能基准测试
    print("\n--- 运行性能基准测试 ---")
    world_size_benchmark = 4
    num_elements_benchmark = 100 * 1024**2 # 100MB
    mp.spawn(all_reduce_benchmark, args=(world_size_benchmark, num_elements_benchmark), nprocs=world_size_benchmark, join=True)
    mp.spawn(reduce_scatter_benchmark, args=(world_size_benchmark, num_elements_benchmark), nprocs=world_size_benchmark, join=True)