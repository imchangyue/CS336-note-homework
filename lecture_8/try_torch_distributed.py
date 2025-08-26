
import torch
import time
import os
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp

def get_device(rank: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    else:
        return torch.device("cpu")

def collective_operations_main(rank: int, world_size: int):
    setup(rank, world_size)

    dist.barrier() # 等待所有进程到达此处
    tensor = torch.tensor([0.,1,2,3],device= get_device(rank)) + rank
    print(f"Rank {rank} before all_reduce: {tensor}",flush = True)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} after all_reduce: {tensor}",flush = True)
    dist.barrier() # 等待所有进程到达此处

    input = torch.arange(world_size*4, dtype=torch.float32, device=get_device(rank)) + rank
    output = torch.empty(1,device=get_device(rank)) # 只需要一个元素

def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
def cleanup():
    torch.distributed.destroy_process_group()
if __name__ == "__main__":
    main()