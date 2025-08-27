import os
import torch
import torch.distributed as dist
import time
import numpy as np
import csv

# 设置环境变量，用于进程组初始化
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

def run_benchmark(rank, world_size, size_mb, backend):
    """
    执行all-reduce操作的基准测试函数。
    """
    
    # 1. 初始化进程组
    print(f"进程 {rank}/{world_size}: 使用 {backend} 后端初始化...")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"进程 {rank}/{world_size}: 初始化完成。")

    # 确定张量大小
    tensor_size_bytes = size_mb * 1024 * 1024
    num_elements = tensor_size_bytes // 4  # float32 = 4 bytes

    # 2. 创建张量并将其放到正确设备上
    if backend == "nccl":
        if torch.cuda.is_available():
            device = f'cuda:0'
            tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
        else:
            print("警告：未找到CUDA设备，NCCL测试无法执行。")
            dist.destroy_process_group()
            return None
    else:  # Gloo 后端
        tensor = torch.randn(num_elements, dtype=torch.float32)

    # 3. 预热（Warm-up）
    print(f"进程 {rank}/{world_size}: 开始预热...")
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize()
    print(f"进程 {rank}/{world_size}: 预热完成。")

    # 4. 计时（Benchmarking）
    timings = []
    num_iterations = 20
    
    print(f"进程 {rank}/{world_size}: 开始计时，迭代 {num_iterations} 次...")
    for _ in range(num_iterations):
        start_time = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize()
        end_time = time.time()
        timings.append(end_time - start_time)

    # 5. 收集所有进程的计时结果
    all_timings_obj = [None for _ in range(world_size)]
    dist.all_gather_object(all_timings_obj, timings)

    # 6. 销毁进程组
    dist.destroy_process_group()
    
    # 7. 在主进程 (rank 0) 中聚合并返回结果
    if rank == 0:
        aggregated_timings = [item for sublist in all_timings_obj for item in sublist]
        avg_time_ms = np.mean(aggregated_timings) * 1000
        print(f"\n--- 结果 ---")
        print(f"后端: {backend}, 进程数: {world_size}, 数据大小: {size_mb} MB")
        print(f"平均耗时: {avg_time_ms:.4f} ms\n")
        return {
            'backend': backend,
            'world_size': world_size,
            'size_mb': size_mb,
            'avg_time_ms': avg_time_ms
        }
    return None

def main():
    """
    主函数，用于组织不同的实验运行并保存结果到 CSV 文件。
    """
    backends = ["gloo", "nccl"]
    sizes_mb = [1, 10, 100] # 1GB
    
    # 考虑只有一个GPU，所以对NCCL后端只测试world_size=1
    gloo_world_sizes = [2, 4, 6]
    nccl_world_sizes = [1] 
    
    all_results = []
    
    # Gloo 后端测试
    print("----- 开始 Gloo 后端测试 (CPU) -----")
    for size_mb in sizes_mb:
        for world_size in gloo_world_sizes:
            print(f"\n运行: 后端=Gloo, 进程数={world_size}, 大小={size_mb} MB")
            
            result = torch.multiprocessing.spawn(
                run_benchmark,
                args=(world_size, size_mb, "gloo"),
                nprocs=world_size,
                join=True
            )
            if result:
                all_results.append(result)
    
    # NCCL 后端测试
    print("\n----- 开始 NCCL 后端测试 (单GPU) -----")
    if torch.cuda.is_available():
        for size_mb in sizes_mb:
            for world_size in nccl_world_sizes:
                print(f"\n运行: 后端=NCCL, 进程数={world_size}, 大小={size_mb} MB")
                
                result = torch.multiprocessing.spawn(
                    run_benchmark,
                    args=(world_size, size_mb, "nccl"),
                    nprocs=world_size,
                    join=True
                )
                if result:
                    all_results.append(result)
    else:
        print("\n警告: 未检测到CUDA设备，已跳过NCCL基准测试。")
    
    # 将所有结果写入 CSV 文件
    csv_file = 'assignment2-systems/cs336_systems/benchmark_results.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['backend', 'world_size', 'size_mb', 'avg_time_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n\n所有实验结果已保存到 '{csv_file}' 文件。")


if __name__ == '__main__':
    main()