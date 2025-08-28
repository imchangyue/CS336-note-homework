import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 1. 初始化进程组
# 这步是分布式通信的第一步，它让所有进程能够互相识别并建立通信
def setup(rank, world_size):
    """
    为每个子进程设置分布式环境。
    Args:
        rank (int): 当前进程的唯一ID（0到world_size-1）。
        world_size (int): 进程总数。
    """
    # 设置主节点的地址，这里使用本机IP地址，因为所有进程都在同一台机器上
    os.environ["MASTER_ADDR"] = "localhost"
    # 设置主节点的端口号，确保该端口未被占用
    os.environ["MASTER_PORT"] = "29500"
    # 初始化进程组
    # 'gloo' 是一个用于CPU通信的后端，适用于多进程在同一台机器上运行的情况
    # 对于GPU通信，通常使用性能更优的 'nccl' 后端
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# 2. 分布式操作示例
# 这是每个子进程都会执行的任务
def distributed_demo(rank, world_size):
    """
    演示一个简单的分布式操作：all_reduce。
    每个进程生成自己的数据，然后所有进程将数据相加并同步。
    Args:
        rank (int): 当前进程的ID。
        world_size (int): 进程总数。
    """
    # 调用setup函数，初始化当前进程的分布式环境
    setup(rank, world_size)

    # 每个进程生成一个独立的随机张量（Tensor），包含3个0到9之间的整数
    # 注意，在这一步，每个进程的`data`都是不同的
    data = torch.randint(0, 10, (3,))
    print(f"进程 {rank} 的数据 (all_reduce前): {data}")

    # 核心分布式操作：dist.all_reduce
    # all_reduce 会将所有进程的张量进行元素级求和，并将结果同步回每个进程
    # 例如，如果有4个进程，各自的data是[1,2,3], [4,5,6], [7,8,9], [1,1,1]
    # 执行 all_reduce后，所有进程的data都会变成 [13, 16, 20]
    # async_op=False 表示这是一个阻塞操作，所有进程必须等待此操作完成才能继续
    dist.all_reduce(data, async_op=False)

    print(f"进程 {rank} 的数据 (all_reduce后): {data}")


# 3. 程序入口
# 只有当脚本直接运行时，这段代码才会执行
if __name__ == "__main__":
    # 定义要启动的进程总数
    world_size = 4
    
    # torch.multiprocessing.spawn 是一个启动多进程的工具
    # 它会创建 nprocs 个子进程，并让每个子进程都执行 fn 指定的函数
    mp.spawn(
        fn=distributed_demo,  # 指定每个子进程要执行的函数
        args=(world_size,),   # 传递给 fn 函数的额外参数，这里只传递了world_size
        nprocs=world_size,    # 要启动的进程数量
        join=True             # 主进程会等待所有子进程执行完毕后才退出
    )