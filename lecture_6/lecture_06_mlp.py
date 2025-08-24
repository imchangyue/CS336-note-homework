import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
     
    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            with nvtx.range(f"Layer {i+1}"):
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        return x

def run_mlp(dim, num_layers, batch_size, num_steps, use_optimizer):
    device = get_device()
    
    with nvtx.range("define model"):
        model = MLP(dim, num_layers).to(device)
    
    optimizer = torch.optim.Adam(model.parameters()) if use_optimizer else None
    
    with nvtx.range("create input"):
        x = torch.randn(batch_size, dim).to(device)
    
    # 预热几步
    for _ in range(3):
        y = model(x).mean()
        y.backward()
        if use_optimizer:
            optimizer.step()
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)
    
    # 同步GPU确保预热完成
    torch.cuda.synchronize()
    
    for step in range(num_steps):
        # 在第10步之后开始profiling（给足够的预热时间）
        if step == 10:
            torch.cuda.cudart().cudaProfilerStart()
        
        nvtx.range_push(f"Step_{step}")
        
        if use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)
        
        with nvtx.range("forward"):
            y = model(x).mean()  # 修复：使用正确的变量名 x
        
        with nvtx.range("backward"):
            y.backward()
        
        if use_optimizer:
            with nvtx.range("optimizer step"):
                print(f"Step {step}, loss: {y.item()}")  # 打印loss
                optimizer.step()
        
        nvtx.range_pop()
    
    # 停止profiling
    torch.cuda.cudart().cudaProfilerStop()

def main():
    if torch.cuda.is_available():
        print("CUDA is available. Starting MLP training...")
        run_mlp(dim=4096, num_layers=24, batch_size=512, num_steps=50, use_optimizer=True)
        print("Training completed.")
    else:
        print("CUDA is not available. Please run on a machine with a CUDA-capable GPU.")

if __name__ == "__main__":
    main()