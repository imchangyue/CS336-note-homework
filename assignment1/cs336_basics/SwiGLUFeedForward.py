import torch
import torch.nn as nn
from einops import einsum

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        d_ff = int(8 / 3 * d_model)
        self.d_ff = (d_ff // 64) * 64
        
        # 初始化权重
        self.W1 = nn.Parameter(torch.empty(d_model, self.d_ff))  # [d_model, d_ff]
        self.W2 = nn.Parameter(torch.empty(self.d_ff, d_model))  # [d_ff, d_model]
        self.W3 = nn.Parameter(torch.empty(d_model, self.d_ff))  # [d_model, d_ff]
        
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        手动实现SiLU激活函数: x * σ(x)
        σ(x) = 1 / (1 + exp(-x))
        """
        return x / (1 + torch.exp(-x)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程，使用手动实现的SiLU
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 计算 W1x 和 W3x
        W1x = einsum(x, self.W1, 'b s d, d f -> b s f')
        W3x = einsum(x, self.W3, 'b s d, d f -> b s f')
        
        # 使用手动实现的SiLU激活函数并进行门控
        glued = self.silu(W1x) * W3x
        
        # 通过W2变换
        output = einsum(glued, self.W2, 'b s f, f d -> b s d')
        return output