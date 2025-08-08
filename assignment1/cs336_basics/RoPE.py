import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        # RoPE 论文中的公式，用于计算旋转频率
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        
        # 为每个位置计算旋转角度
        t = torch.arange(max_seq_len, device=device)
        freqs_flat = torch.outer(t, freqs).to(device)

        # 预计算 cos 和 sin 值
        cos_vals = torch.cos(freqs_flat) # shape: [max_seq_len, d_k / 2]
        sin_vals = torch.sin(freqs_flat) # shape: [max_seq_len, d_k / 2]

        # 将 cos 和 sin 扩展到 d_k 维度
        cos_vals = torch.repeat_interleave(cos_vals, 2, dim=-1) # shape: [max_seq_len, d_k]
        sin_vals = torch.repeat_interleave(sin_vals, 2, dim=-1) # shape: [max_seq_len, d_k]

        # 将预计算的值注册为 buffer，而不是参数
        self.register_buffer('cos_vals', cos_vals, persistent=False)
        self.register_buffer('sin_vals', sin_vals, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)
            token_positions (torch.Tensor): 令牌位置张量，形状为 (..., seq_len)
        Returns:
            torch.Tensor: 经过 RoPE 旋转后的张量，形状与 x 相同
        """
        # 1. 重塑 x 以便后续操作。
        # 这里可以使用 rearrange 或手动实现，但为了通用性，我们直接按位置索引
        
        # 确保 cos 和 sin 缓冲区的维度与 x 匹配，以便广播
        # 注意: token_positions 可能是多维的
        seq_len = x.shape[-2]
        
        # 使用 token_positions 从预计算的 cos 和 sin 中选择对应的值
        # 这里假设 token_positions 的形状可以广播到 x 的前几个维度
        cos_pos = self.cos_vals[token_positions]
        sin_pos = self.sin_vals[token_positions]

        # 2. 将 x 向量分成两半进行旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # 将 cos 和 sin 的维度与 x_even/x_odd 匹配
        cos_pos_half = cos_pos[..., 0::2]
        sin_pos_half = sin_pos[..., 0::2]
        
        # 3. 应用旋转
        # 旋转后的偶数索引元素 = 偶数元素 * cos - 奇数元素 * sin
        x_even_rotated = x_even * cos_pos_half - x_odd * sin_pos_half
        # 旋转后的奇数索引元素 = 偶数元素 * sin + 奇数元素 * cos
        x_odd_rotated = x_even * sin_pos_half + x_odd * cos_pos_half
        
        # 4. 将旋转后的两半重新合并
        result = torch.empty_like(x)
        result[..., 0::2] = x_even_rotated
        result[..., 1::2] = x_odd_rotated
        
        return result