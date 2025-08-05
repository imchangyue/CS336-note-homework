import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        """
        RoPE旋转位置编码实现
        Args:
            dim: 嵌入维度(必须是偶数)
            base: 用于计算角度的基数Θ
        """
        super().__init__()
        assert dim % 2 == 0, "维度必须是偶数"
        
        self.dim = dim
        self.base = base
        
        # 预计算并缓存所有可能的位置和维度对应的sin/cos值
        self.register_buffer("inv_freq", None, persistent=False)
        self._precompute_freqs()

    def _precompute_freqs(self):
        """预计算频率张量"""
        # 计算1/Θ^(2k/d) for k ∈ [0, d/2-1]
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", theta, persistent=False)

    def _compute_rotary_emb(self, x: torch.Tensor, seq_len: int):
        """
        计算旋转位置编码
        Args:
            x: 输入张量 [..., seq_len, dim]
            seq_len: 序列长度
        Returns:
            (cosθ, sinθ) 对
        """
        # 生成位置序列 [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)  # [seq_len]
        
        # 计算外积得到位置*频率矩阵 [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]
        
        # 重复两次以便后续处理 [seq_len, dim]
        freqs = freqs.repeat_interleave(2, dim=-1)  # [seq_len, dim]
        
        # 计算cos和sin
        cos = torch.cos(freqs)  # [seq_len, dim]
        sin = torch.sin(freqs)  # [seq_len, dim]
        
        return cos, sin

    def rotate_half(self, x: torch.Tensor):
        """将输入张量的后一半维度旋转"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        应用旋转位置编码
        Args:
            x: 输入张量 [..., seq_len, dim]
            cos: cos值 [seq_len, dim]
            sin: sin值 [seq_len, dim]
        Returns:
            旋转后的张量 [..., seq_len, dim]
        """
        # 旋转公式: q' = q * cosθ + rotate_half(q) * sinθ
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, dim]
        Returns:
            旋转后的张量 [batch_size, seq_len, dim]
        """
        seq_len = x.size(1)
        cos, sin = self._compute_rotary_emb(x, seq_len)
        
        # 确保cos/sin与x的维度匹配
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        
        return self.apply_rotary_emb(x, cos, sin)

