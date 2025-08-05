import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        构造 RMSNorm 层。
        
        Args:
            d_model (int): 模型的隐藏维度
            eps (float): 数值稳定性的 epsilon 值
            device (torch.device, optional): 存储参数的设备
            dtype (torch.dtype, optional): 参数的数据类型
        """
        super(RMSNorm, self).__init__()
        
        # 可学习的增益参数
        self.gain = nn.Parameter(torch.ones(d_model))
        
        # 数值稳定性的 epsilon 参数
        self.eps = eps
        
        # 设置设备和数据类型
        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 RMSNorm 归一化操作。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, d_model)
        
        Returns:
            torch.Tensor: 返回与输入相同形状的张量
        """
        # 保存输入数据类型
        in_dtype = x.dtype
        
        # 将输入转换为 float32 以进行稳定计算
        x = x.to(torch.float32)
        
        # 计算 RMSNorm：首先计算 RMS(a)，然后应用增益参数
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        
        # 应用增益参数
        result = x_norm * self.gain
        
        # 返回原始数据类型的结果
        return result.to(in_dtype)

