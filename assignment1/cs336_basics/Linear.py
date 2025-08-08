import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Constructs a linear transformation module (without bias).
        Args:
            in_features: int, final dimension of the input.
            out_features: int, final dimension of the output.
            device: torch.device | None, the device to store the parameters on.
            dtype: torch.dtype | None, the data type of the parameters.
        """
        super(Linear, self).__init__()
        
        # Initialize weight matrix W (d_out x d_in)
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # Initialize the weights with truncated normal distribution
        init.trunc_normal_(self.W, mean=0, std=2 / (in_features + out_features)**0.5, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation y = W * x to the input tensor x.
        Args:
            x: torch.Tensor, the input tensor of shape (batch_size, in_features)
        
        Returns:
            torch.Tensor: the transformed output of shape (batch_size, out_features)
        """
        # Perform the linear transformation (W is of shape [out_features, in_features])
        return einsum(x, self.W, 'b s i, o i -> b s o')

