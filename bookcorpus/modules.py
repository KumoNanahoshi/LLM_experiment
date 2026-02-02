import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        # Force float32 for variance calculation for stability
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.type_as(x) * self.weight)

class HyperConnection(nn.Module):
    """
    Implements LayerScale style connection.
    y = x + alpha * f(x)
    If vector: alpha is [D]. If scalar: alpha is [1].
    """
    def __init__(self, dim: int, type: str = "vector", init_value: float = 1e-2):
        super().__init__()
        if type == "vector":
            self.alpha = nn.Parameter(torch.full((dim,), init_value))
        else:
            self.alpha = nn.Parameter(torch.tensor(init_value))

    def forward(self, x: Tensor, sublayer_out: Tensor) -> Tensor:
        return x + self.alpha * sublayer_out