import torch
from jaxtyping import Float
from torch import Tensor

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Precompute cache
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs) # [L, D/2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Complex64
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: Float[Tensor, "B T H D"], start_pos: int = 0) -> Float[Tensor, "B T H D"]:
        B, T, H, D = x.shape
        # View as complex: [B, T, H, D/2]
        x_complex = torch.view_as_complex(x.float().reshape(B, T, H, -1, 2))
        
        # Slicing for cache or inference
        end_pos = start_pos + T
        freqs = self.freqs_cis[start_pos:end_pos].view(1, T, 1, -1) # Broadcast
        
        # Rotate
        x_rotated = torch.view_as_real(x_complex * freqs).flatten(3)
        return x_rotated.type_as(x)