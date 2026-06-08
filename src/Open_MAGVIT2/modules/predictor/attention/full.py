"""Full self-attention wrapper."""

from __future__ import annotations

import torch.nn as nn
from torch.nn import MultiheadAttention as MHA


class FullAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHA(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, self_attn_mask=None):
        y = self.norm1(x)
        mask = ~self_attn_mask if self_attn_mask is not None else None
        out, _ = self.attn(y, y, y, attn_mask=mask)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x
