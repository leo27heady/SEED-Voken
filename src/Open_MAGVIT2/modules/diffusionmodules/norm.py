"""Video normalization utilities (encoder/decoder v2)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameWiseGroupNorm(nn.Module):
    """GroupNorm applied independently per time step on (B, C, T, H, W)."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"FrameWiseGroupNorm expects 5D input, got {x.dim()}D")
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.gn(y)
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
