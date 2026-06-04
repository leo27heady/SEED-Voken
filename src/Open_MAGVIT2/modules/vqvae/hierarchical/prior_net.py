"""Lightweight prior heads for HQ-style learned categorical priors (5D video)."""

from typing import Tuple

import torch
from torch import nn

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import ConvBlock3D


class GaussianPriorHead(nn.Module):
    """Maps conditioning features to a prior mean field z_pri over the tap grid."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock3D(
                in_channels, width, kernel_size=(3, 3, 3), causal=True, padding=1
            ),
            nn.ReLU(True),
            ConvBlock3D(
                width, out_channels, kernel_size=(1, 1, 1), causal=True, padding=0
            ),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)
