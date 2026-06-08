"""Stream fusion for reversible predictor (CE checkpoint)."""

from __future__ import annotations

import torch
import torch.nn as nn


class StreamFusion(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConcatStreamFusion(StreamFusion):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class ConvStreamFusion(StreamFusion):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


def build_stream_fusion(mode: str, dim: int) -> StreamFusion:
    if mode == "concat":
        return ConcatStreamFusion(dim)
    if mode == "conv":
        return ConvStreamFusion(dim)
    raise ValueError(f"Unknown stream fusion mode: {mode!r}")
