"""Reversible coupling block for predictor stages."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.predictor.parallel_residual import ParallelPredictorResidual


class ReversibleCouplingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        has_cross_attn: bool = False,
        parent_dim: int | None = None,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.F = ParallelPredictorResidual(
            dim=dim, n_heads=n_heads, has_cross_attn=has_cross_attn,
            parent_dim=parent_dim, mlp_ratio=mlp_ratio,
        )
        self.G = ParallelPredictorResidual(
            dim=dim, n_heads=n_heads, has_cross_attn=has_cross_attn,
            parent_dim=parent_dim, mlp_ratio=mlp_ratio,
        )
        self._self_attn_mask = None
        self._cross_attn_mask = None
        self._cross_kv = None
        self._parent_mode = "dual_stream"
        self._parent_o1 = None
        self._parent_o2 = None

    def set_masks(
        self,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> None:
        self._self_attn_mask = self_attn_mask
        self._cross_attn_mask = cross_attn_mask

    def set_parent(
        self,
        *,
        parent_mode: str,
        o1: torch.Tensor | None,
        o2: torch.Tensor | None,
        fused: torch.Tensor | None,
        layer_idx: int,
    ) -> None:
        self._parent_mode = parent_mode
        self._parent_o1 = o1
        self._parent_o2 = o2
        if parent_mode == "dual_stream":
            self._cross_kv = o1 if layer_idx % 2 == 0 else o2
        else:
            self._cross_kv = fused

    def _kwargs(self) -> dict:
        return {
            "self_attn_mask": self._self_attn_mask,
            "cross_kv": self._cross_kv,
            "cross_attn_mask": self._cross_attn_mask,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i1, i2 = torch.chunk(x, 2, dim=-1)
        o2 = i2 + self.F(i1, **self._kwargs())
        o1 = i1 + self.G(o2, **self._kwargs())
        return torch.cat([o1, o2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        o1, o2 = torch.chunk(y, 2, dim=-1)
        i1 = o1 - self.G(o2, **self._kwargs())
        i2 = o2 - self.F(i1, **self._kwargs())
        return torch.cat([i1, i2], dim=-1)
