"""Factorized space-time attention for fine predictor stages."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as MHA


def build_spatial_window_mask(
    n_spatial: int,
    spatial_window: int,
    device: torch.device,
) -> torch.Tensor:
    """Bool mask (n_sp, n_sp): True = allowed attention."""
    side = int(n_spatial**0.5)
    sw = spatial_window
    mask = torch.zeros(n_spatial, n_spatial, dtype=torch.bool, device=device)
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr in range(-sw, sw + 1):
                for dc in range(-sw, sw + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < side and 0 <= nc < side:
                        mask[i, nr * side + nc] = True
    return mask


class FactorizedSpaceTimeBlock(nn.Module):
    """Alternate causal temporal and spatial self-attention."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_spatial: int,
        t_window: int = 1,
        spatial_window: int | None = None,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.n_spatial = n_spatial
        self.t_window = t_window
        self.spatial_window = spatial_window
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.temporal_attn = MHA(dim, n_heads, batch_first=True)
        self.spatial_attn = MHA(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.norm_m = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, self_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        t_len = n // self.n_spatial
        xt = x.view(b, t_len, self.n_spatial, d).permute(0, 2, 1, 3).reshape(
            b * self.n_spatial, t_len, d
        )
        yt = self.norm_t(xt)
        ta_mask = None
        if self_attn_mask is not None:
            n_sp = self.n_spatial
            ta_mask = ~self_attn_mask[: t_len * n_sp, : t_len * n_sp][::n_sp, ::n_sp]
        t_out, _ = self.temporal_attn(yt, yt, yt, attn_mask=ta_mask)
        xt = xt + t_out
        xs = xt.view(b, self.n_spatial, t_len, d).permute(0, 2, 1, 3).reshape(
            b * t_len, self.n_spatial, d
        )
        ys = self.norm_s(xs)
        sa_mask = None
        if self.spatial_window is not None:
            local = build_spatial_window_mask(
                self.n_spatial, self.spatial_window, x.device
            )
            sa_mask = ~local
        s_out, _ = self.spatial_attn(ys, ys, ys, attn_mask=sa_mask)
        xs = xs + s_out
        x = xs.view(b, t_len, self.n_spatial, d).reshape(b, n, d)
        x = x + self.mlp(self.norm_m(x))
        return x
