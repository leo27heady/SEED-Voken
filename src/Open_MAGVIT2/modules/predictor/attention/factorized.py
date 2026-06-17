"""Factorized space-time attention for fine predictor stages."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as MHA

from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import (
    DEFAULT_KIT,
    BlockKit,
    build_mlp,
    build_norm,
)
from src.Open_MAGVIT2.modules.predictor.attention.utils import mha_self_attention


def build_spatial_window_mask(
    n_spatial: int,
    spatial_window: int,
    device: torch.device,
) -> torch.Tensor:
    """Bool mask (n_sp, n_sp): True = allowed attention. Vectorized on device."""
    side = int(n_spatial**0.5)
    sw = spatial_window
    r = torch.arange(side, device=device)
    c = torch.arange(side, device=device)
    gr, gc = torch.meshgrid(r, c, indexing="ij")
    src_idx = (gr * side + gc).reshape(-1)

    dr = torch.arange(-sw, sw + 1, device=device)
    dc = torch.arange(-sw, sw + 1, device=device)
    gdr, gdc = torch.meshgrid(dr, dc, indexing="ij")
    noff = gdr.numel()

    nr = gr.reshape(-1, 1).expand(-1, noff) + gdr.reshape(1, -1)
    nc = gc.reshape(-1, 1).expand(-1, noff) + gdc.reshape(1, -1)
    valid = (nr >= 0) & (nr < side) & (nc >= 0) & (nc < side)
    dst_idx = nr * side + nc

    mask = torch.zeros(n_spatial, n_spatial, dtype=torch.bool, device=device)
    src_rep = src_idx.unsqueeze(1).expand(-1, noff)
    mask[src_rep[valid], dst_idx[valid]] = True
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
        kit: BlockKit = DEFAULT_KIT,
    ) -> None:
        super().__init__()
        self.n_spatial = n_spatial
        self.t_window = t_window
        self.spatial_window = spatial_window
        self.kit = kit
        self.norm_t = build_norm(dim, kit.norm_type)
        self.norm_s = build_norm(dim, kit.norm_type)
        self.temporal_attn = MHA(dim, n_heads, batch_first=True)
        self.spatial_attn = MHA(dim, n_heads, batch_first=True)
        self.mlp = build_mlp(dim, mlp_ratio, kit.mlp_type)
        self.norm_m = build_norm(dim, kit.norm_type)
        if spatial_window is not None:
            allowed = build_spatial_window_mask(n_spatial, spatial_window, torch.device("cpu"))
            self.register_buffer(
                "spatial_attn_mask",
                ~allowed,
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        rope_temporal=None,
        rope_spatial=None,
    ) -> torch.Tensor:
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
        t_out = mha_self_attention(
            self.temporal_attn, yt, attn_mask=ta_mask, rope_apply=rope_temporal,
            qk_norm=self.kit.qk_norm,
        )
        xt = xt + t_out
        xs = xt.view(b, self.n_spatial, t_len, d).permute(0, 2, 1, 3).reshape(
            b * t_len, self.n_spatial, d
        )
        ys = self.norm_s(xs)
        sa_mask = self.spatial_attn_mask if self.spatial_window is not None else None
        s_out = mha_self_attention(
            self.spatial_attn, ys, attn_mask=sa_mask, rope_apply=rope_spatial,
            qk_norm=self.kit.qk_norm,
        )
        xs = xs + s_out
        x = xs.view(b, t_len, self.n_spatial, d).reshape(b, n, d)
        x = x + self.mlp(self.norm_m(x))
        return x
