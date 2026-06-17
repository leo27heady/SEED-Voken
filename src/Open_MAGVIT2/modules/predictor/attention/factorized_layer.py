"""Factorized predictor layer with optional cross-attention and dual-stream routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import DEFAULT_KIT, BlockKit
from src.Open_MAGVIT2.modules.predictor.attention.factorized import FactorizedSpaceTimeBlock
from src.Open_MAGVIT2.modules.predictor.cross_conditioning import (
    CrossConditioning,
    remap_cross_state_dict,
)


class FactorizedPredictorLayer(nn.Module):
    """Process dual streams with factorized space-time attention."""

    # legacy checkpoint key -> unified submodule key (CrossConditioning, §5.5)
    _CROSS_REMAP = {
        "cross_q_f": "cross_f.q", "cross_k_f": "cross_f.k",
        "cross_v_f": "cross_f.v", "cross_attn_f": "cross_f.attn",
        "cross_q_g": "cross_g.q", "cross_k_g": "cross_g.k",
        "cross_v_g": "cross_g.v", "cross_attn_g": "cross_g.attn",
    }

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_spatial: int,
        *,
        t_window: int = 1,
        spatial_window: int | None = None,
        has_cross_attn: bool = False,
        parent_dim: int | None = None,
        kit: BlockKit = DEFAULT_KIT,
    ) -> None:
        super().__init__()
        self.has_cross_attn = has_cross_attn
        kw = dict(t_window=t_window, spatial_window=spatial_window, kit=kit)
        self.block_f = FactorizedSpaceTimeBlock(dim, n_heads, n_spatial, **kw)
        self.block_g = FactorizedSpaceTimeBlock(dim, n_heads, n_spatial, **kw)
        if has_cross_attn:
            kv_dim = parent_dim if parent_dim is not None else dim
            # construction order (f then g; q,k,v,attn within each) preserves the
            # old RNG draw order -> fresh seeded init stays byte-identical.
            self.cross_f = CrossConditioning(dim, n_heads, kv_dim)
            self.cross_g = CrossConditioning(dim, n_heads, kv_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        remap_cross_state_dict(state_dict, prefix, self._CROSS_REMAP)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def _cross(
        self,
        x: torch.Tensor,
        cross_kv: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
        *,
        module: CrossConditioning,
    ) -> torch.Tensor:
        if cross_kv is None:
            return x
        return x + module(x, cross_kv, cross_mask)

    def forward(
        self,
        x: torch.Tensor,
        *,
        self_attn_mask: torch.Tensor | None,
        cross_kv: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
        layer_idx: int,
        parent_mode: str,
        parent_o1: torch.Tensor | None,
        parent_o2: torch.Tensor | None,
        parent_fused: torch.Tensor | None,
        rope_temporal=None,
        rope_spatial=None,
    ) -> torch.Tensor:
        i1, i2 = x.chunk(2, dim=-1)
        o1 = self.block_f(i1, self_attn_mask, rope_temporal=rope_temporal, rope_spatial=rope_spatial)
        o2 = self.block_g(i2, self_attn_mask, rope_temporal=rope_temporal, rope_spatial=rope_spatial)
        if self.has_cross_attn and parent_mode != "none":
            if parent_mode == "dual_stream":
                kv_f = parent_o1 if layer_idx % 2 == 0 else parent_o2
                kv_g = parent_o2 if layer_idx % 2 == 0 else parent_o1
            else:
                kv_f = kv_g = parent_fused
            o1 = self._cross(o1, kv_f, cross_attn_mask, module=self.cross_f)
            o2 = self._cross(o2, kv_g, cross_attn_mask, module=self.cross_g)
        return torch.cat([o1, o2], dim=-1)
