"""Factorized predictor layer with optional cross-attention and dual-stream routing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as MHA

from src.Open_MAGVIT2.modules.predictor.attention.factorized import FactorizedSpaceTimeBlock


class FactorizedPredictorLayer(nn.Module):
    """Process dual streams with factorized space-time attention."""

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
    ) -> None:
        super().__init__()
        self.has_cross_attn = has_cross_attn
        kw = dict(t_window=t_window, spatial_window=spatial_window)
        self.block_f = FactorizedSpaceTimeBlock(dim, n_heads, n_spatial, **kw)
        self.block_g = FactorizedSpaceTimeBlock(dim, n_heads, n_spatial, **kw)
        if has_cross_attn:
            kv_dim = parent_dim if parent_dim is not None else dim
            self.cross_q_f = nn.Linear(dim, dim)
            self.cross_k_f = nn.Linear(kv_dim, dim)
            self.cross_v_f = nn.Linear(kv_dim, dim)
            self.cross_attn_f = MHA(dim, n_heads, batch_first=True)
            self.cross_q_g = nn.Linear(dim, dim)
            self.cross_k_g = nn.Linear(kv_dim, dim)
            self.cross_v_g = nn.Linear(kv_dim, dim)
            self.cross_attn_g = MHA(dim, n_heads, batch_first=True)

    def _cross(
        self,
        x: torch.Tensor,
        cross_kv: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
        *,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        attn: MHA,
    ) -> torch.Tensor:
        if cross_kv is None:
            return x
        q = q_proj(x)
        k = k_proj(cross_kv)
        v = v_proj(cross_kv)
        ca_mask = ~cross_mask if cross_mask is not None else None
        out, _ = attn(q, k, v, attn_mask=ca_mask)
        return x + out

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
    ) -> torch.Tensor:
        i1, i2 = x.chunk(2, dim=-1)
        o1 = self.block_f(i1, self_attn_mask)
        o2 = self.block_g(i2, self_attn_mask)
        if self.has_cross_attn and parent_mode != "none":
            if parent_mode == "dual_stream":
                kv_f = parent_o1 if layer_idx % 2 == 0 else parent_o2
                kv_g = parent_o2 if layer_idx % 2 == 0 else parent_o1
            else:
                kv_f = kv_g = parent_fused
            o1 = self._cross(
                o1, kv_f, cross_attn_mask,
                q_proj=self.cross_q_f, k_proj=self.cross_k_f,
                v_proj=self.cross_v_f, attn=self.cross_attn_f,
            )
            o2 = self._cross(
                o2, kv_g, cross_attn_mask,
                q_proj=self.cross_q_g, k_proj=self.cross_k_g,
                v_proj=self.cross_v_g, attn=self.cross_attn_g,
            )
        return torch.cat([o1, o2], dim=-1)
