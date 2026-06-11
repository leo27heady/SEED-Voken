"""Parallel residual block for predictor stages."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as MHA

from src.Open_MAGVIT2.modules.predictor.attention.utils import mha_self_attention


class ParallelPredictorResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        has_cross_attn: bool = False,
        parent_dim: int | None = None,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = MHA(dim, n_heads, batch_first=True)
        if has_cross_attn:
            kv_dim = parent_dim if parent_dim is not None else dim
            self.cross_attn_q = nn.Linear(dim, dim)
            self.cross_attn_k = nn.Linear(kv_dim, dim)
            self.cross_attn_v = nn.Linear(kv_dim, dim)
            self.cross_attn = MHA(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.self_attn.out_proj.weight)
        if self.self_attn.out_proj.bias is not None:
            nn.init.zeros_(self.self_attn.out_proj.bias)
        if self.has_cross_attn:
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            if self.cross_attn.out_proj.bias is not None:
                nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        self_attn_mask: torch.Tensor | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.norm(x)
        sa_mask = ~self_attn_mask if self_attn_mask is not None else None
        sa_out = mha_self_attention(self.self_attn, y, attn_mask=sa_mask)
        delta = sa_out
        if self.has_cross_attn and cross_kv is not None:
            q = self.cross_attn_q(y)
            k = self.cross_attn_k(cross_kv)
            v = self.cross_attn_v(cross_kv)
            ca_mask = ~cross_attn_mask if cross_attn_mask is not None else None
            ca_out, _ = self.cross_attn(q, k, v, attn_mask=ca_mask)
            delta = delta + ca_out
        delta = delta + self.mlp(y)
        return x + delta
