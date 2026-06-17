"""Parallel residual block for predictor stages."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as MHA

from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import (
    DEFAULT_KIT,
    BlockKit,
    build_mlp,
    build_norm,
    mlp_out_linear,
)
from src.Open_MAGVIT2.modules.predictor.attention.utils import mha_self_attention
from src.Open_MAGVIT2.modules.predictor.cross_conditioning import (
    CrossConditioning,
    remap_cross_state_dict,
)


class ParallelPredictorResidual(nn.Module):
    # legacy checkpoint key -> unified submodule key (CrossConditioning, §5.5)
    _CROSS_REMAP = {
        "cross_attn_q": "cross.q", "cross_attn_k": "cross.k",
        "cross_attn_v": "cross.v", "cross_attn": "cross.attn",
    }

    def __init__(
        self,
        dim: int,
        n_heads: int,
        has_cross_attn: bool = False,
        parent_dim: int | None = None,
        mlp_ratio: int = 4,
        kit: BlockKit = DEFAULT_KIT,
    ) -> None:
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.kit = kit
        self.norm = build_norm(dim, kit.norm_type, eps=1e-6)
        self.self_attn = MHA(dim, n_heads, batch_first=True)
        if has_cross_attn:
            kv_dim = parent_dim if parent_dim is not None else dim
            # q,k,v,attn order matches the old cross_attn_q/k/v/cross_attn order,
            # keeping module-traversal (and so trunc_normal_) draws identical.
            self.cross = CrossConditioning(dim, n_heads, kv_dim)
        self.mlp = build_mlp(dim, mlp_ratio, kit.mlp_type)
        self._init_weights()

    def _load_from_state_dict(self, state_dict, prefix, *args):
        remap_cross_state_dict(state_dict, prefix, self._CROSS_REMAP)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        mlp_out = mlp_out_linear(self.mlp)
        nn.init.zeros_(mlp_out.weight)
        nn.init.zeros_(mlp_out.bias)
        nn.init.zeros_(self.self_attn.out_proj.weight)
        if self.self_attn.out_proj.bias is not None:
            nn.init.zeros_(self.self_attn.out_proj.bias)
        if self.has_cross_attn:
            nn.init.zeros_(self.cross.attn.out_proj.weight)
            if self.cross.attn.out_proj.bias is not None:
                nn.init.zeros_(self.cross.attn.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        self_attn_mask: torch.Tensor | None = None,
        cross_kv: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        rope_apply=None,
    ) -> torch.Tensor:
        y = self.norm(x)
        sa_mask = ~self_attn_mask if self_attn_mask is not None else None
        sa_out = mha_self_attention(
            self.self_attn, y, attn_mask=sa_mask, rope_apply=rope_apply, qk_norm=self.kit.qk_norm
        )
        delta = sa_out
        if self.has_cross_attn and cross_kv is not None:
            delta = delta + self.cross(y, cross_kv, cross_attn_mask)
        delta = delta + self.mlp(y)
        return x + delta
