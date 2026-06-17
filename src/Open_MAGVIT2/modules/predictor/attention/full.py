"""Full self-attention wrapper."""

from __future__ import annotations

import torch.nn as nn
from torch.nn import MultiheadAttention as MHA

from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import (
    DEFAULT_KIT,
    BlockKit,
    build_mlp,
    build_norm,
)
from src.Open_MAGVIT2.modules.predictor.attention.utils import mha_self_attention


class FullAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, kit: BlockKit = DEFAULT_KIT) -> None:
        super().__init__()
        self.kit = kit
        self.norm1 = build_norm(dim, kit.norm_type)
        self.attn = MHA(dim, n_heads, batch_first=True)
        self.norm2 = build_norm(dim, kit.norm_type)
        self.mlp = build_mlp(dim, mlp_ratio, kit.mlp_type)

    def forward(self, x, self_attn_mask=None, rope_apply=None):
        y = self.norm1(x)
        # mask convention: schedule masks are True=allowed -> invert to True=disallow.
        mask = ~self_attn_mask if self_attn_mask is not None else None
        out = mha_self_attention(
            self.attn, y, attn_mask=mask, rope_apply=rope_apply, qk_norm=self.kit.qk_norm
        )
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x
