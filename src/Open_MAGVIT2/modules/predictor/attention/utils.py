"""Attention helpers: SDPA path and mask utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention as MHA


def is_pure_causal_mha_mask(attn_mask: torch.Tensor) -> bool:
    """True when attn_mask is upper-triangular disallow (standard MHA causal)."""
    if attn_mask.dim() != 2 or attn_mask.shape[0] != attn_mask.shape[1]:
        return False
    n = attn_mask.shape[0]
    if n <= 1:
        return True
    expected = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=attn_mask.device),
        diagonal=1,
    )
    return torch.equal(attn_mask, expected)


def mha_self_attention(
    module: MHA,
    x: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Self-attention; uses SDPA with is_causal when mask is standard causal."""
    if attn_mask is not None and is_pure_causal_mha_mask(attn_mask):
        b, seq, dim = x.shape
        n_heads = module.num_heads
        head_dim = dim // n_heads
        if module.in_proj_bias is None:
            q, k, v = F._in_projection_packed(  # noqa: SLF001
                x, x, x, module.in_proj_weight, None
            )
        else:
            q, k, v = F._in_projection_packed(  # noqa: SLF001
                x, x, x, module.in_proj_weight, module.in_proj_bias
            )
        q = q.view(b, seq, n_heads, head_dim).transpose(1, 2)
        k = k.view(b, seq, n_heads, head_dim).transpose(1, 2)
        v = v.view(b, seq, n_heads, head_dim).transpose(1, 2)
        dropout_p = float(module.dropout) if module.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True,
        )
        out = out.transpose(1, 2).reshape(b, seq, dim)
        return module.out_proj(out)

    out, _ = module(x, x, x, attn_mask=attn_mask)
    return out
