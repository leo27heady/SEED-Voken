"""Attention helpers: unified SDPA self-attention path + mask utilities.

`mha_self_attention` always unpacks q/k/v and runs `F.scaled_dot_product_attention`
so a RoPE rotation can be hooked between in-projection and the score computation
(PLAN_V2 §6.2). With `rope_apply=None` it is numerically equivalent to the legacy
`nn.MultiheadAttention(attn_mask=...)` call (the bit-equality gate
`tests/predictor/test_mha_polarity_equivalence.py`).

Mask convention (unchanged from the callers): `attn_mask` is a bool tensor where
``True`` means *disallow* (the polarity the callers already produce via ``~allowed``),
or ``None``. It is converted to an additive ``-inf`` bias for SDPA; a pure-causal
mask uses the fused ``is_causal`` fast path when no RoPE is applied.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention as MHA

RopeApply = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


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
    rope_apply: Optional[RopeApply] = None,
    qk_norm: bool = False,
) -> torch.Tensor:
    """Self-attention via manual q/k/v unpacking + SDPA.

    ``rope_apply`` (if given) rotates q and k after in-projection / head-split and
    before SDPA — this is the RoPE seam. ``qk_norm`` applies a parameter-free
    RMSNorm to q,k *before* the rotation ("norm then rotate"). ``attn_mask`` is
    bool (True = disallow) or None.
    """
    from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import maybe_qk_norm

    b, seq, dim = x.shape
    n_heads = module.num_heads
    head_dim = dim // n_heads
    q, k, v = F._in_projection_packed(  # noqa: SLF001
        x, x, x, module.in_proj_weight, module.in_proj_bias
    )
    q = q.view(b, seq, n_heads, head_dim).transpose(1, 2)
    k = k.view(b, seq, n_heads, head_dim).transpose(1, 2)
    v = v.view(b, seq, n_heads, head_dim).transpose(1, 2)
    q, k = maybe_qk_norm(q, k, qk_norm)
    if rope_apply is not None:
        q, k = rope_apply(q, k)
    dropout_p = float(module.dropout) if module.training else 0.0

    attn_bias = None
    is_causal = False
    if attn_mask is not None:
        # Fused causal kernel only when no RoPE-induced asymmetry concerns; with
        # rope we still pass the same additive mask (SDPA handles it identically).
        if rope_apply is None and is_pure_causal_mha_mask(attn_mask):
            is_causal = True
        else:
            attn_bias = torch.zeros(
                attn_mask.shape, dtype=q.dtype, device=q.device
            ).masked_fill_(attn_mask, float("-inf"))

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=is_causal
    )
    out = out.transpose(1, 2).reshape(b, seq, dim)
    return module.out_proj(out)
