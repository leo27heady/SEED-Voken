"""Bit-equality gate for the unified SDPA self-attention path (PLAN_V2 §6.2).

`mha_self_attention(..., rope_apply=None)` must reproduce the legacy
`nn.MultiheadAttention(attn_mask=~allowed)` call across the mask shapes the
predictor actually uses (causal, block-causal, windowed temporal, spatial-window).
This must pass BEFORE RoPE lands so a polarity/numeric off-by-one can't be
misattributed to RoPE.
"""

import torch
from torch.nn import MultiheadAttention as MHA

from src.Open_MAGVIT2.modules.predictor.attention.utils import mha_self_attention


def _legacy(module, x, allowed_mask):
    mask = ~allowed_mask if allowed_mask is not None else None
    out, _ = module(x, x, x, attn_mask=mask)
    return out


def _new(module, x, allowed_mask):
    mask = ~allowed_mask if allowed_mask is not None else None
    return mha_self_attention(module, x, attn_mask=mask)


def _causal_allowed(n):
    i = torch.arange(n)
    return (i.unsqueeze(1) - i.unsqueeze(0)) >= 0  # lower-tri incl diag = allowed


def _window_allowed(n, w):
    i = torch.arange(n)
    diff = i.unsqueeze(1) - i.unsqueeze(0)
    return (diff >= 0) & (diff < w)


def _check(module, x, allowed):
    module.eval()
    with torch.no_grad():
        a = _legacy(module, x, allowed)
        b = _new(module, x, allowed)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (a - b).abs().max().item()


def test_equiv_pure_causal():
    torch.manual_seed(0)
    m = MHA(32, 4, batch_first=True)
    x = torch.randn(2, 7, 32)
    _check(m, x, _causal_allowed(7))


def test_equiv_windowed():
    torch.manual_seed(1)
    m = MHA(32, 4, batch_first=True)
    x = torch.randn(2, 9, 32)
    _check(m, x, _window_allowed(9, 3))


def test_equiv_no_mask():
    torch.manual_seed(2)
    m = MHA(32, 4, batch_first=True)
    x = torch.randn(2, 5, 32)
    _check(m, x, None)


def test_equiv_block_causal():
    """Coarse-stage style mask: causal over frames, repeated over spatial blocks."""
    torch.manual_seed(3)
    m = MHA(32, 4, batch_first=True)
    t, n_sp = 3, 4
    frame = _causal_allowed(t)
    allowed = frame.repeat_interleave(n_sp, 0).repeat_interleave(n_sp, 1)
    x = torch.randn(2, t * n_sp, 32)
    _check(m, x, allowed)


def test_equiv_spatial_window():
    """Spatial-window mask (symmetric, includes self)."""
    torch.manual_seed(4)
    m = MHA(32, 4, batch_first=True)
    from src.Open_MAGVIT2.modules.predictor.attention.factorized import (
        build_spatial_window_mask,
    )
    allowed = build_spatial_window_mask(16, 1, torch.device("cpu"))
    x = torch.randn(2, 16, 32)
    _check(m, x, allowed)
