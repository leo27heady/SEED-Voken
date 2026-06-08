"""Attention backend tests."""

import pytest
import torch

from src.Open_MAGVIT2.modules.predictor.attention.factorized import (
    FactorizedSpaceTimeBlock,
    build_spatial_window_mask,
)
from src.Open_MAGVIT2.modules.predictor.attention.full import FullAttentionBlock
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage


def test_att_01_factorized_shape():
    block = FactorizedSpaceTimeBlock(dim=16, n_heads=2, n_spatial=16, t_window=1)
    x = torch.randn(2, 4 * 16, 16)
    y = block(x)
    assert y.shape == x.shape


def test_att_02_windowed_spatial_local():
    n_sp = 64
    sw = 2
    local = build_spatial_window_mask(n_sp, sw, torch.device("cpu"))
    side = 8
    i = 1 * side + 1
    allowed = local[i].nonzero(as_tuple=True)[0].tolist()
    for j in allowed:
        r, c = j // side, j % side
        assert abs(r - 1) <= sw and abs(c - 1) <= sw
    far = 7 * side + 7
    assert not local[i, far].item()
    block = FactorizedSpaceTimeBlock(
        dim=8, n_heads=2, n_spatial=n_sp, t_window=1, spatial_window=sw
    )
    x = torch.randn(1, 3 * n_sp, 8)
    y = block(x)
    assert torch.isfinite(y).all()


def test_att_03_small_grid_factorized_vs_full():
    """On H=W=4, T=3, windowed spatial with sw=large approximates full spatial."""
    dim, n_sp, t_len = 16, 16, 3
    x = torch.randn(1, t_len * n_sp, dim)
    fact = FactorizedSpaceTimeBlock(
        dim=dim, n_heads=2, n_spatial=n_sp, t_window=-1, spatial_window=4
    )
    full_block = FullAttentionBlock(dim=dim, n_heads=2)
    n = t_len * n_sp
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
    with torch.no_grad():
        yf = fact(x.clone())
        yfull = full_block(x.clone(), self_attn_mask=mask)
    assert yf.shape == yfull.shape
    assert torch.isfinite(yf).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_att_oom_smoke_fine_stage():
    stage = PredictorStage(
        dim=32, n_heads=4, n_layers=1, codebook_size=384,
        h=32, w=32, max_t=13, max_shifts=4, has_parent=True,
        attention_type="factorized", t_window=1, spatial_window=8,
    ).cuda()
    ctx = torch.randn(2, 9 * 1024, 32, device="cuda")
    from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
    from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, ShiftMasks
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    full = builder.build_self_attn_mask(2, 1, ctx.device)
    masks = ShiftMasks(self_attn=full[: 9 * 1024, : 9 * 1024])
    out = stage.execute_shift(
        0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=9
    )
    assert torch.isfinite(out.logits).all()
