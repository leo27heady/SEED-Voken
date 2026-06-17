"""Factorized 3D RoPE: relative-invariance, length-generalization, axis budgets,
and end-to-end integration incl. the n_spatial==1 top stage (PLAN_V2 §6.1/§6.5)."""

import torch

from src.Open_MAGVIT2.modules.predictor.attention.rope import Rotary3D, default_axes
from tests.predictor.stack_factory import bigstep_stack, lite_stack


def test_default_axes_even_and_budgeted():
    for hd in (8, 16, 32, 64):
        dt, dh, dw = default_axes(hd, has_spatial=True)
        assert dt % 2 == 0 and dh % 2 == 0 and dw % 2 == 0
        assert dt + dh + dw <= hd and dt >= dh and dt >= dw
    # no-spatial (1x1 top): all budget on time, none on space
    assert default_axes(16, has_spatial=False) == (16, 0, 0)


def test_no_spatial_appliers_are_none():
    rope = Rotary3D(16, has_spatial=False)
    assert rope.spatial_applier(1, 1) is None  # nothing to rotate spatially
    assert rope.temporal_applier(3) is not None


def test_temporal_rope_is_relative():
    """q_i . k_j depends only on (i - j) when the whole head_dim rotates by time."""
    torch.manual_seed(0)
    head_dim = 16
    rope = Rotary3D(head_dim, has_spatial=False)  # axes (16,0,0)
    seq = 8
    q0 = torch.randn(head_dim)
    k0 = torch.randn(head_dim)
    q = q0.expand(1, 1, seq, head_dim).contiguous()
    k = k0.expand(1, 1, seq, head_dim).contiguous()
    apply = rope.temporal_applier(seq)
    qr, kr = apply(q, k)
    scores = (qr[0, 0] @ kr[0, 0].t())  # (seq, seq)
    # constant along each diagonal (same i-j) -> relative
    for i in range(seq - 1):
        for j in range(seq - 1):
            assert torch.allclose(scores[i, j], scores[i + 1, j + 1], atol=1e-4)


def test_length_generalization_no_nan():
    rope = Rotary3D(16, has_spatial=False)
    q = torch.randn(1, 1, 50, 16)
    k = torch.randn(1, 1, 50, 16)
    qr, kr = rope.temporal_applier(50)(q, k)  # far beyond any training t
    assert torch.isfinite(qr).all() and torch.isfinite(kr).all()


def test_spatial_rope_rotates_only_spatial_dims():
    """Spatial applier leaves the temporal budget [0:D_t] untouched."""
    torch.manual_seed(1)
    rope = Rotary3D(16, has_spatial=True)  # (8,4,4)
    d_t = rope.axes[0]
    q = torch.randn(1, 1, 16, 16)  # 4x4 spatial
    k = torch.randn(1, 1, 16, 16)
    qr, kr = rope.spatial_applier(4, 4)(q, k)
    assert torch.allclose(qr[..., :d_t], q[..., :d_t])  # time dims unchanged
    assert not torch.allclose(qr[..., d_t:], q[..., d_t:])  # spatial dims rotated


def test_rope_bigstep_forward_and_grad():
    """RoPE end-to-end on the big-step stack: full-attn 1x1 top + factorized mid/fine."""
    torch.manual_seed(2)
    vae, prep, orch, stages, _ = bigstep_stack(pos_encoding="rope")
    vae.eval()
    # no absolute position params under rope
    for st in stages:
        assert not hasattr(st, "temporal_pos")
        assert st.rope is not None
    batch = prep.encode_and_schedule(vae, torch.randn(1, 3, 9, 32, 32), stages)
    out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce)
    out.loss_ce.backward()
    for s, stage in enumerate(stages):
        assert any(p.grad is not None for p in stage.parameters()), f"stage {s} no grad"


def test_rope_lite_parallel_and_ar_finite():
    torch.manual_seed(3)
    vae, prep, orch, stages, _ = lite_stack(pos_encoding="rope")
    vae.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, torch.randn(1, 3, 9, 32, 32), stages)
        assert torch.isfinite(orch.forward_parallel(batch).loss_ce)
        assert torch.isfinite(orch.forward_autoregressive(batch).loss_ce)
