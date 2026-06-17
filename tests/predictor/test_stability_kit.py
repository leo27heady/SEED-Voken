"""Stability kit (PLAN_V2 §6.4): RMSNorm / SwiGLU / QK-norm flags.

Default (layernorm/mlp/qk_norm off) must be byte-identical to legacy (guarded
globally by the golden fixture); the non-default flags must build, run, and
backprop on the lite + big-step stacks.
"""

import torch

from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import (
    RMSNorm,
    SwiGLU,
    build_mlp,
    build_norm,
    maybe_qk_norm,
)
from tests.predictor.stack_factory import bigstep_stack


def test_build_norm_and_mlp_types():
    assert isinstance(build_norm(16, "layernorm"), torch.nn.LayerNorm)
    assert isinstance(build_norm(16, "rmsnorm"), RMSNorm)
    assert isinstance(build_mlp(16, 4, "swiglu"), SwiGLU)


def test_rmsnorm_shape_and_finite():
    x = torch.randn(2, 5, 16)
    assert torch.isfinite(RMSNorm(16)(x)).all()
    assert RMSNorm(16)(x).shape == x.shape


def test_qk_norm_noop_when_disabled():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    q2, k2 = maybe_qk_norm(q, k, enabled=False)
    assert q2 is q and k2 is k
    q3, k3 = maybe_qk_norm(q, k, enabled=True)
    assert not torch.allclose(q3, q)  # normalized


def _build_with_kit(**predictor_overrides):
    """Build a bigstep stack but override stage flags via PredictorStage kwargs."""
    import torch.nn as nn

    from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
    from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
    from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, StageSpec
    from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage
    from tests.predictor.stack_factory import V2_STAGE_ATTN, smoke_vae_32_bigstep

    vae = smoke_vae_32_bigstep()
    specs = (
        StageSpec("h1_w1", 1, 1, 512, 16),
        StageSpec("h4_w4", 4, 4, 256, 32),
        StageSpec("h16_w16", 16, 16, 128, 32),
    )
    schedule = PyramidSchedule.from_stage_specs(specs, t_context=5, t_total=9)
    stages = nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=2, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=9, max_shifts=4, has_parent=s > 0,
            **V2_STAGE_ATTN[s], **predictor_overrides,
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule)
    return vae, prep, orch, stages


def test_stability_kit_full_stack_runs_and_grads():
    torch.manual_seed(0)
    vae, prep, orch, stages = _build_with_kit(
        norm_type="rmsnorm", mlp_type="swiglu", qk_norm=True, pos_encoding="rope",
    )
    vae.eval()
    batch = prep.encode_and_schedule(vae, torch.randn(1, 3, 9, 32, 32), stages)
    out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce)
    out.loss_ce.backward()
    for s, stage in enumerate(stages):
        assert any(p.grad is not None for p in stage.parameters()), f"stage {s} no grad"
