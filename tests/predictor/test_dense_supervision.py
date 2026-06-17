"""Dense causal supervision (PLAN_V2 §8.1a): alignment, no-leakage, _lasttok
equivalence, and that it runs + backprops."""

import torch

from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from tests.predictor.stack_factory import lite_stack, smoke_vae_32


def _video(b=1):
    return torch.randn(b, 3, 9, 32, 32)


def test_dense_supervision_alignment():
    """Every context position p targets gt frame p + k + 1; canonical is the tail."""
    _, _, _, _, schedule = lite_stack()
    for s in range(schedule.S):
        ctx_end = schedule._context_end[s]
        n_sp = schedule.stages[s].n_spatial
        for k in range(schedule.shifts_per_stage(s)):
            sup = schedule.build_shift_supervision(s, k, dense=True)
            # ctx_end+1 frames supervised
            assert sup.query_positions.numel() == (ctx_end + 1) * n_sp
            assert sup.target_positions.numel() == (ctx_end + 1) * n_sp
            # first target frame = k+1, last target frame = ctx_end + k + 1 = tgt_t
            assert int(sup.target_positions[0]) == (k + 1) * n_sp
            tgt_t = schedule.target_token_index(s, k)
            assert int(sup.target_positions[-1]) == (tgt_t + 1) * n_sp - 1


def test_dense_no_future_leakage():
    """Each supervised query frame p must NOT attend its own target frame p+k+1.

    Structural guard: the canonical (single-frame) target tgt_t lies strictly
    after the last context frame the query attends to, and the self-attn mask is
    causal so position p never sees p+1.."""
    _, _, _, _, schedule = lite_stack()
    from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
    b = PyramidMaskBuilder(schedule)
    for s in range(schedule.S):
        n_sp = schedule.stages[s].n_spatial
        m = b.build_self_attn_mask(s, schedule.t_total, torch.device("cpu"))  # True=allowed
        # frame-level causality: same-frame spatial attention is allowed, but no
        # frame may attend a strictly later frame (so query p can't see target p+1).
        frame_mask = m[::n_sp, ::n_sp]
        t = frame_mask.shape[0]
        upper = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        assert not (frame_mask & upper).any()


def _dense_stack():
    from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, StageSpec
    from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage
    from tests.predictor.stack_factory import V2_STAGE_ATTN
    import torch.nn as nn

    vae = smoke_vae_32()
    specs = (
        StageSpec("h4_w4", 4, 4, 512, 32),
        StageSpec("h8_w8", 8, 8, 256, 32),
        StageSpec("h16_w16", 16, 16, 128, 32),
    )
    schedule = PyramidSchedule.from_stage_specs(specs, t_context=5, t_total=9)
    stages = nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=2, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=9, max_shifts=4, has_parent=s > 0,
            **V2_STAGE_ATTN[s],
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1], dense_supervision=True)
    orch = EnvelopeOrchestrator(stages, schedule)
    return vae, prep, orch, stages


def test_dense_runs_and_lasttok_present():
    torch.manual_seed(0)
    vae, prep, orch, stages = _dense_stack()
    vae.eval()
    batch = prep.encode_and_schedule(vae, _video(), stages)
    out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce) and out.loss_ce.item() > 0
    assert out.loss_ce_lasttok is not None and torch.isfinite(out.loss_ce_lasttok)
    out.loss_ce.backward()
    for s, stage in enumerate(stages):
        assert any(p.grad is not None for p in stage.parameters()), f"stage {s} no grad"


def test_lasttok_equals_loss_ce_when_not_dense():
    """Single-frame (legacy) mode: loss_ce_lasttok == loss_ce exactly."""
    torch.manual_seed(1)
    vae, prep, orch, stages, _ = lite_stack()  # dense off
    vae.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, _video(), stages)
        out = orch.forward_train(batch)
    assert torch.allclose(out.loss_ce, out.loss_ce_lasttok)
