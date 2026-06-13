"""Orchestrator integration smoke test (lite 32px geometry — grad-safe)."""

import torch

from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from tests.predictor.stack_factory import lite_stack, smoke_vae_32


def _build_vae():
    return smoke_vae_32()


def _stack(lambda_mse=0.0):
    vae, prep, orch, stages, _ = lite_stack(n_layers=2, lambda_pred_mse=lambda_mse)
    return vae, prep, orch, stages


def _video(b=1):
    return torch.randn(b, 3, 9, 32, 32)


def test_orc_01_smoke_forward():
    torch.manual_seed(0)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = _video()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce)
    assert out.loss_ce.item() > 0


def test_orc_03_dfs_order():
    schedule = PyramidSchedule.from_v2_64s()
    assert schedule.envelope_order() == [
        (0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3),
    ]


def test_orc_04_all_stages_grad():
    torch.manual_seed(1)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = _video()
    batch = prep.encode_and_schedule(vae, video, stages)
    out = orch.forward_train(batch)
    out.loss_ce.backward()
    for s, stage in enumerate(stages):
        assert any(p.grad is not None for p in stage.parameters()), f"stage {s} no grad"


def test_orc_05_frozen_vae():
    vae = _build_vae()
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in vae.parameters())


def test_orc_06_mse_skipped():
    vae, prep, orch, stages = _stack(lambda_mse=0.0)
    vae.eval()
    video = _video()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        batch.video = video
        out = orch.forward_train(batch)
    assert out.loss_mse is None


def test_orc_07_mse_finite():
    torch.manual_seed(2)
    vae, prep, orch, stages = _stack(lambda_mse=0.5)
    vae.eval()
    video = _video()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        batch.video = video
        out = orch.forward_train(batch)
        zero_acts = {k: torch.zeros_like(v) for k, v in batch.activations.items()}
        level_indices = [out.pred_indices[s] for s in range(3)]
        recon = vae.decode_from_indices(
            level_indices, zero_acts, encoder_bottleneck=batch.encoder_bottleneck,
        )
    # lite geometry: context 5, horizon frames 5..8
    mse = torch.mean((recon[:, :, 5:9] - video[:, :, 5:9]) ** 2)
    assert torch.isfinite(mse)
