"""Orchestrator integration smoke test."""

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage


def _build_vae():
    ddconfig = dict(
        double_z=False, z_channels=32, resolution=64, in_channels=3, out_ch=3,
        ch=64, ch_mult=[1, 2, 2, 4], num_res_blocks=2,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="native", tap_key_format="spatial",
        sequence_length=13, latent_key="h8_w8",
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h8_w8=32, h16_w16=128, h32_w32=128),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[1536, 768, 384], dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4,
    )


def _stack(lambda_mse=0.0):
    vae = _build_vae()
    schedule = PyramidSchedule.from_v2_64s()
    stages = torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=2, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=13, max_shifts=4, has_parent=s > 0,
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=lambda_mse)
    return vae, prep, orch, stages


def test_orc_01_smoke_forward():
    torch.manual_seed(0)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
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
    video = torch.randn(1, 3, 13, 64, 64)
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
    video = torch.randn(1, 3, 13, 64, 64)
    batch = prep.encode_and_schedule(vae, video, stages)
    batch.video = video
    out = orch.forward_train(batch)
    assert out.loss_mse is None


def test_orc_07_mse_finite():
    torch.manual_seed(2)
    vae, prep, orch, stages = _stack(lambda_mse=0.5)
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    batch = prep.encode_and_schedule(vae, video, stages)
    batch.video = video
    out = orch.forward_train(batch)
    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel
    zero_acts = {k: torch.zeros_like(v) for k, v in batch.activations.items()}
    level_indices = [out.pred_indices[s] for s in range(3)]
    recon = vae.decode_from_indices(
        level_indices, zero_acts, encoder_bottleneck=batch.encoder_bottleneck,
    )
    mse = torch.mean((recon[:, :, 9:13] - video[:, :, 9:13]) ** 2)
    assert torch.isfinite(mse)
