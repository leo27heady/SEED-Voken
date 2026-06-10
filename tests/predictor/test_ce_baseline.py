"""PRED-CE-01: untrained predictor CE near sum of log(K) per shift."""

import math

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage


def _stack():
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
    vae = VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4,
    )
    schedule = PyramidSchedule.from_v2_64s()
    stages = torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=2, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=13, max_shifts=4, has_parent=s > 0,
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=0.0)
    return vae, prep, orch, stages, schedule


def test_pred_ce_01_baseline_log_k():
    torch.manual_seed(0)
    vae, prep, orch, stages, schedule = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(2, 3, 13, 64, 64)
    expected = 0.0
    for s in range(schedule.S):
        k_size = schedule.stages[s].codebook_size
        for k in range(schedule.shifts_per_stage(s)):
            expected += math.log(k_size)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce)
    assert abs(float(out.loss_ce) - expected) < 0.5
