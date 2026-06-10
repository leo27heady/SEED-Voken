"""PAR-MEM-01: parallel context mode matches train on 32px lite geometry."""

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, StageSpec
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage


def _lite_stack():
    ddconfig = dict(
        double_z=False, z_channels=16, resolution=32, in_channels=3, out_ch=3,
        ch=32, ch_mult=[1, 2, 2, 4], num_res_blocks=1, num_groups=8,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="native", tap_key_format="spatial",
        sequence_length=9, latent_key="h4_w4",
        blocks_sq="h4_w4_x1,h8_w8_x1,h16_w16_x1",
        tap_channels=dict(h4_w4=16, h8_w8=64, h16_w16=64),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[4096, 2048, 1024], dim_dict=[96, 96, 96],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    vae = VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4,
    )
    stages_spec = (
        StageSpec("h4_w4", 4, 4, 4096, 96),
        StageSpec("h8_w8", 8, 8, 2048, 96),
        StageSpec("h16_w16", 16, 16, 1024, 96),
    )
    schedule = PyramidSchedule.from_stage_specs(stages_spec, t_context=5, t_total=9)
    stages = torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=1, codebook_size=spec.codebook_size,
            codebook_dim=96, h=spec.H, w=spec.W, max_t=9, max_shifts=4,
            has_parent=s > 0, attention_type="factorized" if s == 2 else "full",
            t_window=1 if s == 2 else 3, spatial_window=4 if s == 2 else None,
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=0.0)
    return vae, prep, orch, stages


def test_par_mem_01_parallel_context_matches_train():
    torch.manual_seed(1)
    vae, prep, orch, stages = _lite_stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 9, 32, 32)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        train_out = orch.forward_train(batch)
        ctx_out = orch.forward_parallel(batch, parallel_mode="context")
    assert torch.allclose(train_out.loss_ce, ctx_out.loss_ce, rtol=0, atol=1e-5)


def test_inf_06_parallel_context_mode():
    test_par_mem_01_parallel_context_matches_train()
