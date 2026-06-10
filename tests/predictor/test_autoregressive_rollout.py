"""Inference mode tests: parallel vs autoregressive."""

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
    return vae, prep, orch, stages


def test_inf_01_parallel_oracle_uses_full_context():
    torch.manual_seed(0)
    vae, prep, orch, stages = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        train_out = orch.forward_train(batch)
        par_out = orch.forward_parallel(batch)
    assert torch.isfinite(par_out.loss_ce)
    assert par_out.logits[(2, 0)].shape[1] > train_out.logits[(2, 0)].shape[1]


def test_inf_02_ar_differs_from_parallel():
    torch.manual_seed(1)
    vae, prep, orch, stages = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        par_out = orch.forward_parallel(batch)
        ar_out = orch.forward_autoregressive(batch)
    assert torch.isfinite(ar_out.loss_ce)
    assert ar_out.ar_context_len is not None
    # Parallel oracle sees full native-T tokens; AR shift-0 uses context length only.
    assert par_out.logits[(2, 0)].shape[1] > ar_out.logits[(2, 0)].shape[1]


def test_inf_03_ar_context_growth():
    torch.manual_seed(2)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        init_len = batch.context_embed[2].shape[1]
        ar_out = orch.forward_autoregressive(batch)
    assert ar_out.ar_context_len == init_len + 4 * 32 * 32


def test_inf_04_ar_decode_shape():
    torch.manual_seed(4)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        ar_out = orch.forward_autoregressive(batch)
        level_indices = [ar_out.pred_indices[s] for s in range(3)]
        zero_acts = {k: torch.zeros_like(v) for k, v in batch.activations.items()}
        recon = vae.decode_from_indices(
            level_indices, zero_acts, encoder_bottleneck=batch.encoder_bottleneck,
        )
    assert recon.shape == video.shape


def test_inf_05_greedy_stable():
    torch.manual_seed(3)
    vae, prep, orch, stages = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        torch.manual_seed(3)
        a = orch.forward_autoregressive(batch)
        torch.manual_seed(3)
        b = orch.forward_autoregressive(batch)
    assert torch.isfinite(a.loss_ce)
    assert torch.allclose(a.loss_ce, b.loss_ce, rtol=0, atol=1e-6)
    for s in a.pred_indices:
        assert torch.equal(a.pred_indices[s], b.pred_indices[s])
