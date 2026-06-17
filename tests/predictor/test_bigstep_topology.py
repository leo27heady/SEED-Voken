"""Predictor on the big-step topology (top 1x1@T3 / mid 4x4@T5 / fine 16x16@T9).

Covers the 1x1 GLOBAL top stage (n_spatial==1) and the x4 parent<-child cross-attn
ratio end-to-end through all three rollout modes + a backward pass.
"""

import torch

from tests.predictor.stack_factory import bigstep_stack


def _video(b=1):
    return torch.randn(b, 3, 9, 32, 32)


def test_bigstep_schedule_geometry():
    _, _, _, _, schedule = bigstep_stack()
    assert schedule.S == 3
    assert [s.n_spatial for s in schedule.stages] == [1, 16, 256]
    assert [schedule.native_t(i) for i in range(3)] == [3, 5, 9]
    assert [schedule.shifts_per_stage(i) for i in range(3)] == [1, 2, 4]
    assert schedule.envelope_order() == [
        (0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3),
    ]


def test_bigstep_forward_train_finite():
    torch.manual_seed(0)
    vae, prep, orch, stages, _ = bigstep_stack()
    vae.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, _video(), stages)
        out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce) and out.loss_ce.item() > 0


def test_bigstep_parallel_and_ar_finite():
    torch.manual_seed(1)
    vae, prep, orch, stages, _ = bigstep_stack()
    vae.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, _video(), stages)
        out_par = orch.forward_parallel(batch)
        out_ar = orch.forward_autoregressive(batch)
    assert torch.isfinite(out_par.loss_ce)
    assert torch.isfinite(out_ar.loss_ce)


def test_bigstep_all_stages_grad_incl_1x1_top():
    """Gradient must flow through the single-spatial-token coarse stage."""
    torch.manual_seed(2)
    vae, prep, orch, stages, _ = bigstep_stack()
    vae.eval()
    batch = prep.encode_and_schedule(vae, _video(), stages)
    out = orch.forward_train(batch)
    out.loss_ce.backward()
    for s, stage in enumerate(stages):
        assert any(p.grad is not None for p in stage.parameters()), f"stage {s} no grad"


def test_bigstep_token_grids():
    """The predictor data path lands tokens on the native big-step grids."""
    vae, prep, _, stages, _ = bigstep_stack()
    vae.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, _video(), stages)
    grids = [tuple(batch.gt_indices[s].shape[1:]) for s in range(3)]
    assert grids == [(3, 1, 1), (5, 4, 4), (9, 16, 16)], grids


def test_bigstep_full_model_rope_dense(tmp_path):
    """End-to-end VideoHierPredictorModel on the big-step SQ VAE with RoPE +
    dense supervision + stability kit (exercises _copy_codebooks, schedule-from-VAE,
    config->stage wiring, forward_batch, and the _lasttok metric)."""
    import yaml

    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel
    from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel

    cfg_src = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_bigstep_pyr_smoke.yaml"
    with open(cfg_src, "r", encoding="utf-8") as f:
        vae_cfg = yaml.safe_load(f)
    vae = VideoHierVQModel(**vae_cfg["model"]["init_args"])
    ckpt = tmp_path / "mock_bigstep.ckpt"
    torch.save({"state_dict": vae.state_dict()}, ckpt)

    model = VideoHierPredictorModel(
        vae_config=cfg_src,
        vae_ckpt=str(ckpt),
        t_context=5,
        t_total=9,
        parent_conditioning="dual_stream",
        predictor=dict(
            dim=[64, 32, 32], n_heads=[4, 4, 4], n_layers=[1, 1, 1],
            temporal_windows=[-1, 3, 2], cross_spatial_window=1,
            output_mode="composite", pos_encoding="rope",
            norm_type="rmsnorm", mlp_type="swiglu", qk_norm=True,
            attention={
                "coarse": {"type": "full"},
                "mid": {"type": "factorized", "t_window": 3},
                "fine": {"type": "factorized", "t_window": 2, "spatial_window": 4},
            },
        ),
        loss=dict(lambda_ce=1.0, log_pred_mse=False, dense_supervision=True),
        inference=dict(mode="autoregressive", commit="argmax", parallel_mode="context"),
    )
    # geometry derived from the VAE config
    assert [model.schedule.native_t(i) for i in range(3)] == [3, 5, 9]
    assert [model.predictor_stages[i].rope is not None for i in range(3)] == [True, True, True]
    loss, out, _ = model.forward_batch(_video(), inference_mode="train")
    assert torch.isfinite(loss) and torch.isfinite(out.loss_ce)
    assert out.loss_ce_lasttok is not None and torch.isfinite(out.loss_ce_lasttok)
