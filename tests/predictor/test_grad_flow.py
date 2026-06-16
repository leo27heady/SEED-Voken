"""Gradient-flow regression tests (Path A, Phase 2).

Guards review §2 findings:
- every trainable predictor parameter receives gradient in train mode
  (the @torch.no_grad in batch_prep used to freeze codebook_proj at random init)
- pred-MSE is logging-only (no gradient, not part of the optimized loss)
- AR mode and train mode are the same computation at shift 0
"""

import pytest
import torch

from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

LITE_CFG = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml"


def _build_model(*, cross_spatial_window: int, fine_t_window: int):
    torch.manual_seed(0)
    m = VideoHierPredictorModel(
        vae_config=LITE_CFG,
        vae_ckpt=None,
        freeze_encoder=True,
        t_context=5,
        t_total=9,
        parent_conditioning="dual_stream",
        predictor=dict(
            dim=[64, 32, 16], n_layers=[2, 2, 1], n_heads=[4, 4, 4],
            temporal_windows=[-1, 3, fine_t_window],
            cross_spatial_window=cross_spatial_window,
            attention={"coarse": {"type": "full"},
                       "mid": {"type": "factorized", "t_window": 3},
                       "fine": {"type": "factorized", "t_window": fine_t_window,
                                "spatial_window": 4}},
        ),
        loss=dict(lambda_ce=1.0, log_pred_mse=True, shift_ce_weights="uniform"),
        inference=dict(mode="autoregressive", commit="argmax", parallel_mode="context"),
    )
    m.train()
    # Break the zero-init of output heads / residual branches so gradients can
    # reach the input side (mimics the state after a few optimizer steps).
    with torch.no_grad():
        for p in m.predictor_stages.parameters():
            if p.requires_grad and p.abs().sum() == 0:
                p.add_(torch.randn_like(p) * 0.02)
    return m


@pytest.fixture(scope="module")
def model():
    """Recommended Path-A settings: 3x3 parent cross-window, fine t_window 2."""
    return _build_model(cross_spatial_window=1, fine_t_window=2)


@pytest.fixture(scope="module")
def video():
    torch.manual_seed(1)
    return (torch.randn(2, 3, 9, 32, 32)).clamp(-1, 1)


def test_all_trainable_params_receive_grad(model, video):
    model.zero_grad(set_to_none=True)
    loss, out, _ = model.forward_batch(
        video, inference_mode="train", compute_pred_mse=False
    )
    loss.backward()

    missing = []
    for name, p in model.predictor_stages.named_parameters():
        if not p.requires_grad:
            continue
        # horizon temporal_pos slots are a known Path-C gap (review §2.1#2):
        # train mode never consumes positions beyond the context, so only the
        # context slice is asserted here.
        if "temporal_pos" in name:
            continue
        if p.grad is None or p.grad.abs().sum() == 0:
            missing.append(name)
    assert not missing, f"parameters with no gradient: {missing}"


def test_codebook_proj_receives_grad(model, video):
    model.zero_grad(set_to_none=True)
    loss, _, _ = model.forward_batch(
        video, inference_mode="train", compute_pred_mse=False
    )
    loss.backward()
    for s, stage in enumerate(model.predictor_stages):
        proj = stage.codebook_proj
        if not hasattr(proj, "weight"):
            continue  # Identity when codebook_dim == dim
        assert proj.weight.grad is not None, f"stage {s} codebook_proj grad is None"
        assert proj.weight.grad.abs().sum() > 0, f"stage {s} codebook_proj grad all-zero"


def test_context_temporal_pos_receives_grad(model, video):
    model.zero_grad(set_to_none=True)
    loss, _, _ = model.forward_batch(
        video, inference_mode="train", compute_pred_mse=False
    )
    loss.backward()
    for s, stage in enumerate(model.predictor_stages):
        ctx_end = model.schedule._context_end[s]
        g = stage.temporal_pos.grad
        assert g is not None, f"stage {s} temporal_pos grad is None"
        assert g[0, : ctx_end + 1].abs().sum() > 0, f"stage {s} context temporal_pos all-zero"


def test_pred_mse_is_logging_only(model, video):
    loss, out, _ = model.forward_batch(
        video, inference_mode="train", compute_pred_mse=True
    )
    assert out.loss_mse is not None
    assert not out.loss_mse.requires_grad
    # optimized loss is CE-only
    assert torch.allclose(loss, model.lambda_ce * out.loss_ce)


def test_legacy_one_to_one_cross_mask_kills_qk_grads(video):
    """Documents the review follow-up finding: with the legacy 1-to-1 spatial
    cross mask and fine t_window=1, the supervised query frame has exactly one
    allowed parent key, so softmax is constant and fine-stage cross q/k get
    EXACT zero gradient. cross_spatial_window>=1 is the fix (see `model`)."""
    legacy = _build_model(cross_spatial_window=0, fine_t_window=1)
    legacy.zero_grad(set_to_none=True)
    loss, _, _ = legacy.forward_batch(
        video, inference_mode="train", compute_pred_mse=False
    )
    loss.backward()
    fine = legacy.predictor_stages[2]
    # CrossConditioning (§5.5) renamed the explicit q/k/v projections from
    # cross_q_f/cross_k_f/cross_v_f to cross_f.q/cross_f.k/cross_f.v (+ _g).
    qk_sum = sum(
        p.grad.abs().sum().item()
        for name, p in fine.named_parameters()
        if "cross" in name and (".q." in name or ".k." in name) and p.grad is not None
    )
    v_sum = sum(
        p.grad.abs().sum().item()
        for name, p in fine.named_parameters()
        if "cross" in name and ".v." in name and p.grad is not None
    )
    assert qk_sum == 0.0, "legacy mask unexpectedly produced q/k gradient"
    assert v_sum > 0.0


def test_train_and_ar_identical_at_shift_zero(model, video):
    model.eval()
    try:
        with torch.no_grad():
            _, out_train, _ = model.forward_batch(
                video, inference_mode="train", compute_pred_mse=False
            )
            _, out_ar, _ = model.forward_batch(
                video, inference_mode="autoregressive", compute_pred_mse=False
            )
        for s in range(model.schedule.S):
            ce_t = out_train.ce_breakdown[(s, 0)]
            ce_a = out_ar.ce_breakdown[(s, 0)]
            assert torch.allclose(ce_t, ce_a, atol=1e-6), (
                f"stage {s} shift-0 CE differs between train ({ce_t}) and AR ({ce_a})"
            )
    finally:
        model.train()
