"""Unit tests for hierarchical video VQ components."""

import os
import tempfile

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel, _layer_codebook_usage_logs
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Decoder, Encoder
from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down
from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import (
    GaussianSQQuantizer,
    calc_distance,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.hier_elbo_loss import compute_hier_elbo_loss
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import audit_encoder_taps
from src.Open_MAGVIT2.utils.video_viz import make_comparison_grid, videos_to_row_dict


def _ddconfig(resolution=64):
    return dict(
        ch=128,
        in_channels=3,
        num_res_blocks=2,
        z_channels=64,
        out_ch=3,
        resolution=resolution,
        ch_mult=[1, 2, 2, 4],
    )


def _native_hierarchy(resolution=64, latent_key=None):
    if resolution == 64:
        coarse, fine = "h8_w8", "h64_w64"
        taps = {coarse: 64, fine: 128}
    else:
        coarse, fine = "h16_w16", "h64_w64"
        taps = {coarse: 64, fine: 256}
    if latent_key is None:
        latent_key = coarse
    return {
        "mode": "sqvae2",
        "token_grid": "native",
        "latent_key": latent_key,
        "blocks_sq": f"{coarse}_x1,{fine}_x1",
        "tap_channels": taps,
    }


def test_gaussian_sq_forward_backward():
    q = GaussianSQQuantizer(size_dict=32, dim_dict=16, flg_loss_continuous=True)
    z = torch.randn(2, 16, 3, 8, 8, requires_grad=True)
    var = torch.tensor([60.0])
    out = q(z, var_q_pos=var, flg_train=True)
    out.aux_loss.backward()
    assert z.grad is not None
    assert out.z_q.shape == z.shape


def test_calc_distance_stays_finite_fp32_under_large_residual():
    """Crash-scale hierarchical residual (D=96, |z|~75) must not overflow to inf."""
    d, k = 96, 2048
    z = torch.full((1, 1, 1, 1, d), 74.94, dtype=torch.float16)
    cb = torch.randn(k, d)
    distances = calc_distance(z, cb)
    assert distances.dtype == torch.float32
    assert torch.isfinite(distances).all()
    assert distances.max().item() > 65504.0

    q = GaussianSQQuantizer(size_dict=k, dim_dict=d, in_channels=16)
    z5d = torch.full((2, 16, 5, 8, 8), 75.0, dtype=torch.float16)
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        out = q(z5d, var_q_pos=torch.tensor([60.0]), flg_train=True, flg_quant_det=True)
    assert torch.isfinite(out.aux_loss).item()
    assert torch.isfinite(out.perplexity).item()


def test_gaussian_sq_large_codebook_with_in_channels_proj():
    """in_channels != dim_dict: any size_dict must not break distance reshape."""
    bs, t_len, h, w = 16, 3, 4, 4
    q = GaussianSQQuantizer(
        size_dict=4096,
        dim_dict=96,
        in_channels=16,
        flg_loss_continuous=True,
    )
    z = torch.randn(bs, 16, t_len, h, w, requires_grad=True)
    out = q(z, var_q_pos=torch.tensor([60.0]), flg_train=True, flg_quant_det=True)
    out.aux_loss.backward()
    assert out.z_q.shape == z.shape
    assert out.indices.shape == (bs, t_len, h, w)


def test_perplexity_uniform_and_peaked():
    k = 32
    bs, t_len, h, w = 2, 3, 4, 4
    uniform = torch.full((bs, t_len, h, w, k), 1.0 / k)
    peaked = torch.zeros(bs, t_len, h, w, k)
    peaked[..., 0] = 1.0

    def _perplexity_from_prob(prob_pos):
        p = prob_pos.mean(dim=(0, 1, 2, 3))
        return torch.exp(-torch.sum(p * torch.log(p + 1e-7)))

    ppx_uni = _perplexity_from_prob(uniform)
    ppx_peak = _perplexity_from_prob(peaked)
    assert abs(ppx_uni.item() - k) < 0.5
    assert ppx_peak.item() < 1.5

    q = GaussianSQQuantizer(size_dict=k, dim_dict=8)
    z = torch.randn(bs, 8, t_len, h, w)
    out_uni = q(z, var_q_pos=torch.tensor([60.0]), flg_train=False, flg_quant_det=True)
    assert out_uni.perplexity.item() > k * 0.5


def test_kld_discrete_mean_grid_invariant():
    """Mean over (T,H,W) makes discrete KL independent of grid size for uniform posteriors."""
    k = 16
    var = torch.tensor([60.0])

    def discrete_kl(t_len, h, w):
        q = GaussianSQQuantizer(size_dict=k, dim_dict=8)
        z = torch.randn(2, 8, t_len, h, w)
        out = q(z, var_q_pos=var, flg_train=False, flg_quant_det=True)
        return out.aux_loss.item()

    kl_small = discrete_kl(2, 4, 4)
    kl_large = discrete_kl(5, 16, 16)
    assert abs(kl_small - kl_large) < 0.5 * max(abs(kl_small), abs(kl_large), 1.0)


def test_gaussian_sq_perplexity_finite():
    q = GaussianSQQuantizer(size_dict=32, dim_dict=16)
    z = torch.randn(1, 16, 3, 8, 8)
    out = q(z, var_q_pos=torch.tensor([60.0]), flg_train=True)
    assert torch.isfinite(out.perplexity)
    assert out.perplexity.item() < 1e6


def test_gaussian_sq_logs_active_codes():
    q = GaussianSQQuantizer(size_dict=32, dim_dict=16)
    z = torch.randn(2, 16, 3, 8, 8)
    out = q(z, var_q_pos=torch.tensor([60.0]), flg_train=True)
    ac = int(out.log_stats["active_codes"].item())
    uf = float(out.log_stats["usage_fraction"].item())
    assert 1 <= ac <= 32
    assert abs(uf - ac / 32.0) < 1e-5


def test_flg_loss_continuous_false_disables_continuous_kl():
    hier_auto = build_top_down(
        hierarchy_cfg=_native_hierarchy(64),
        quantizer_cfg={
            "type": "sq",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "flg_loss_continuous": "auto",
        },
        z_channels=64,
        width=64,
    )
    assert hier_auto.blocks[1].quantizer.flg_loss_continuous is True

    hier_off = build_top_down(
        hierarchy_cfg=_native_hierarchy(64),
        quantizer_cfg={
            "type": "sq",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "flg_loss_continuous": False,
        },
        z_channels=64,
        width=64,
    )
    assert hier_off.blocks[0].quantizer.flg_loss_continuous is False
    assert hier_off.blocks[1].quantizer.flg_loss_continuous is False


def test_layer_codebook_usage_logs_helper():
    q = GaussianSQQuantizer(size_dict=16, dim_dict=8)
    z = torch.randn(1, 8, 3, 4, 4)
    out = q(z, var_q_pos=torch.tensor([60.0]), flg_train=True)
    logs = _layer_codebook_usage_logs([out], "train")
    assert "train/active_codes_layer_1" in logs
    assert "train/code_usage_frac_layer_1" in logs
    assert "train/perplexity_frac_layer_1" in logs


def test_mse_per_pixel_logging():
    x = torch.randn(1, 3, 2, 8, 8)
    x_rec = x + 0.1 * torch.randn_like(x)
    q = GaussianSQQuantizer(16, 8)
    r = q(torch.randn(1, 8, 2, 4, 4), var_q_pos=torch.tensor([60.0]), flg_train=True)
    _, log = compute_hier_elbo_loss(x, x_rec, [r])
    assert "loss/mse_per_pixel" in log
    assert log["loss/mse_per_pixel"].item() < log["loss/mse"].item()


def test_gaussian_sq_prior_zero_vs_uniform():
    z = torch.randn(1, 16, 3, 4, 4)
    var = torch.tensor([60.0])
    q_zero = GaussianSQQuantizer(32, 16, prior="zero")
    q_uni = GaussianSQQuantizer(32, 16, prior="uniform")
    out_z = q_zero(z, var_q_pos=var, flg_train=True)
    out_u = q_uni(z, var_q_pos=var, flg_train=True)
    assert out_z.aux_loss.item() != out_u.aux_loss.item()


def test_gaussian_sq_learned_prior():
    z = torch.randn(1, 16, 3, 4, 4)
    z_pri = torch.randn(1, 16, 3, 4, 4)
    var = torch.tensor([60.0])
    q = GaussianSQQuantizer(32, 16, prior="learned")
    out = q(z, var_q_pos=var, z_pri=z_pri, var_q_pri=var, flg_train=True)
    assert torch.isfinite(out.aux_loss)


def test_kl_weights_scaling():
    x = torch.randn(1, 3, 2, 8, 8)
    x_rec = x + 0.1 * torch.randn_like(x)
    q = GaussianSQQuantizer(16, 8)
    var = torch.tensor([60.0])
    z = torch.randn(1, 8, 2, 4, 4)
    r1 = q(z, var_q_pos=var, flg_train=True)
    r2 = q(z, var_q_pos=var, flg_train=True)
    w = torch.tensor([1.0, 3.0])
    _, log = compute_hier_elbo_loss(x, x_rec, [r1, r2], kl_weights=w)
    assert abs(log["loss/kl_layer_2"].item() - 3.0 * r2.aux_loss.item()) < 1e-4
    assert abs(log["loss/kl_layer_1_raw"].item() - r1.aux_loss.item()) < 1e-4


def test_usage_regularizer_active():
    q = GaussianSQQuantizer(
        32, 16, usage_reg_weight=0.5, usage_reg_target_perplexity=10.0
    )
    z = torch.randn(1, 16, 3, 8, 8)
    out = q(z, var_q_pos=torch.tensor([60.0]), flg_train=True)
    q_off = GaussianSQQuantizer(32, 16)
    out_off = q_off(z, var_q_pos=torch.tensor([60.0]), flg_train=True)
    assert out.aux_loss.item() > out_off.aux_loss.item()


def test_sqvae2_three_levels_forward_shapes():
    hier = build_top_down(
        hierarchy_cfg={
            "mode": "sqvae2",
            "token_grid": "native",
            "latent_key": "h16_w16",
            "blocks_sq": "h16_w16_x1,h32_w32_x1,h64_w64_x1",
            "tap_channels": {
                "h16_w16": 64,
                "h32_w32": 64,
                "h64_w64": 128,
            },
        },
        quantizer_cfg={
            "type": "sq",
            "prior": "zero",
            "size_dict": [64, 64, 64],
            "dim_dict": [32, 32, 32],
            "log_param_q_init": [4.09434, 4.09434, 4.09434],
        },
        z_channels=32,
        width=32,
    )
    assert hier.num_layers == 3
    acts = {
        "h16_w16": torch.randn(1, 64, 3, 16, 16),
        "h32_w32": torch.randn(1, 64, 5, 32, 32),
        "h64_w64": torch.randn(1, 128, 9, 64, 64),
    }
    h = torch.randn(1, 32, 3, 16, 16)
    z_q, results = hier(acts, encoder_bottleneck=h, flg_train=True)
    assert z_q.shape == h.shape
    assert len(results) == 3


def test_sqvae2_learned_prior_chain_builds_heads():
    hier = build_top_down(
        hierarchy_cfg={
            "mode": "sqvae2",
            "token_grid": "native",
            "latent_key": "t3_h16_w16",
            "blocks_sq": "t3_h16_w16_x1,t5_h32_w32_x1",
            "tap_channels": {"t3_h16_w16": 64, "t5_h32_w32": 64},
        },
        quantizer_cfg={
            "type": "sq",
            "prior": {"mode": "learned_chain", "width": 16},
            "size_dict": [32, 32],
            "dim_dict": [16, 16],
            "log_param_q_init": [4.09434, 4.09434],
        },
        z_channels=64,
        width=16,
    )
    assert len(hier.prior_heads) == 2
    assert hier.blocks[0].quantizer.prior == "learned"


def test_sqvae2_native_multilevel_shapes():
    dd = _ddconfig()
    enc = Encoder(**dd)
    dec = Decoder(**dd)
    hier = build_top_down(
        hierarchy_cfg=_native_hierarchy(64),
        quantizer_cfg={
            "type": "sq",
            "prior": "zero",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        z_channels=64,
        width=64,
    )
    x = torch.randn(1, 3, 5, 64, 64)
    h, acts = enc(x, return_intermediates=True)
    z_q, results = hier(acts, encoder_bottleneck=h, flg_train=True)
    assert z_q.shape == h.shape
    assert results[0].indices.shape[1:] == (2, 8, 8)
    assert results[1].indices.shape[1:] == (5, 64, 64)
    assert results[0].indices.shape != results[1].indices.shape
    x_rec = dec(z_q)
    loss, _ = compute_hier_elbo_loss(x, x_rec, results)
    loss.backward()


def test_progressive_arelbo_loss_finite():
    model = VideoHierVQModel(
        ddconfig=_ddconfig(64),
        hierarchy=_native_hierarchy(64),
        quantizer={
            "type": "sq",
            "prior": "zero",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        use_ema=False,
        progressive_coding=True,
    )
    x = torch.randn(1, 3, 5, 64, 64)
    x_rec, layer_results = model(x, flg_train=True)
    progressive_recs = model.decode_progressive(x, flg_quant_det=False, flg_train=True)
    loss, log_dict = compute_hier_elbo_loss(
        x, x_rec, layer_results, progressive_recs=progressive_recs
    )
    assert torch.isfinite(loss)
    assert "loss/mse_progressive_L1" in log_dict
    loss.backward()


def test_encode_tokens_api():
    model = VideoHierVQModel(
        ddconfig=_ddconfig(resolution=64),
        hierarchy=_native_hierarchy(64),
        quantizer={
            "type": "sq",
            "prior": "zero",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        use_ema=False,
        progressive_coding=True,
    )
    x = torch.randn(1, 3, 5, 64, 64)
    tok = model.encode_tokens(x)
    assert len(tok["levels"]) == 2
    assert tok["levels"][0]["shape"] == (2, 8, 8)
    assert tok["levels"][1]["shape"] == (5, 64, 64)
    assert tok["z_q"].ndim == 5


def test_video_hier_vq_model_forward_and_log_images():
    model = VideoHierVQModel(
        ddconfig=_ddconfig(resolution=64),
        hierarchy=_native_hierarchy(64),
        quantizer={
            "type": "sq",
            "prior": "zero",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        use_ema=False,
    )
    x = torch.randn(2, 3, 5, 64, 64)
    x_rec, layer_results = model(x, flg_train=True)
    assert x_rec.shape == x.shape
    loss, log_dict = compute_hier_elbo_loss(x, x_rec, layer_results)
    assert torch.isfinite(loss)
    assert torch.isfinite(log_dict["train/perplexity_layer_1"])

    batch = {"video": x}
    log = model.log_images(batch)
    assert log["inputs"].shape == x.shape
    has_progressive = any(k.startswith("progressive_") for k in log)
    if model.progressive_coding:
        assert has_progressive


def test_shape_audit_t5_64():
    audit = audit_encoder_taps(_ddconfig(64), sequence_length=5)
    assert "h8_w8" in audit
    assert "h64_w64" in audit


def test_shape_audit_three_level_64_v2_taps():
    """Spatial tap keys for shapes3d_sqvae2_64_S_v2 (T=13, 3-level)."""
    from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import (
        validate_hierarchy_taps,
    )

    dd = dict(
        double_z=False,
        z_channels=32,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
    )
    keys = ["h8_w8", "h16_w16", "h32_w32"]
    taps = {
        "h8_w8": 32,
        "h16_w16": 128,
        "h32_w32": 128,
    }
    validate_hierarchy_taps(dd, 13, keys, taps)


def test_validate_hierarchy_taps_rejects_plan_style_names():
    """Roadmap-style names (t3_h4_w4, …) are not valid unless the encoder emits them."""
    from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import (
        validate_hierarchy_taps,
    )

    dd = dict(
        double_z=False,
        z_channels=32,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
    )
    try:
        validate_hierarchy_taps(
            dd,
            13,
            ["t3_h4_w4", "t5_h8_w8", "t9_h16_w16", "t13_h32_w32"],
        )
        raise AssertionError("expected ValueError for missing tap keys")
    except ValueError as e:
        assert "Missing" in str(e) or "t3_h4_w4" in str(e)


def test_hier_video_recon_loss_perceptual():
    from src.Open_MAGVIT2.modules.losses.hier_video_loss import HierVideoReconLoss

    x = torch.randn(1, 3, 5, 32, 32, requires_grad=True)
    x_rec = x + 0.05 * torch.randn_like(x)
    loss_mod = HierVideoReconLoss(perceptual_weight=0.1)
    cb = torch.tensor(0.0)
    g_loss, g_log = loss_mod(x, x_rec, cb, 0, 0)
    assert torch.isfinite(g_loss)
    assert "train/perceptual" in g_log


def test_video_viz_grid_labels():
    x = torch.randn(3, 5, 32, 32)
    rec = torch.randn(3, 5, 32, 32)
    prog = {"progressive_L1": rec, "progressive_L1-L2": rec}
    rows = videos_to_row_dict(x, rec, prog)
    png = make_comparison_grid(rows, cell_size=32, label_width=120)
    assert png.size[0] > png.size[1] // 2
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "grid.png")
        png.save(path)
        assert os.path.getsize(path) > 0


if __name__ == "__main__":
    test_gaussian_sq_forward_backward()
    test_perplexity_uniform_and_peaked()
    test_kld_discrete_mean_grid_invariant()
    test_gaussian_sq_perplexity_finite()
    test_gaussian_sq_logs_active_codes()
    test_flg_loss_continuous_false_disables_continuous_kl()
    test_layer_codebook_usage_logs_helper()
    test_gaussian_sq_prior_zero_vs_uniform()
    test_gaussian_sq_learned_prior()
    test_kl_weights_scaling()
    test_usage_regularizer_active()
    test_sqvae2_three_levels_forward_shapes()
    test_sqvae2_learned_prior_chain_builds_heads()
    test_mse_per_pixel_logging()
    test_sqvae2_native_multilevel_shapes()
    test_progressive_arelbo_loss_finite()
    test_encode_tokens_api()
    test_video_hier_vq_model_forward_and_log_images()
    test_shape_audit_t5_64()
    test_video_viz_grid_labels()
    print("all tests passed")
