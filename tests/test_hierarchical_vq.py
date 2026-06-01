"""Unit tests for hierarchical video VQ components."""

import os
import tempfile

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Decoder, Encoder
from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down
from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import GaussianSQQuantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.hier_elbo_loss import compute_hier_elbo_loss
from src.Open_MAGVIT2.modules.vqvae.hierarchical.lfq_adapter import LFQAdapter
from src.Open_MAGVIT2.modules.vqvae.hierarchical.deterministic_vq import DeterministicVQQuantizer
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
        coarse, fine = "t2_h8_w8", "t5_h64_w64"
        taps = {coarse: 64, fine: 128}
    else:
        coarse, fine = "t3_h16_w16", "t9_h64_w64"
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


def test_gaussian_sq_perplexity_finite():
    q = GaussianSQQuantizer(size_dict=32, dim_dict=16)
    z = torch.randn(1, 16, 3, 8, 8)
    out = q(z, var_q_pos=torch.tensor([60.0]), flg_train=True)
    assert torch.isfinite(out.perplexity)
    assert out.perplexity.item() < 1e6


def test_gaussian_sq_prior_zero_vs_uniform():
    z = torch.randn(1, 16, 3, 4, 4)
    var = torch.tensor([60.0])
    q_zero = GaussianSQQuantizer(32, 16, prior="zero")
    q_uni = GaussianSQQuantizer(32, 16, prior="uniform")
    out_z = q_zero(z, var_q_pos=var, flg_train=True)
    out_u = q_uni(z, var_q_pos=var, flg_train=True)
    assert out_z.aux_loss.item() != out_u.aux_loss.item()


def test_vq_indices_shape():
    q = DeterministicVQQuantizer(512, 64)
    z = torch.randn(1, 64, 3, 16, 16)
    out = q(z, flg_train=True)
    assert out.indices.shape == (1, 3, 16, 16)


def test_lfq_adapter_with_projection():
    q = LFQAdapter(dim=64, codebook_size=512)
    z = torch.randn(1, 64, 3, 8, 8)
    out = q(z, flg_train=True)
    assert out.z_q.shape == z.shape
    assert out.indices.shape == (1, 3, 8, 8)


def test_rsq_topdown_end_to_end():
    dd = _ddconfig()
    enc = Encoder(**dd)
    dec = Decoder(**dd)
    hier = build_top_down(
        hierarchy_cfg={"mode": "rsqvae", "num_layers": 2, "blocks_sq": "8x2"},
        quantizer_cfg={
            "type": "sq",
            "size_dict": [64],
            "dim_dict": [64],
            "log_param_q_init": [4.09434],
        },
        z_channels=64,
    )
    x = torch.randn(1, 3, 5, 64, 64)
    h = enc(x)
    z_q, results = hier(h, flg_train=True)
    assert z_q.shape == h.shape
    x_rec = dec(z_q)
    assert x_rec.shape == x.shape
    loss, _ = compute_hier_elbo_loss(x, x_rec, results)
    loss.backward()


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


def test_pyramid_progressive_matches_forward():
    dd = _ddconfig(64)
    enc = Encoder(**dd)
    hier = build_top_down(
        hierarchy_cfg={
            "mode": "sqvae2",
            "token_grid": "pyramid",
            "latent_key": "t2_h8_w8",
            "blocks_sq": "t2_h8_w8_x1,t5_h64_w64_u2",
            "tap_channels": {"t2_h8_w8": 64, "t5_h64_w64": 128},
        },
        quantizer_cfg={
            "type": "sq",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        z_channels=64,
        width=64,
    )
    x = torch.randn(1, 3, 5, 64, 64)
    h, acts = enc(x, return_intermediates=True)
    z_fwd, _ = hier(acts, encoder_bottleneck=h, flg_train=False, flg_quant_det=True)
    _, partial = hier.forward_progressive(
        acts, encoder_bottleneck=h, flg_quant_det=True, flg_train=False
    )
    assert (partial[-1] - z_fwd).abs().max().item() < 1e-5


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


def test_rsq_encode_tokens_api():
    model = VideoHierVQModel(
        ddconfig=_ddconfig(64),
        hierarchy={"mode": "rsqvae", "num_layers": 2, "blocks_sq": "8x2"},
        quantizer={
            "type": "sq",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        use_ema=False,
    )
    x = torch.randn(1, 3, 5, 64, 64)
    tok = model.encode_tokens(x)
    assert len(tok["levels"]) == 2
    assert tok["activations"] is None
    indices = [level["indices"] for level in tok["levels"]]
    x_rec = model.decode_from_indices(indices)
    assert x_rec.shape == x.shape


def test_vq_only_no_trainable_log_param_q():
    hier = build_top_down(
        hierarchy_cfg=_native_hierarchy(64),
        quantizer_cfg={
            "type": "vq",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
        },
        z_channels=64,
        width=64,
    )
    assert not hier._has_sq_layers
    assert not any(p is hier.log_param_q_scalar for p in hier.parameters())


def test_sqvae2_pyramid_u2_doubles_spatial():
    hier = build_top_down(
        hierarchy_cfg={
            "mode": "sqvae2",
            "token_grid": "pyramid",
            "latent_key": "t2_h8_w8",
            "blocks_sq": "t5_h64_w64_x1,t5_h64_w64_u2",
            "tap_channels": {"t5_h64_w64": 128},
        },
        quantizer_cfg={
            "type": "sq",
            "size_dict": [64, 64],
            "dim_dict": [64, 64],
            "log_param_q_init": [4.09434, 4.09434],
        },
        z_channels=64,
        width=64,
    )
    inj = hier.blocks[1]
    assert hasattr(inj, "spatial_up")
    z = torch.zeros(1, 64, 5, 32, 32)
    z_up = inj.spatial_up(z)
    assert z_up.shape[3:] == (64, 64)


def test_lfq_rsq_raises():
    try:
        build_top_down(
            hierarchy_cfg={"mode": "rsqvae", "num_layers": 2},
            quantizer_cfg={"type": "lfq", "size_dict": [256], "dim_dict": [18]},
            z_channels=64,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


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
    assert any(k.startswith("progressive_") for k in log)


def test_shape_audit_t5_64():
    audit = audit_encoder_taps(_ddconfig(64), sequence_length=5)
    assert "t2_h8_w8" in audit
    assert "t5_h64_w64" in audit


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
    test_gaussian_sq_perplexity_finite()
    test_gaussian_sq_prior_zero_vs_uniform()
    test_vq_indices_shape()
    test_lfq_adapter_with_projection()
    test_rsq_topdown_end_to_end()
    test_sqvae2_native_multilevel_shapes()
    test_pyramid_progressive_matches_forward()
    test_progressive_arelbo_loss_finite()
    test_rsq_encode_tokens_api()
    test_vq_only_no_trainable_log_param_q()
    test_sqvae2_pyramid_u2_doubles_spatial()
    test_lfq_rsq_raises()
    test_encode_tokens_api()
    test_video_hier_vq_model_forward_and_log_images()
    test_shape_audit_t5_64()
    test_video_viz_grid_labels()
    print("all tests passed")
