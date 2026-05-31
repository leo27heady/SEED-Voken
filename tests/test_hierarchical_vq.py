"""Unit tests for hierarchical video VQ components."""

import os
import tempfile

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Decoder, Encoder
from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down
from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import GaussianSQQuantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.hier_elbo_loss import compute_hier_elbo_loss
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


def test_gaussian_sq_forward_backward():
    q = GaussianSQQuantizer(size_dict=32, dim_dict=16, flg_loss_continuous=True)
    z = torch.randn(2, 16, 3, 8, 8, requires_grad=True)
    var = torch.tensor([60.0])
    out = q(z, var_q_pos=var, flg_train=True)
    out.aux_loss.backward()
    assert z.grad is not None
    assert out.z_q.shape == z.shape


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


def test_sqvae2_topdown_end_to_end():
    dd = _ddconfig()
    enc = Encoder(**dd)
    dec = Decoder(**dd)
    hier = build_top_down(
        hierarchy_cfg={
            "mode": "sqvae2",
            "blocks_sq": "t2_h8_w8_x1,t3_h16_w16_x1",
            "tap_channels": {"t2_h8_w8": 64, "t3_h16_w16": 256},
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
    _, acts = enc(x, return_intermediates=True)
    z_q, results = hier(acts, flg_train=True)
    x_rec = dec(z_q)
    assert x_rec.shape == x.shape
    loss, _ = compute_hier_elbo_loss(x, x_rec, results)
    loss.backward()


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


def test_video_hier_vq_model_forward_and_log_images():
    model = VideoHierVQModel(
        ddconfig=_ddconfig(resolution=64),
        hierarchy={
            "mode": "sqvae2",
            "blocks_sq": "t2_h8_w8_x1,t3_h16_w16_x1",
            "tap_channels": {"t2_h8_w8": 64, "t3_h16_w16": 256},
        },
        quantizer={
            "type": "sq",
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
    assert "train/perplexity_layer_1" in log_dict

    batch = {"video": x}
    log = model.log_images(batch)
    assert log["inputs"].shape == x.shape
    assert log["reconstructions"].shape == x.shape
    assert any(k.startswith("progressive_") for k in log)


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
    test_rsq_topdown_end_to_end()
    test_sqvae2_topdown_end_to_end()
    test_lfq_rsq_raises()
    test_video_hier_vq_model_forward_and_log_images()
    test_video_viz_grid_labels()
    print("all tests passed")
