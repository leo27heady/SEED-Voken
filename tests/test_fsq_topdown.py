"""FSQ integration with the pyramid topdown + full VideoHierVQModel e2e (Path 2).

Confirms FSQ composes with the progressive finest-state pyramid: native token grids,
exact decode_from_indices roundtrip, progressive partials, and a full-model
forward/backward where kl_total == 0 (FSQ has no auxiliary loss) and no
posterior_var is logged.
"""

import torch
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down
from src.Open_MAGVIT2.modules.vqvae.hierarchical.fsq import FSQLayerQuantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.hier_elbo_loss import compute_hier_elbo_loss

LITE_CFG = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml"
Z_CH = 16

PYR = {
    "mode": "sqvae2",
    "token_grid": "pyramid",
    "tap_key_format": "spatial",
    "latent_key": "h4_w4",
    "blocks_sq": "h4_w4_x1,h8_w8_u2,h16_w16_u2",
    "temporal_up": [2, 2],
    "decoder_source": "finest_state",
    "temporal_align_mode": "causal",
    "tap_channels": {"h4_w4": Z_CH, "h8_w8": 64, "h16_w16": 64},
}
FSQ = {"type": "fsq", "levels": [[8, 8, 4, 4], [8, 4, 4], [4, 4, 4]]}  # K = 1024 / 128 / 64


def _acts(b=2):
    torch.manual_seed(0)
    return {
        "h4_w4": torch.randn(b, Z_CH, 3, 4, 4),
        "h8_w8": torch.randn(b, 64, 5, 8, 8),
        "h16_w16": torch.randn(b, 64, 9, 16, 16),
    }, torch.randn(b, Z_CH, 3, 4, 4)


def _td():
    torch.manual_seed(0)
    return build_top_down(hierarchy_cfg=dict(PYR), quantizer_cfg=dict(FSQ), z_channels=Z_CH, width=16)


def test_fsq_layers_built():
    td = _td()
    assert all(isinstance(b.quantizer, FSQLayerQuantizer) for b in td.blocks)
    assert [td.level_metadata(i)["codebook_size"] for i in range(3)] == [1024, 128, 64]
    # all-FSQ => no SQ parameters in the optimizer (log_param_q_scalar is a buffer)
    assert not td._has_sq_layers


def test_fsq_token_grids_match_native_taps():
    td = _td()
    acts, bottleneck = _acts()
    with torch.no_grad():
        z_out, results = td(acts, encoder_bottleneck=bottleneck, flg_train=False, flg_quant_det=True)
    assert [tuple(r.indices.shape[1:]) for r in results] == [(3, 4, 4), (5, 8, 8), (9, 16, 16)]
    assert tuple(z_out.shape) == (2, Z_CH, 9, 16, 16)
    assert all(r.aux_loss.item() == 0.0 for r in results)


def test_fsq_decode_from_indices_roundtrip():
    td = _td()
    td.eval()
    acts, bottleneck = _acts()
    with torch.no_grad():
        z_out, results = td(acts, encoder_bottleneck=bottleneck, flg_train=False, flg_quant_det=True)
        z_dec = td.decode_from_indices([r.indices for r in results], acts, encoder_bottleneck=bottleneck)
    assert torch.allclose(z_out, z_dec, atol=1e-5), (z_out - z_dec).abs().max().item()


def test_fsq_mixed_per_layer_levels_validation():
    import pytest

    # one fsq layer but two level specs -> error (H1)
    bad = {"type": "sq", "per_layer": ["sq", "sq", "fsq"], "levels": [[4, 4], [4, 4]],
           "size_dict": [32, 32, 32], "dim_dict": [24, 24, 24], "log_param_q_init": [4.09434]}
    with pytest.raises(ValueError, match="levels length"):
        build_top_down(hierarchy_cfg=dict(PYR), quantizer_cfg=bad, z_channels=Z_CH, width=16)
    # fsq layer without levels -> error
    bad2 = {"type": "sq", "per_layer": ["sq", "fsq", "fsq"], "size_dict": [32, 32, 32],
            "dim_dict": [24, 24, 24], "log_param_q_init": [4.09434]}
    with pytest.raises(ValueError, match="require quantizer.levels"):
        build_top_down(hierarchy_cfg=dict(PYR), quantizer_cfg=bad2, z_channels=Z_CH, width=16)


def test_fsq_full_model_progressive_e2e():
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    hier = dict(PYR)
    hier["sequence_length"] = 9
    dec_cfg = dict(z_channels=Z_CH, ch=32, ch_mult=[1, 2], num_res_blocks=1,
                   num_groups=8, out_ch=3, in_channels=3, resolution=32, double_z=False)
    torch.manual_seed(0)
    model = VideoHierVQModel(
        ddconfig=cfg["ddconfig"], hierarchy=hier, quantizer=dict(FSQ),
        dec_ddconfig=dec_cfg, progressive_coding=True, use_ema=False,
    )
    x = torch.rand(1, 3, 9, 32, 32) * 2 - 1

    x_rec, results = model(x, flg_train=True, flg_quant_det=False)
    assert tuple(x_rec.shape) == (1, 3, 9, 32, 32)
    assert [tuple(r.indices.shape[1:]) for r in results] == [(3, 4, 4), (5, 8, 8), (9, 16, 16)]

    prog = model.decode_progressive(x, flg_quant_det=False, flg_train=True)
    assert len(prog) == 3 and all(tuple(p.shape) == (1, 3, 9, 32, 32) for p in prog)

    loss, log = compute_hier_elbo_loss(x, x_rec, results, progressive_recs=prog)
    assert log["loss/kl_total"].item() == 0.0          # FSQ aux == 0
    assert torch.allclose(log["loss/total"], log["loss/distortion"])
    assert not any(k.startswith("train/posterior_var") for k in log)  # FSQ has none
    loss.backward()  # gradients flow through encoder/decoder/in_proj
    assert any(p.grad is not None for p in model.encoder.parameters())
    fsq0 = model.hier_quant.blocks[0].quantizer
    assert fsq0.in_proj.weight.grad is not None and fsq0.in_proj.weight.grad.abs().sum() > 0


def test_progressive_loss_target_matches_val_target_unclamped():
    """Regression: the progressive training distortion must be on the UNCLAMPED
    recon (clamp=False), identical to the unclamped forward x_rec used by val.
    Clamping only the loss path let the decoder drift out of [-1,1], inflating the
    (unclamped) val_ema/mse ~800x while clamped visuals looked clean."""
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    hier = dict(PYR)
    hier["sequence_length"] = 9
    dec_cfg = dict(z_channels=Z_CH, ch=32, ch_mult=[1, 2], num_res_blocks=1,
                   num_groups=8, out_ch=3, in_channels=3, resolution=32, double_z=False)
    torch.manual_seed(0)
    model = VideoHierVQModel(
        ddconfig=cfg["ddconfig"], hierarchy=hier, quantizer=dict(FSQ),
        dec_ddconfig=dec_cfg, progressive_coding=True, use_ema=False,
    ).eval()
    x = torch.rand(1, 3, 9, 32, 32) * 2 - 1
    with torch.no_grad():
        prog_clamped = model.decode_progressive(x, flg_quant_det=True, clamp=True)
        prog_raw = model.decode_progressive(x, flg_quant_det=True, clamp=False)
        x_rec, _ = model(x, flg_train=False, flg_quant_det=True)  # val/forward target (unclamped)
    for c, r in zip(prog_clamped, prog_raw):
        assert (c <= 1.0).all() and (c >= -1.0).all()       # viz path bounded
        assert torch.allclose(c, r.clamp(-1, 1), atol=1e-6)  # clamp=True == raw.clamp
    # the loss target (progressive L3, unclamped) == the val target (forward x_rec)
    assert torch.allclose(prog_raw[-1], x_rec, atol=1e-5)


def test_fsq_train_equals_eval_full_recon():
    """Full-model train (no Gumbel) == eval recon: the soft/hard gap is gone."""
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    hier = dict(PYR)
    hier["sequence_length"] = 9
    dec_cfg = dict(z_channels=Z_CH, ch=32, ch_mult=[1, 2], num_res_blocks=1,
                   num_groups=8, out_ch=3, in_channels=3, resolution=32, double_z=False)
    torch.manual_seed(0)
    model = VideoHierVQModel(
        ddconfig=cfg["ddconfig"], hierarchy=hier, quantizer=dict(FSQ),
        dec_ddconfig=dec_cfg, use_ema=False,
    ).eval()
    x = torch.rand(1, 3, 9, 32, 32) * 2 - 1
    with torch.no_grad():
        rec_train, _ = model(x, flg_train=True, flg_quant_det=False)
        rec_eval, _ = model(x, flg_train=False, flg_quant_det=True)
    assert torch.allclose(rec_train, rec_eval, atol=1e-5)
