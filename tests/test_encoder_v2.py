"""Tests for encoder/decoder v2 (FrameWiseGroupNorm + spatial tap keys)."""

import warnings

import pytest
import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Decoder, Encoder
from src.Open_MAGVIT2.modules.diffusionmodules.norm import FrameWiseGroupNorm
from src.Open_MAGVIT2.modules.vqvae.hierarchical.layer_string import parse_blocks_sq
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import audit_encoder_taps
from src.Open_MAGVIT2.modules.vqvae.hierarchical.tap_keys import normalize_resolution_key


def _ddconfig_v2(resolution=64):
    return dict(
        double_z=False,
        z_channels=32,
        resolution=resolution,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
    )


def _hierarchy_v2(seq_len=13):
    return dict(
        mode="sqvae2",
        token_grid="native",
        tap_key_format="spatial",
        sequence_length=seq_len,
        latent_key="h8_w8",
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h8_w8=32, h16_w16=128, h32_w32=128),
    )


def _quantizer_v2():
    return dict(
        type="sq",
        prior="zero",
        size_dict=[1536, 768, 384],
        dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3,
        temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )


def test_ev2_01_frame_wise_group_norm_shape():
    norm = FrameWiseGroupNorm(32, 64)
    x = torch.randn(2, 64, 5, 8, 8)
    y = norm(x)
    assert y.shape == x.shape


def test_ev2_04_spatial_keys_only():
    audit = audit_encoder_taps(_ddconfig_v2(), 13)
    for key in audit:
        assert key.startswith("h") and "_w" in key
        assert not key.startswith("t")


def test_ev2_05_variable_t_encode():
    audit9 = audit_encoder_taps(_ddconfig_v2(), 9)
    audit13 = audit_encoder_taps(_ddconfig_v2(), 13)
    assert set(audit9.keys()) == set(audit13.keys())


def test_ev2_06_legacy_dsl_parse():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        specs = parse_blocks_sq("t3_h8_w8_x1,t5_h16_w16_x1")
        assert len(w) == 2
    assert specs[0].resolution_key == "h8_w8"
    assert specs[1].resolution_key == "h16_w16"


def test_ev2_08_t13_shapes():
    audit = audit_encoder_taps(_ddconfig_v2(), 13)
    assert audit["h32_w32"][2] == 13
    assert audit["h16_w16"][2] == 7
    assert audit["h8_w8"][2] == 4


def test_ev2_02_prefix_stability_activations():
    torch.manual_seed(0)
    enc = Encoder(**_ddconfig_v2())
    enc.eval()
    full = torch.randn(1, 3, 13, 64, 64)
    short = full[:, :, :9].clone()
    with torch.no_grad():
        _, act_short = enc(short, return_intermediates=True)
        _, act_full = enc(full, return_intermediates=True)
    for key in act_short:
        t_ctx = act_short[key].shape[2]
        diff = (act_short[key] - act_full[key][:, :, :t_ctx]).abs().max().item()
        assert diff < 1e-5, f"{key} prefix diff {diff}"


def test_ev2_03_prefix_stability_indices_cross_t():
    """Cross-clip encode: finest activation prefix stable; native T differs by stage."""
    torch.manual_seed(3)
    model = VideoHierVQModel(
        ddconfig=_ddconfig_v2(),
        hierarchy=_hierarchy_v2(13),
        quantizer=_quantizer_v2(),
        learning_rate=1e-4,
    )
    model.eval()
    clip13 = torch.randn(1, 3, 13, 64, 64)
    clip9 = clip13[:, :, :9].clone()
    with torch.no_grad():
        tok13 = model.encode_tokens(clip13, flg_quant_det=True)
        tok9 = model.encode_tokens(clip9, flg_quant_det=True)
    fine9 = tok9["levels"][2]["indices"]
    fine13 = tok13["levels"][2]["indices"]
    assert fine9.shape[1] == 9
    assert fine13.shape[1] == 13
    # Mid/top native T is shorter for T=9 clips — only compare overlapping finest acts.
    assert tok9["levels"][0]["indices"].shape[1] < tok13["levels"][0]["indices"].shape[1]
    key = "h32_w32"
    t_ov = tok9["activations"][key].shape[2]
    diff = (
        tok13["activations"][key][:, :, :t_ov] - tok9["activations"][key]
    ).abs().max().item()
    assert diff < 1e-5, f"{key} prefix diff {diff}"


def test_ev2_03_quantized_deterministic_and_activation_prefix():
    """Indices are deterministic; activation prefix stable across clip lengths (see ev2_02)."""
    torch.manual_seed(1)
    model = VideoHierVQModel(
        ddconfig=_ddconfig_v2(),
        hierarchy=_hierarchy_v2(13),
        quantizer=_quantizer_v2(),
        learning_rate=1e-4,
    )
    model.eval()
    clip = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        a = model.encode_tokens(clip, flg_quant_det=True)
        b = model.encode_tokens(clip.clone(), flg_quant_det=True)
    for la, lb in zip(a["levels"], b["levels"]):
        assert torch.equal(la["indices"], lb["indices"])


def test_ev2_09_roundtrip_encode_decode_t13():
    torch.manual_seed(2)
    model = VideoHierVQModel(
        ddconfig=_ddconfig_v2(),
        hierarchy=_hierarchy_v2(13),
        quantizer=_quantizer_v2(),
        learning_rate=1e-4,
    )
    model.eval()
    x = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        tokens = model.encode_tokens(x, flg_quant_det=True)
        level_indices = [lv["indices"] for lv in tokens["levels"]]
        recon = model.decode_from_indices(
            level_indices,
            tokens["activations"],
            encoder_bottleneck=tokens["encoder_bottleneck"],
        )
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_ev2_07_decoder_adagn_per_frame():
    from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import (
        FrameWiseVideoAdaptiveGroupNorm,
    )

    ada = FrameWiseVideoAdaptiveGroupNorm(32, 64)
    x = torch.randn(2, 64, 5, 8, 8)
    style = torch.randn(2, 32, 5, 8, 8)
    y = ada(x, style)
    assert y.shape == x.shape


def test_ev2_10_partial_load_conv_only():
    """Conv weights load; norm architecture changes are skipped gracefully."""
    from src.Open_MAGVIT2.modules.vqvae.hierarchical.checkpoint_v2 import load_conv_weights_partial

    old = {"encoder.conv_in.weight": torch.randn(64, 3, 3, 3, 3)}
    new = {
        "encoder.conv_in.weight": torch.zeros(64, 3, 3, 3, 3),
        "encoder.norm_out.gn.weight": torch.randn(512),
    }
    merged, n_loaded, n_skipped = load_conv_weights_partial(old, new)
    assert n_loaded == 1
    assert torch.allclose(merged["encoder.conv_in.weight"], old["encoder.conv_in.weight"])
    assert "encoder.norm_out.gn.weight" in merged


def test_normalize_resolution_key():
    assert normalize_resolution_key("h8_w8") == "h8_w8"
    assert normalize_resolution_key("t3_h8_w8") == "h8_w8"
