"""Tokenizer tests for the big-step pyramid (16/4/1 spatial @ 9/5/3 temporal).

Covers: u4 layer-string parsing/plumbing, the decoupled encoder temporal stride
(back-compat default + explicit list), factor-aware state-chain audit, and a full
encoder->topdown forward + decode_from_indices round-trip with the 1x1 global top.
"""

import pytest
import torch

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import (
    Encoder,
    _resolve_temporal_strides,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import (
    audit_encoder_taps,
    validate_hierarchy_taps,
    validate_state_chain,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.video_inj_topdown import SQVAE2TopDown


BIGSTEP_DD = dict(
    double_z=False, z_channels=16, resolution=32, in_channels=3, out_ch=3, ch=32,
    ch_mult=[1, 2, 2, 2, 4, 4], temporal_downsample=[False, False, True, False, True],
    num_res_blocks=1, num_groups=8,
)


def test_temporal_downsample_default_is_legacy():
    # default (None) reproduces the old ch_mult!=1 coupling exactly
    assert _resolve_temporal_strides([1, 2, 2, 4], None, 3) == [False, True, True]
    # explicit equivalent list gives the same flags
    assert _resolve_temporal_strides([1, 2, 2, 4], [False, True, True], 3) == [False, True, True]


def test_temporal_downsample_length_validated():
    with pytest.raises(ValueError):
        _resolve_temporal_strides([1, 2, 2, 4], [True, False], 3)


def test_default_encoder_byte_identical():
    """Encoder with temporal_downsample=None == legacy weights/outputs."""
    legacy_dd = dict(
        double_z=False, z_channels=16, resolution=32, in_channels=3, out_ch=3, ch=32,
        ch_mult=[1, 2, 2, 4], num_res_blocks=1, num_groups=8,
    )
    torch.manual_seed(0)
    enc_default = Encoder(**legacy_dd)
    torch.manual_seed(0)
    enc_explicit = Encoder(**legacy_dd, temporal_downsample=[False, True, True])
    enc_default.eval(); enc_explicit.eval()
    x = torch.randn(1, 3, 9, 32, 32)
    with torch.no_grad():
        a = enc_default(x)
        b = enc_explicit(x)
    assert torch.equal(a, b)


def test_bigstep_encoder_taps():
    audit = audit_encoder_taps(BIGSTEP_DD, sequence_length=9)
    # (C, T, H, W) at the three quantized taps
    assert audit["h16_w16"][1:] == (64, 9, 16, 16)
    assert audit["h4_w4"][1:] == (64, 5, 4, 4)
    assert audit["h1_w1"][1:] == (16, 3, 1, 1)


def test_bigstep_validate_taps_and_chain():
    keys = ["h1_w1", "h4_w4", "h16_w16"]
    audit = validate_hierarchy_taps(
        BIGSTEP_DD, 9, keys, dict(h1_w1=16, h4_w4=64, h16_w16=64)
    )
    validate_state_chain(audit, keys, [False, True, True], temporal_up=[2, 2], layer_factor=[1, 4, 4])
    # wrong factor (claiming x2) must fail loudly
    with pytest.raises(ValueError):
        validate_state_chain(audit, keys, [False, True, True], temporal_up=[2, 2], layer_factor=[1, 2, 2])


def _bigstep_topdown():
    hier = dict(
        mode="sqvae2", token_grid="pyramid", tap_key_format="spatial", sequence_length=9,
        latent_key="h1_w1", blocks_sq="h1_w1_x1,h4_w4_u4,h16_w16_u4", temporal_up=[2, 2],
        decoder_source="finest_state", temporal_align_mode="causal",
        tap_channels=dict(h1_w1=16, h4_w4=64, h16_w16=64),
    )
    q = dict(type="sq", prior="zero", size_dict=[512, 2048, 1024], dim_dict=[16, 64, 64],
             log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0))
    return SQVAE2TopDown(hier, q, z_channels=16, width=16)


def test_bigstep_topdown_roundtrip():
    torch.manual_seed(0)
    enc = Encoder(**BIGSTEP_DD)
    td = _bigstep_topdown()
    enc.eval(); td.eval()
    x = torch.randn(2, 3, 9, 32, 32)
    with torch.no_grad():
        _, acts = enc(x, return_intermediates=True)
        z_state, results = td(acts, flg_train=False, flg_quant_det=True)
        # finest z_state grid + per-layer native grids
        assert tuple(z_state.shape) == (2, 16, 9, 16, 16)
        assert [tuple(r.indices.shape[1:]) for r in results] == [(3, 1, 1), (5, 4, 4), (9, 16, 16)]
        # decode_from_indices reproduces the forward finest state exactly
        z_dec = td.decode_from_indices([r.indices for r in results], acts)
        assert torch.allclose(z_dec, z_state, atol=1e-4)
