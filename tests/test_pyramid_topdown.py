"""Temporal pyramid + finest-state decoding tests (Path A, Phase 3).

Guards correction #1 from docs/PATH_A_IMPLEMENTATION_PLAN.md: u2 layers must
upsample time as well as space (T -> 2T - 1 per level) so per-level token grids
stay on the native tap grids and the predictor's 1/2/4 shift hierarchy holds.
"""

import pytest
import torch
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Decoder
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import validate_state_chain

LITE_CFG = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml"

Z_CH = 16

PYR_HIERARCHY = {
    "mode": "sqvae2",
    "token_grid": "pyramid",
    "tap_key_format": "spatial",
    "latent_key": "h4_w4",
    "blocks_sq": "h4_w4_x1,h8_w8_u2,h16_w16_u2",
    "temporal_up": [2, 2],
    "decoder_source": "finest_state",
    "tap_channels": {"h4_w4": Z_CH, "h8_w8": 64, "h16_w16": 64},
}

QUANT_CFG = {
    "type": "sq",
    "prior": "zero",
    "size_dict": [32, 32, 32],
    "dim_dict": [24, 24, 24],
    "log_param_q_init": [4.09434],
    "temperature": {"init": 1.0},
}


def _fake_acts(b=2):
    return {
        "h4_w4": torch.randn(b, Z_CH, 3, 4, 4),
        "h8_w8": torch.randn(b, 64, 5, 8, 8),
        "h16_w16": torch.randn(b, 64, 9, 16, 16),
    }, torch.randn(b, Z_CH, 3, 4, 4)


def _build_topdown():
    torch.manual_seed(0)
    return build_top_down(
        hierarchy_cfg=dict(PYR_HIERARCHY),
        quantizer_cfg=dict(QUANT_CFG),
        z_channels=Z_CH,
        width=16,
    )


def test_pyramid_token_grids_match_native_taps():
    td = _build_topdown()
    acts, bottleneck = _fake_acts()
    z_out, results = td(acts, encoder_bottleneck=bottleneck, flg_train=False, flg_quant_det=True)
    grids = [tuple(r.indices.shape[1:]) for r in results]
    assert grids == [(3, 4, 4), (5, 8, 8), (9, 16, 16)], grids
    assert tuple(z_out.shape) == (2, Z_CH, 9, 16, 16)


def test_pyramid_decode_from_indices_roundtrip():
    td = _build_topdown()
    td.eval()
    acts, bottleneck = _fake_acts()
    with torch.no_grad():
        z_out, results = td(
            acts, encoder_bottleneck=bottleneck, flg_train=False, flg_quant_det=True
        )
        indices = [r.indices for r in results]
        z_dec = td.decode_from_indices(indices, acts, encoder_bottleneck=bottleneck)
    assert torch.allclose(z_out, z_dec, atol=1e-5), (
        f"max diff {(z_out - z_dec).abs().max().item()}"
    )


def test_pyramid_progressive_partials_on_finest_grid():
    td = _build_topdown()
    acts, bottleneck = _fake_acts()
    with torch.no_grad():
        z_out, partial = td.forward_progressive(
            acts, encoder_bottleneck=bottleneck, flg_quant_det=True
        )
    assert len(partial) == 3
    for p in partial:
        assert tuple(p.shape) == (2, Z_CH, 9, 16, 16)
    assert torch.allclose(partial[-1], z_out)


def test_finest_state_decoder_shape():
    dec = Decoder(
        z_channels=Z_CH, ch=32, ch_mult=[1, 2], num_res_blocks=1,
        num_groups=8, out_ch=3, in_channels=3, resolution=32, double_z=False,
    )
    z = torch.randn(1, Z_CH, 9, 16, 16)
    with torch.no_grad():
        out = dec(z)
    assert tuple(out.shape) == (1, 3, 9, 32, 32)


def test_schedule_invariant_between_native_and_pyramid():
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    native = PyramidSchedule.from_encoder_audit(
        cfg["ddconfig"], cfg["hierarchy"], cfg["quantizer"], t_context=5, t_total=9
    )
    pyr_hier = dict(PYR_HIERARCHY)
    pyr = PyramidSchedule.from_encoder_audit(
        cfg["ddconfig"], pyr_hier, cfg["quantizer"], t_context=5, t_total=9
    )
    assert [s.spatial_key for s in native.stages] == [s.spatial_key for s in pyr.stages]
    assert [native.native_t(s) for s in range(3)] == [pyr.native_t(s) for s in range(3)]
    assert native._context_end == pyr._context_end


def test_log_param_q_max_clamps_variance():
    import math

    cfg = dict(QUANT_CFG)
    cfg["log_param_q_max"] = 4.5
    td = build_top_down(
        hierarchy_cfg=dict(PYR_HIERARCHY), quantizer_cfg=cfg, z_channels=Z_CH, width=16
    )
    with torch.no_grad():
        td.log_param_q_scalar.fill_(10.0)  # runaway scenario
    v = td._var_q_slice(0, td.num_layers - 1)
    assert v.max().item() <= math.exp(4.5) * (1 + 1e-5)
    # gradient must still flow below the ceiling
    with torch.no_grad():
        td.log_param_q_scalar.fill_(2.0)
    v = td._var_q_slice(0, td.num_layers - 1)
    assert torch.allclose(v, torch.full_like(v, math.exp(2.0)))


def test_state_chain_guard():
    audit = {
        "h4_w4": (1, Z_CH, 3, 4, 4),
        "h8_w8": (1, 64, 5, 8, 8),
        "h16_w16": (1, 64, 9, 16, 16),
    }
    keys = ["h4_w4", "h8_w8", "h16_w16"]
    ups = [False, True, True]
    validate_state_chain(audit, keys, ups, temporal_up=[2, 2])  # ok
    with pytest.raises(ValueError, match="temporal_up"):
        validate_state_chain(audit, keys, ups, temporal_up=[1, 1])


def test_spatial_only_u2_rejected_when_taps_halve_t():
    """Legacy spatial-only u2 against T-halving taps must fail loudly."""
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    hier = dict(PYR_HIERARCHY)
    hier["temporal_up"] = [1, 1]
    hier["sequence_length"] = 9
    with pytest.raises(ValueError, match="temporal_up"):
        VideoHierVQModel(
            ddconfig=cfg["ddconfig"], hierarchy=hier, quantizer=dict(QUANT_CFG),
            use_ema=False,
        )


def test_full_model_pyramid_end_to_end():
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    hier = dict(PYR_HIERARCHY)
    hier["sequence_length"] = 9
    dec_cfg = dict(
        z_channels=Z_CH, ch=32, ch_mult=[1, 2], num_res_blocks=1,
        num_groups=8, out_ch=3, in_channels=3, resolution=32, double_z=False,
    )
    torch.manual_seed(0)
    model = VideoHierVQModel(
        ddconfig=cfg["ddconfig"], hierarchy=hier, quantizer=dict(QUANT_CFG),
        dec_ddconfig=dec_cfg, use_ema=False,
    )
    x = torch.rand(1, 3, 9, 32, 32) * 2 - 1
    with torch.no_grad():
        x_rec, layer_results = model(x, flg_train=False, flg_quant_det=True)
    assert tuple(x_rec.shape) == (1, 3, 9, 32, 32)
    grids = [tuple(r.indices.shape[1:]) for r in layer_results]
    assert grids == [(3, 4, 4), (5, 8, 8), (9, 16, 16)], grids

    # encode_tokens / decode_from_indices path used by the predictor
    with torch.no_grad():
        tokens = model.encode_tokens(x, flg_quant_det=True)
        recon = model.decode_from_indices(
            [lv["indices"] for lv in tokens["levels"]],
            {k: torch.zeros_like(v) for k, v in tokens["activations"].items()},
            encoder_bottleneck=torch.zeros_like(tokens["encoder_bottleneck"]),
        )
    assert tuple(recon.shape) == (1, 3, 9, 32, 32)
