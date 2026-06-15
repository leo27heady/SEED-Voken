"""Progressive causal temporal-alignment fix (Path 2, P2).

The trilinear alignment of progressive partials mapped output frame 1 onto coarse
frame 0 (the causal-boundary frame), so the L1-only row was stale/fuzzy at t=1
(verified: decoded t1 MSE 8.7x t0 despite identical latent). 'causal' mode replays
the learned Upsampler chain instead. The fix lives ONLY in forward_progressive;
the full recon (forward / decode_from_indices) was already correct (B1).
"""

import torch

from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down

Z_CH = 16
PYR = {
    "mode": "sqvae2",
    "token_grid": "pyramid",
    "tap_key_format": "spatial",
    "latent_key": "h4_w4",
    "blocks_sq": "h4_w4_x1,h8_w8_u2,h16_w16_u2",
    "temporal_up": [2, 2],
    "decoder_source": "finest_state",
    "tap_channels": {"h4_w4": Z_CH, "h8_w8": 64, "h16_w16": 64},
}
QUANT = {
    "type": "sq",
    "prior": "zero",
    "size_dict": [32, 32, 32],
    "dim_dict": [24, 24, 24],
    "log_param_q_init": [4.09434],
    "temperature": {"init": 1.0},
}


def _td(mode="trilinear"):
    torch.manual_seed(0)
    h = dict(PYR)
    h["temporal_align_mode"] = mode
    return build_top_down(hierarchy_cfg=h, quantizer_cfg=dict(QUANT), z_channels=Z_CH, width=16)


def _acts(b=2):
    torch.manual_seed(1)
    return {
        "h4_w4": torch.randn(b, Z_CH, 3, 4, 4),
        "h8_w8": torch.randn(b, 64, 5, 8, 8),
        "h16_w16": torch.randn(b, 64, 9, 16, 16),
    }, torch.randn(b, Z_CH, 3, 4, 4)


def test_invalid_mode_rejected():
    import pytest

    with pytest.raises(ValueError, match="temporal_align_mode"):
        _td("bilinear")


def test_default_is_trilinear():
    assert build_top_down(
        hierarchy_cfg=dict(PYR), quantizer_cfg=dict(QUANT), z_channels=Z_CH, width=16
    ).temporal_align_mode == "trilinear"


def test_trilinear_duplicates_l1_frame0_and_1():
    """Reproduces the bug: trilinear puts coarse frame 0 on output frames 0 AND 1."""
    td = _td("trilinear")
    acts, bottleneck = _acts()
    with torch.no_grad():
        _, partial = td.forward_progressive(acts, encoder_bottleneck=bottleneck, flg_quant_det=True)
    l1 = partial[0]  # (b, Z, 9, 16, 16)
    assert torch.allclose(l1[:, :, 0], l1[:, :, 1], atol=1e-6)  # the artifact


def test_causal_fixes_l1_t1():
    """'causal' replay makes L1 frame 1 distinct from frame 0 (no stale duplicate)."""
    td = _td("causal")
    acts, bottleneck = _acts()
    with torch.no_grad():
        _, partial = td.forward_progressive(acts, encoder_bottleneck=bottleneck, flg_quant_det=True)
    l1 = partial[0]
    assert (l1[:, :, 0] - l1[:, :, 1]).abs().max() > 1e-4


def test_partials_on_finest_grid_both_modes():
    for mode in ("trilinear", "causal"):
        td = _td(mode)
        acts, bottleneck = _acts()
        with torch.no_grad():
            z_out, partial = td.forward_progressive(
                acts, encoder_bottleneck=bottleneck, flg_quant_det=True
            )
        assert len(partial) == 3
        for p in partial:
            assert tuple(p.shape) == (2, Z_CH, 9, 16, 16), (mode, p.shape)
        # final partial == the full finest z_state, unchanged by the fix
        assert torch.allclose(partial[-1], z_out, atol=1e-5)


def test_decode_from_indices_unaffected_by_mode():
    """B1: the gate scores decode_from_indices, which already uses the learned
    upsampler — it must be identical regardless of temporal_align_mode, and equal
    to the forward finest z_state."""
    acts, bottleneck = _acts()
    outs = {}
    for mode in ("trilinear", "causal"):
        td = _td(mode)
        td.eval()
        with torch.no_grad():
            z_out, results = td(acts, encoder_bottleneck=bottleneck, flg_train=False, flg_quant_det=True)
            z_dec = td.decode_from_indices([r.indices for r in results], acts, encoder_bottleneck=bottleneck)
        assert torch.allclose(z_out, z_dec, atol=1e-5)  # gate path == forward full path
        outs[mode] = z_dec
    # same seed/weights -> decode identical across modes (fix is isolated to progressive)
    assert torch.allclose(outs["trilinear"], outs["causal"], atol=1e-6)
