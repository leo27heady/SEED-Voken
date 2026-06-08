"""Validation gates for SQ-VAE-2 v2 before predictor training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Encoder
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import audit_encoder_taps


def build_model():
    ddconfig = dict(
        double_z=False,
        z_channels=32,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
    )
    hierarchy = dict(
        mode="sqvae2",
        token_grid="native",
        tap_key_format="spatial",
        sequence_length=13,
        latent_key="h8_w8",
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h8_w8=32, h16_w16=128, h32_w32=128),
    )
    quantizer = dict(
        type="sq",
        prior="zero",
        size_dict=[1536, 768, 384],
        dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3,
        temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig,
        hierarchy=hierarchy,
        quantizer=quantizer,
        learning_rate=1e-4,
    )


def check_prefix_stability(ddconfig: dict) -> float:
    enc = Encoder(**ddconfig)
    enc.eval()
    full = torch.randn(1, 3, 13, 64, 64)
    short = full[:, :, :9].clone()
    with torch.no_grad():
        _, act_short = enc(short, return_intermediates=True)
        _, act_full = enc(full, return_intermediates=True)
    max_diff = 0.0
    for key in act_short:
        t_ctx = act_short[key].shape[2]
        diff = (act_short[key] - act_full[key][:, :, :t_ctx]).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Optional v2 checkpoint")
    args = parser.parse_args()

    dd = dict(
        double_z=False, z_channels=32, resolution=64, in_channels=3, out_ch=3,
        ch=64, ch_mult=[1, 2, 2, 4], num_res_blocks=2,
    )
    model = build_model()
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()

    audit = audit_encoder_taps(dd, 13)

    print("=== VAE v2 validation gates ===")
    print(f"Spatial taps @ T=13: {sorted(audit.keys())}")
    assert audit["h32_w32"][2] == 13
    assert audit["h16_w16"][2] == 7
    assert audit["h8_w8"][2] == 4
    print("Shape audit: PASS")

    max_diff = check_prefix_stability(dd)
    print(f"Prefix activation max diff: {max_diff:.2e}")
    if max_diff >= 1e-5:
        print("Prefix stability: FAIL")
        return 1
    print("Prefix stability: PASS")

    x = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        rec, _ = model(x)
    mse = ((rec - x) ** 2).mean().item()
    print(f"Random-init reconstruction MSE: {mse:.4f}")
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
