"""Validation gates for SQ-VAE-2 VAE configs before predictor training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Encoder
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import audit_encoder_taps

V2_DEFAULTS = dict(
    ddconfig=dict(
        double_z=False,
        z_channels=32,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
    ),
    hierarchy=dict(
        mode="sqvae2",
        token_grid="native",
        tap_key_format="spatial",
        sequence_length=13,
        latent_key="h8_w8",
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h8_w8=32, h16_w16=128, h32_w32=128),
    ),
    quantizer=dict(
        type="sq",
        prior="zero",
        size_dict=[1536, 768, 384],
        dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3,
        temperature=dict(init=1.0, decay=1e-5, min=0.3),
    ),
    expected_taps=dict(h32_w32=13, h16_w16=7, h8_w8=4),
    t_context_short=9,
)

V4_DEFAULTS = dict(
    ddconfig=dict(
        double_z=False,
        z_channels=32,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 2, 2],
        num_res_blocks=2,
        num_groups=32,
    ),
    hierarchy=dict(
        mode="sqvae2",
        token_grid="native",
        tap_key_format="spatial",
        sequence_length=17,
        latent_key="h4_w4",
        blocks_sq="h4_w4_x1,h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h4_w4=32, h8_w8=128, h16_w16=128, h32_w32=128),
    ),
    quantizer=dict(
        type="sq",
        prior="zero",
        size_dict=[2048, 1024, 512, 512],
        dim_dict=[32, 32, 32, 32],
        log_param_q_init=[4.09434] * 4,
        temperature=dict(init=1.0, decay=1e-5, min=0.3),
    ),
    expected_taps=dict(h32_w32=17, h16_w16=9, h8_w8=5, h4_w4=3),
    t_context_short=9,
)


def load_profile(name: str) -> dict:
    if name == "v2":
        return V2_DEFAULTS
    if name == "v4":
        return V4_DEFAULTS
    raise ValueError(f"Unknown profile {name!r}; use v2 or v4")


def load_from_yaml(path: str) -> dict:
    from src.Open_MAGVIT2.modules.vqvae.hierarchical.layer_string import parse_blocks_sq
    from src.Open_MAGVIT2.modules.vqvae.hierarchical.tap_keys import normalize_resolution_key

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    init = cfg["model"]["init_args"]
    seq = init["hierarchy"]["sequence_length"]
    block_keys = [
        normalize_resolution_key(spec.resolution_key)
        for spec in parse_blocks_sq(init["hierarchy"]["blocks_sq"])
    ]
    return dict(
        ddconfig=init["ddconfig"],
        hierarchy=init["hierarchy"],
        quantizer=init["quantizer"],
        block_keys=block_keys,
        sequence_length=seq,
        t_context_short=min(9, seq - 1),
    )


def build_model(profile: dict) -> VideoHierVQModel:
    return VideoHierVQModel(
        ddconfig=profile["ddconfig"],
        hierarchy=profile["hierarchy"],
        quantizer=profile["quantizer"],
        learning_rate=1e-4,
    )


def check_prefix_stability(ddconfig: dict, t_total: int, t_short: int) -> float:
    enc = Encoder(**ddconfig)
    enc.eval()
    res = ddconfig.get("resolution", 64)
    full = torch.randn(1, 3, t_total, res, res)
    short = full[:, :, :t_short].clone()
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
    parser.add_argument("--ckpt", type=str, default=None, help="Optional VAE checkpoint")
    parser.add_argument(
        "--profile",
        type=str,
        default="v2",
        choices=["v2", "v4"],
        help="Built-in validation profile (default: v2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional yaml config; overrides --profile model init_args",
    )
    args = parser.parse_args()

    if args.config:
        profile = load_from_yaml(args.config)
        t_total = profile["sequence_length"]
        block_keys = profile["block_keys"]
        expected = None
        label = Path(args.config).stem
    else:
        profile = load_profile(args.profile)
        t_total = profile["hierarchy"]["sequence_length"]
        block_keys = list(profile["expected_taps"].keys())
        expected = profile["expected_taps"]
        label = args.profile

    dd = profile["ddconfig"]
    model = build_model(profile)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()

    audit = audit_encoder_taps(dd, t_total)

    print(f"=== VAE validation gates ({label}) ===")
    print(f"Spatial taps @ T={t_total}: {sorted(audit.keys())}")
    for key in block_keys:
        assert key in audit, f"Missing tap {key!r}; available {sorted(audit.keys())}"
    if expected is not None:
        for key, t_native in expected.items():
            assert audit[key][2] == t_native, f"{key}: expected T={t_native}, got {audit[key][2]}"
    print("Shape audit: PASS")

    t_short = profile.get("t_context_short", 9)
    max_diff = check_prefix_stability(dd, t_total, t_short)
    print(f"Prefix activation max diff (T={t_short} vs T={t_total}): {max_diff:.2e}")
    if max_diff >= 1e-5:
        print("Prefix stability: FAIL")
        return 1
    print("Prefix stability: PASS")

    res = dd.get("resolution", 64)
    x = torch.randn(1, 3, t_total, res, res)
    with torch.no_grad():
        rec, _ = model(x)
    mse = ((rec - x) ** 2).mean().item()
    print(f"Random-init reconstruction MSE: {mse:.4f}")
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
