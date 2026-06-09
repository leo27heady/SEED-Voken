"""Migrate jq0ynj5m (temporal keys) checkpoint conv weights to VAE v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.vqvae.hierarchical.checkpoint_v2 import load_conv_weights_partial


def main() -> int:
    parser = argparse.ArgumentParser(description="Partial jq0ynj5m → v2 weight migration")
    parser.add_argument("--old_ckpt", required=True, help="Legacy checkpoint path")
    parser.add_argument("--out", required=True, help="Output merged checkpoint path")
    parser.add_argument(
        "--vae_config",
        default="configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml",
        help="Target v2 VAE config",
    )
    args = parser.parse_args()

    with open(ROOT / args.vae_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]["init_args"]
    model = VideoHierVQModel(**model_cfg)

    old = torch.load(args.old_ckpt, map_location="cpu", weights_only=False)
    old_state = old.get("state_dict", old)
    merged, n_loaded, n_skipped = load_conv_weights_partial(old_state, model.state_dict())
    model.load_state_dict(merged, strict=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, out_path)
    print(f"Saved {out_path}")
    print(f"Loaded {n_loaded} tensors, skipped {n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
