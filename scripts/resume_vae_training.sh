#!/usr/bin/env bash
# Resume SQ-VAE-2 v2 training from a checkpoint (64px Shapes3D, T=13).
# Usage (from repo root):
#   source venv/bin/activate
#   export PYTHONPATH=.
#   ./scripts/resume_vae_training.sh /path/to/epoch=67-step=56712.ckpt

set -euo pipefail
CONFIG="${CONFIG:-configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml}"
CKPT="${1:?Usage: $0 <checkpoint.ckpt>}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

python main.py fit --config "$CONFIG" --ckpt_path "$CKPT"
