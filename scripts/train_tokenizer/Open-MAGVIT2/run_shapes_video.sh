#! /bin/bash
export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10055}
export NODE_RANK=${3:-0}
export OMP_NUM_THREADS=6

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

echo $MASTER_ADDR
echo $MASTER_PORT

# Path C — hierarchical SQ-VAE-2 @ 64px (production)
# NODE_RANK=$NODE_RANK python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml

# Path C — 32px lite dev tokenizer
# NODE_RANK=$NODE_RANK python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml

# Path C — 4-stage v4 @ T=17
NODE_RANK=$NODE_RANK python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v4.yaml
