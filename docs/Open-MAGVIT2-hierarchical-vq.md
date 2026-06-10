# Hierarchical video VQ (Path C — SQ-VAE-2)

Causal 3D video tokenizer for **Shapes3D** using Open-MAGVIT2 encoder/decoder v2 with **multi-level Gaussian SQ** quantization (SQ-VAE-2 / HQ-VAE style).

> **Scope:** This repo keeps **sqvae2-only** configs. RSQ, LFQ, flat VQ, pyramid ablations, and IBQ stacks were removed. See [PATH_C_README.md](./PATH_C_README.md) for the full training workflow.

## Overview

| Profile | Config | T | Stages | Resolution |
|---------|--------|---|--------|------------|
| **v2 production** | `shapes3d_sqvae2_64_S_v2.yaml` | 13 | 3 | 64px |
| **v2 smoke** | `shapes3d_sqvae2_64_S_v2_smoke.yaml` | 13 | 3 | 64px |
| **32px lite** | `shapes3d_sqvae2_32_S_lite.yaml` | 13 | 3 | 32px |
| **v4** | `shapes3d_sqvae2_64_S_v4.yaml` | 17 | 4 | 64px |

**Training objective:** `hq_elbo` — ARELBO reconstruction distortion plus per-layer Gaussian KL. Perceptual/GAN paths are not wired (discriminator removed).

## Architecture

```
Video (B,C,T,H,W)
    → MAGVIT Encoder v2 → spatial tap dict {h*_w*: features} + bottleneck h
    → SQVAE2TopDown (native token grid per tap)
    → fused z_q on decoder latent grid
    → MAGVIT Decoder v2 → reconstruction
```

### Encoder v2 highlights

- **FrameWiseGroupNorm** — prefix-stable activations across T.
- **Spatial tap keys** — `h8_w8`, `h16_w16`, `h32_w32` (no T in key).
- Run `audit_encoder_taps(ddconfig, sequence_length)` before editing `blocks_sq`.

### v2 stage geometry @ T=13

| Stage | Tap | Native T | Codebook K |
|-------|-----|----------|------------|
| 0 | h8_w8 | 4 | 1536 |
| 1 | h16_w16 | 7 | 768 |
| 2 | h32_w32 | 13 | 384 |

### v4 stage geometry @ T=17

Requires `ch_mult: [1, 2, 2, 2, 2]` and taps `h4_w4,h8_w8,h16_w16,h32_w32`. Incompatible with v2 checkpoints.

## Configuration example (v2)

```yaml
hierarchy:
  mode: sqvae2
  tap_key_format: spatial
  sequence_length: 13
  blocks_sq: "h8_w8_x1,h16_w16_x1,h32_w32_x1"
quantizer:
  type: sq
  prior: zero
  size_dict: [1536, 768, 384]
  dim_dict: [32, 32, 32]
```

### `blocks_sq` grammar

| Token | Meaning |
|-------|---------|
| `h{H}_w{W}_x{N}` | N residual SQ layers at tap `h{H}_w{W}` |

Legacy temporal keys (`t3_h8_w8_x1`) are accepted with a deprecation warning.

## Metrics

| Log key | Meaning |
|---------|---------|
| `loss/mse` | Sum of squared errors / batch (large magnitude) |
| `loss/mse_per_pixel` | `mse / (C·T·H·W)` — use for dashboards |
| `perplexity_frac_layer_*` | Codebook usage fraction per stage |

## Training

```powershell
$env:PYTHONPATH = (Get-Location).Path

# Smoke
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_smoke.yaml

# Full / resume
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml
.\scripts\resume_vae_training.ps1 -CkptPath "checkpoints/.../best.ckpt"

# Pre-predictor gate
python scripts/validate_v2_vae_gate.py --ckpt checkpoints/.../best.ckpt
```

## Code map

```
src/Open_MAGVIT2/
  modules/diffusionmodules/norm.py              # FrameWiseGroupNorm
  modules/diffusionmodules/improved_video_model.py
  modules/vqvae/hierarchical/
    factory.py                                  # sqvae2 only
    quantizer_builder.py                        # Gaussian SQ only
    video_inj_topdown.py                        # SQVAE2TopDown
    gaussian_sq.py
    hier_elbo_loss.py
  models/video_hier_vqgan.py                    # Lightning VAE
  models/video_hier_predictor.py                  # Lightning predictor
```

## Predictor (frozen VAE)

After VAE training, set `vae_ckpt` in `shapes3d_sqvae2_64_S_v2_predict.yaml` and train the envelope predictor. See [PATH_C_README.md](./PATH_C_README.md).

## Tests

```powershell
python -m pytest tests/test_encoder_v2.py tests/test_hierarchical_vq.py -q
python -m pytest tests/predictor/ -q
```
