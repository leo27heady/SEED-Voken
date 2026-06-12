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

---

## Path A (2026-06): corrected ELBO + temporal pyramid + finest-state decoding

### ELBO reductions (gaussian_sq.py)

The categorical KL now follows SQ-VAE/HQ-VAE semantics: an expectation over K
(sum) accumulated over all latent positions (sum over T,H,W), batch-averaged.
The continuous term (flg_loss_continuous layers) is likewise summed over
(C,T,H,W). Both are computed in fp32 under AMP. Previous behavior used means,
shrinking the KL by T*H*W*K relative to the ARELBO distortion and disabling
variance self-annealing (root cause of the observed codebook collapse —
see DEEP_REVIEW_2026-06-11.md section 1.1).

New `loss_cfg` knobs on `VideoHierVQModel`:

```yaml
loss_cfg:
  kl_beta: 1.0          # global KL scale (folds into kl_weights)
  kl_warmup_steps: 0    # linear beta warm-up; escape hatch if recon stalls
  grad_clip: 1.0        # manual optimization ignores trainer gradient_clip_val
```

### Temporal pyramid (u2) and finest-state decoding

```yaml
hierarchy:
  token_grid: pyramid
  blocks_sq: "h4_w4_x1,h8_w8_u2,h16_w16_u2"
  temporal_up: [2, 2]            # one entry per u2 layer; 2 => T -> 2T-1
  decoder_source: finest_state   # decoder consumes finest z_state (default: latent)
```

- `temporal_up: 2` makes the u2 Upsampler use block size (2,2,2); the
  frame-drop rule yields T -> 2T-1, exactly matching the causal encoder chain
  (3 -> 5 -> 9 for the 32px lite setup). This keeps every level's tokens on the
  native tap grids, so the predictor's 1/2/4 shift hierarchy is unchanged.
  `validate_state_chain` (shape_audit.py) fails fast on any mismatch.
- `decoder_source: finest_state` returns the post-final-injection z_state
  (e.g. (B,16,9,16,16)) from forward/decode_from_indices instead of the
  down-fused bottleneck latent — removing the 3x4x4 information bottleneck.
  Progressive partials are aligned to the finest grid so one decoder renders
  every stage row.
- `dec_ddconfig` (model-level) shapes the decoder for the finest grid, e.g.
  one spatial doubling and no temporal upsampling:

```yaml
dec_ddconfig:
  z_channels: 16
  ch: 64
  ch_mult: [1, 2]      # ch_mult[0]==1 => single (1,2,2) Upsampler
  num_res_blocks: 2
  num_groups: 8
  out_ch: 3
  in_channels: 3
  resolution: 32
  double_z: false
```

Defaults (`decoder_source: latent`, no `temporal_up`, no `dec_ddconfig`)
preserve the legacy behavior bit-for-bit; old checkpoints load unchanged.
Tests: tests/test_elbo_scaling.py, tests/test_pyramid_topdown.py.
