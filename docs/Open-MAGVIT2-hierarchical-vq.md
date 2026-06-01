# Hierarchical video VQ (HQ-VAE × Open-MAGVIT2)

This guide describes the **causal 3D video tokenizer** built on the Open-MAGVIT2 encoder/decoder with **multi-level quantization** inspired by [HQ-VAE](https://github.com/kakaobrain/hq-vae). It targets synthetic shape video (`ShapeVideoDataset`) and is the primary path for SEED-Voken shape experiments.

## Overview

| Mode | Config example | Quantization |
|------|----------------|--------------|
| **SQ-VAE-2** (recommended) | `shapes3d_sqvae2_128_L.yaml` | Gaussian stochastic quantizer (SQ) at multiple encoder taps |
| **RSQ-VAE** | `shapes3d_rsqvae_128_L.yaml` | Residual SQ stack on the encoder bottleneck |
| **VQ / LFQ variants** | `shapes3d_sqvae2_vq_128_L.yaml`, `shapes3d_sqvae2_lfq_128_L.yaml` | Deterministic VQ or LFQ per layer |
| **Pyramid SQ-VAE-2** | `shapes3d_sqvae2_pyramid_128_L.yaml` | `u2` upsampling blocks between inject layers |
| **Flat LFQ baseline** | `shapes3d_lfqgan_128_L.yaml` | Single-level MAGVIT LFQ (no hierarchy) |

**Training objective (v1):** `hq_elbo` — ARELBO reconstruction distortion plus per-layer auxiliary terms (KL for SQ, commitment for VQ, entropy/sample-batch terms for LFQ). GAN training (`use_gan: true`) is not wired.

The name `hq_elbo` is historical: for non-SQ quantizers the distortion term is still ARELBO-style MSE, but layer auxiliaries are not Gaussian KL.

## Architecture

```
Video (B,C,T,H,W)
    → MAGVIT Encoder → tap dict {t*_h*_w*: features} + bottleneck h
    → Top-down quantizer (SQVAE2TopDown or RSQTopDown)
    → fused z_q on h’s (T,H,W) grid
    → MAGVIT Decoder → reconstruction
```

### SQ-VAE-2: native vs pyramid

- **`token_grid: native`** — Each layer quantizes at its tap’s native `(T,H,W)`. Residuals are `activation − aligned(z_latent)` and contributions are fused onto the **decoder latent grid** (shape of `h`, not necessarily `latent_key`).
- **`token_grid: pyramid`** — `u2` layers use `InjSQBlock` (upsample `z_state` then inject). `xN` layers use residual SQ on the current pyramid grid.

### `latent_key` vs encoder bottleneck `h`

- **`latent_key`** — Reference tap for initializing the fused latent tensor; used when `encoder_bottleneck` is not passed.
- **`encoder_bottleneck` (`h`)** — Actual MAGVIT bottleneck returned by the encoder; **defines the spatial/temporal grid** onto which all layer contributions are fused. Always pass `h` in training and inference for correct shapes.

### Progressive coding (optional)

When `progressive_coding: true` on `VideoHierVQModel`:

1. Standard forward still produces the final reconstruction.
2. `forward_progressive` builds cumulative partial latents per layer.
3. Each partial latent is decoded; **ARELBO distortions are summed** across those partial reconstructions (HQ-style progressive training).
4. Layer KL/aux terms are unchanged.

Default is `false` — production configs behave as before.

## Configuration

Production SQ-VAE-2 ([`configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_128_L.yaml`](../configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_128_L.yaml)):

```yaml
hierarchy:
  mode: sqvae2
  token_grid: native
  latent_key: t3_h16_w16
  blocks_sq: "t3_h16_w16_x1,t9_h64_w64_x1"
  tap_channels:
    t3_h16_w16: 64
    t9_h64_w64: 256
quantizer:
  type: sq
  prior: zero          # zero | uniform (SQ prior only)
  size_dict: [512, 512]
  dim_dict: [64, 64]
```

### `blocks_sq` grammar

Comma-separated tokens parsed by [`layer_string.py`](../src/Open_MAGVIT2/modules/vqvae/hierarchical/layer_string.py):

| Token | Meaning |
|-------|---------|
| `t{T}_h{H}_w{W}_x{N}` | `N` residual SQ layers at tap `t{T}_h{H}_w{W}` (no upsample) |
| `t{T}_h{H}_w{W}_u2` | One inject layer: upsample `z_state` ×2 (spatial), then SQ inject |

Example: `"t3_h16_w16_x1,t9_h64_w64_x1"` → two native levels with indices shaped like `(3,16,16)` and `(9,64,64)` at 128px, `T=9`.

**Important:** Tap keys depend on `sequence_length`, `resolution`, and `num_res_blocks`. Re-audit when any of these change.

### Opt-in progressive ARELBO

```yaml
model:
  init_args:
    progressive_coding: true
    # progressive_noise_weight: 0.0  # optional HQ-style log(mse + noise)
```

## Tap audit

Before editing `blocks_sq` or `tap_channels`, run a dry encoder forward:

```python
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import (
    audit_encoder_taps,
    format_tap_audit,
)

ddconfig = dict(
    ch=128, in_channels=3, num_res_blocks=4, z_channels=64, out_ch=3,
    resolution=128, ch_mult=[1, 2, 2, 4],
)
audit = audit_encoder_taps(ddconfig, sequence_length=9, resolution=128)
print(format_tap_audit(audit))
```

Example keys (128px, `num_res_blocks=4`, `T=9`): `t3_h16_w16`, `t5_h32_w32`, `t9_h64_w64`, `t9_h128_w128`.

For `T=5` at 128px (dev config): `t2_h16_w16`, `t5_h64_w64` — not the same keys as `T=9`.

Constraint: **`sequence_length` must satisfy `T % 4 == 1`** (causal temporal downsampling).

## Environment and training

```bash
# From repo root
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
# Shape data deps (if using setup_shapekit.sh)
bash setup_shapekit.sh
```

### Smoke test (1 batch)

```bash
set PYTHONPATH=.
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_128_L_dev.yaml
```

### Full training

```bash
set PYTHONPATH=.
bash scripts/train_tokenizer/Open-MAGVIT2/run_shapes_video.sh
# Or directly:
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_128_L.yaml
```

WandB logging and PNG reconstruction grids are configured in the YAML callbacks (`VideoReconstructionLogger`).

### Progressive training smoke

Add to dev or production yaml under `model.init_args`:

```yaml
progressive_coding: true
```

## Inference API

```python
import torch
from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
# ... load model from checkpoint or instantiate from config ...

x = batch["video"]  # (B, C, T, H, W)

# Discrete tokens per level
tok = model.encode_tokens(x, flg_quant_det=True)
for level in tok["levels"]:
    print(level["key"], level["shape"], level["indices"].shape, level["codebook_size"])

z_q = tok["z_q"]
x_rec = model.decode(z_q)

# Reconstruct from indices alone (SQ-VAE-2)
indices = [lev["indices"] for lev in tok["levels"]]
x_from_idx = model.decode_from_indices(
    indices,
    activations=tok["activations"],
    encoder_bottleneck=tok["encoder_bottleneck"],
)

# RSQ: activations not required
# tok = model.encode_tokens(x)  # mode rsqvae
# x_from_idx = model.decode_from_indices([lev["indices"] for lev in tok["levels"]])
```

Progressive visualizations (WandB / `log_images`):

```python
log = model.log_images({"video": x})
# Keys: inputs, reconstructions, progressive_L1, progressive_L1-L2, ...
```

## Config matrix

| File | Purpose |
|------|---------|
| `shapes3d_sqvae2_128_L.yaml` | Production native SQ-VAE-2, T=9, 128px (~371M params) |
| `shapes3d_sqvae2_128_M.yaml` | Medium width/depth, same hierarchy (~168M) |
| `shapes3d_sqvae2_128_S.yaml` | Small width/depth, same hierarchy (~57M) |
| `shapes3d_sqvae2_128_L_dev.yaml` | `fast_dev_run` smoke, T=5 |
| `shapes3d_sqvae2_pyramid_128_L.yaml` | Pyramid + `u2` example |
| `shapes3d_rsqvae_128_L.yaml` | RSQ on bottleneck |
| `shapes3d_sqvae2_vq_128_L.yaml` | VQ quantizer per layer |
| `shapes3d_sqvae2_lfq_128_L.yaml` | LFQ with channel projections |
| `shapes3d_lfqgan_128_L.yaml` | Flat single-level LFQ baseline |

## Tests

```bash
set PYTHONPATH=.
python tests/test_hierarchical_vq.py
```

Covers Gaussian SQ, native multi-level shapes, pyramid progressive consistency, progressive ARELBO backward, RSQ `encode_tokens`, VQ-only `log_param_q` buffer, and model `log_images`.

## Package layout

```
src/Open_MAGVIT2/modules/vqvae/hierarchical/
  base.py              # QuantizerResult
  gaussian_sq.py       # SQ quantizer
  deterministic_vq.py  # VQ
  lfq_adapter.py       # LFQ + Conv3D proj
  quantizer_builder.py # per-layer factory
  video_inj_topdown.py # SQ-VAE-2 top-down
  video_rsq_topdown.py # RSQ top-down
  hier_elbo_loss.py    # ARELBO + layer aux
  shape_audit.py       # encoder tap audit
  factory.py           # build_top_down()
src/Open_MAGVIT2/models/video_hier_vqgan.py  # Lightning module
```

## Known limitations

- No HQ internal pixel decoder; MAGVIT decoder only.
- No hierarchical prior chain (`z_pri`); SQ prior is `zero` or `uniform` only.
- GAN / `vqgan` training path raises `NotImplementedError`.
- `encode_tokens` / `decode_from_indices` for **sqvae2** and **rsqvae**; not all quantizer combos expose full index decode paths for LFQ experiments.
- Progressive ARELBO adds extra decoder passes per step when enabled (memory/latency).

## References

- Open-MAGVIT2: [docs/Open-MAGVIT2.md](./Open-MAGVIT2.md)
- HQ-VAE (conceptual basis for multi-level SQ and ARELBO)
