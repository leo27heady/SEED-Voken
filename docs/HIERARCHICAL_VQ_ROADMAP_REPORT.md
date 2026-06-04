# Hierarchical SQ-VAE-2 roadmap — implementation report

This document describes the Tier A–D extensions added on top of the baseline 2-level SQ-VAE-2 video tokenizer, how to configure them, and how to validate behavior.

## Plan compliance checklist

| Stage | Planned | Status | Notes |
|-------|---------|--------|-------|
| A | 3-level @ 128, 4-level @ 64, per-layer `size_dict` | Done | 64px taps use audited keys at `sequence_length: 13` (see mapping table below) |
| A | Codebooks coarse→fine | Done | e.g. `[1024, 512, 256, 128]` |
| B | `loss_cfg.kl_weights` + raw/weighted logs | Done | `hier_elbo_loss.py` |
| B | Usage regularizer (entropy / active-code) | Partial | Perplexity-target MSE only; no separate active-code target |
| C | Learned prior chain | Done | `prior.mode: learned_chain`, `GaussianPriorHead` |
| C | zero / uniform fallback | Done | `prior: zero` \| `uniform` |
| D | Perceptual + GAN via `loss_cfg` | Done | `HierVideoReconLoss`; 3D PatchGAN (not frame-wise disc) |
| D | `hq_elbo` baseline preserved | Done | `codebook_weight: 0` default |
| Viz | Progressive rows only if enabled | Done | `progressive_coding` |
| Verify | Unit tests + smoke configs | Done | `shapes3d_sqvae2_64_S_smoke*.yaml` |

## Summary of changes

| Feature | Status | Config namespace |
|--------|--------|------------------|
| 3-level hierarchy @ 128 | Done | `hierarchy.blocks_sq`, `quantizer.size_dict` |
| 4-level hierarchy @ 64 (T=12) | Done | `shapes3d_sqvae2_64_S.yaml` |
| Per-layer KL weights | Done | `loss_cfg.kl_weights` |
| Per-layer usage regularizer (perplexity target) | Done | `quantizer.usage_reg_weight`, `usage_reg_target_perplexity` |
| Learned prior chain (HQ-style) | Done | `quantizer.prior.mode: learned_chain` |
| Optional perceptual loss | Done | `loss_cfg.perceptual` |
| Optional GAN (3D PatchGAN) | Done | `loss_cfg.gan` |
| Progressive viz only when enabled | Done | `progressive_coding: true` |

## Config reference

### Hierarchy depth (`hierarchy`)

`blocks_sq` is a comma-separated DSL parsed by `layer_string.py`:

- `t{T}_h{H}_w{W}_xN` — N residual SQ layers at that encoder tap (native grid).
- `t{T}_h{H}_w{W}_u2` — pyramid inject + upsample (requires `token_grid: pyramid`).

**Example — 3 levels @ 128:**

```yaml
hierarchy:
  mode: sqvae2
  token_grid: native
  latent_key: t3_h16_w16
  blocks_sq: "t3_h16_w16_x1,t5_h32_w32_x1,t9_h64_w64_x1"
  tap_channels:
    t3_h16_w16: 32
    t5_h32_w32: 64
    t9_h64_w64: 128
```

**Example — 4 levels @ 64, sequence_length 13** (`T % 4 == 1`). Tap keys are **literal** `(T,H,W)` from the encoder, not nominal grid sizes:

| Roadmap label (intent) | Actual key @ 64px, T=13 |
|------------------------|-------------------------|
| `t3_h4_w4` (coarse)    | `t4_h8_w8` (no 4×4 tap) |
| `t5_h8_w8`             | `t7_h16_w16`            |
| `t9_h16_w16`           | `t7_h16_w16`            |
| `t13_h32_w32`          | `t13_h32_w32`           |
| (finest spatial)       | `t13_h64_w64` (4th level) |

```yaml
hierarchy:
  sequence_length: 13   # fail-fast tap check in VideoHierVQModel
  latent_key: t4_h8_w8
  blocks_sq: "t4_h8_w8_x1,t7_h16_w16_x1,t13_h32_w32_x1,t13_h64_w64_x1"
  tap_channels:
    t4_h8_w8: 32
    t7_h16_w16: 128
    t13_h32_w32: 128
    t13_h64_w64: 64
data:
  init_args:
    train:
      params:
        config:
          sequence_length: 13
```

Run `audit_encoder_taps(ddconfig, sequence_length)` (see `shape_audit.py`) to verify tap keys exist for your resolution and T.

### Codebook sizing (`quantizer`)

Use **larger `size_dict` on coarse layers, smaller on fine**:

```yaml
quantizer:
  size_dict: [1024, 512, 256]      # 3-level
  dim_dict: [32, 32, 32]           # must match ddconfig.z_channels (native ResSQ quantizes z_channels-d tensors)
  log_param_q_init: [4.09434, 4.09434, 4.09434]
```

Health metrics: `train/perplexity_layer_i` should stay **well below** `size_dict[i]` (not pinned at K).

### Prior (`quantizer.prior`)

| Mode | Behavior |
|------|----------|
| `zero` (default) | `log p_prior = 0` |
| `uniform` | Uniform over K codes |
| `learned_chain` | Small conv prior head → `z_pri` → Gaussian distance prior logits |

```yaml
quantizer:
  prior:
    mode: learned_chain
    width: 32
    detach_conditioning: false
```

### Usage regularizer (optional)

Penalizes deviation of batch+space perplexity from a target (per layer if lists are used):

```yaml
quantizer:
  usage_reg_weight: [0.0, 0.0, 0.01]              # per layer
  usage_reg_target_perplexity: [512, 256, 64]      # encourage coarse spread, tight fine
```

Scalar values broadcast to all layers.

### Loss extensions (`loss_cfg`)

```yaml
model:
  init_args:
    loss_cfg:
      kl_weights: [0.5, 0.75, 1.0]    # optional; length = num_layers
      codebook_weight: 0.0             # keep 0 when using hq_elbo (KL already in base loss)
      perceptual:
        enabled: true
        weight: 0.1
      gan:
        enabled: true
        weight: 0.8
        disc_start: 5000
        disc_factor: 1.0
        disc_loss: hinge
```

- **HQ-ELBO** (`training_objective: hq_elbo`) always provides ARELBO distortion + layer KL.
- **Perceptual** adds frame-wise LPIPS on reconstructions.
- **GAN** adds a second optimizer (discriminator); training follows the `video_lfqgan` pattern.

### Progressive training / logging

```yaml
progressive_coding: false   # recommended until L2 perplexity is healthy
```

When `false`, WandB grids show **ground_truth + reconstruction** only. When `true`, progressive rows are logged and ARELBO uses averaged distortion over partial decodes.

## Config files

| File | Purpose |
|------|---------|
| `configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_128_S.yaml` | 3-level @ 128 (production S) |
| `configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S.yaml` | 4-level @ 64, T=13 |
| `configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_smoke.yaml` | Fast smoke (3 epochs, 16 samples) |
| `configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_smoke_learned_prior.yaml` | Smoke + `prior.mode: learned_chain` |
| `configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_smoke_loss_extras.yaml` | Smoke + KL weights + perceptual + GAN |

## Verification

### Unit tests

```bash
PYTHONPATH=. python tests/test_hierarchical_vq.py
```

Covers: KL mean invariance, KL weights, learned prior, 3-level forward, usage reg, progressive loss averaging.

### Smoke training

```bash
PYTHONPATH=. python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_smoke.yaml
PYTHONPATH=. python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_smoke_learned_prior.yaml
PYTHONPATH=. python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_smoke_loss_extras.yaml
```

Before editing `blocks_sq` or `sequence_length`, run a tap audit (example for 64 S):

```python
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import audit_encoder_taps, format_tap_audit
ddconfig = {...}  # match model.init_args.ddconfig
print(format_tap_audit(audit_encoder_taps(ddconfig, sequence_length=9)))
```

### Full S debug (128, 3-level)

```bash
PYTHONPATH=. python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_128_S.yaml --trainer.max_epochs 30
```

Watch:

- `loss/kl_layer_*` similar magnitude across layers
- `train/perplexity_layer_*` not pinned at `size_dict[i]`
- `val_ema/mse` decreasing

## Recommended rollout

1. Train **128 / 3-level** baseline (no prior, no GAN, no progressive).
2. Enable **`loss_cfg.kl_weights`** + usage targets if L2 usage is too uniform.
3. Try **`prior.mode: learned_chain`** on S only.
4. Add **perceptual** (small weight), then **GAN** (late `disc_start`) if recon is blurry.
5. Enable **`progressive_coding`** only after L2 perplexity clearly improves.

## Code map

- `gaussian_sq.py` — SQ quantizer, prior modes, usage reg
- `prior_net.py` — `GaussianPriorHead`
- `video_inj_topdown.py` — multi-level top-down + prior heads
- `hier_elbo_loss.py` — ARELBO + KL weights
- `hier_video_loss.py` — optional perceptual + GAN
- `video_hier_vqgan.py` — Lightning module wiring

---

Config Cheat sheet:

```json
model:
  init_args:
    hierarchy:
      blocks_sq: "t3_h8_w8_x1,t5_h16_w16_x1,t9_h32_w32_x1,t9_h64_w64_x1"  # 4-level @ 64, T=9
      tap_channels: { t3_h8_w8: 32, t5_h16_w16: 128, t9_h32_w32: 128, t9_h64_w64: 64 }
    quantizer:
      size_dict: [1024, 512, 256, 128]   # larger on coarse layers
      dim_dict: [32, 32, 32, 32]         # match z_channels
      prior: zero | uniform | { mode: learned_chain, width: 32 }
      usage_reg_weight: 0.0
      usage_reg_target_perplexity: 0.0
    loss_cfg:
      kl_weights: [0.5, 0.75, 1.0, 1.0]
      codebook_weight: 0.0              # keep 0 for hq_elbo
      perceptual: { enabled: true, weight: 0.1 }
      gan: { enabled: true, weight: 0.5, disc_start: 5000 }
    training_objective: hq_elbo
    progressive_coding: false
```
