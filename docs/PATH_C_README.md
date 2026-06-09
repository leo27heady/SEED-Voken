# Path C — Encoder v2 + T=13 VAE + Hierarchical Video Predictor

End-to-end stack for **causal hierarchical video prediction** on Shapes3D @ 64px.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Encoder/decoder v2 (FrameWiseGroupNorm + spatial tap keys) | Implemented |
| 2 | SQ-VAE-2 retrain @ T=13 | Config + smoke ready; **full train required** |
| 3 | Envelope hierarchical predictor | Implemented |

---

## Architecture (locked spec)

### Stage geometry @ T=13

| Stage | Spatial key | Native T | H×W | Codebook K | Shifts |
|-------|-------------|----------|-----|------------|--------|
| 0 coarse | `h8_w8` | 4 | 8×8 | 1536 | 1 |
| 1 mid | `h16_w16` | 7 | 16×16 | 768 | 2 |
| 2 fine | `h32_w32` | 13 | 32×32 | 384 | 4 |

- **T_context = 9**, **T_total = 13** (4-frame RGB horizon: frames 10–13)
- Causal temporal downsample: `kernel_t=3`, `stride_t=2` per level
- **Envelope order:** `(0,0) → (1,0) → (2,0) → (2,1) → (1,1) → (2,2) → (2,3)`

### Predictor design

```
Video (B,3,13,64,64)
    │
    ▼  frozen VAE.encode_tokens (single pass, spatial tap keys)
Context codebook embeds (shift 0 only) + GT indices for CE/MSE
    │
    ▼  EnvelopeOrchestrator DFS (7 events)
PredictorStage × 3  ── dual_stream parent cross-attn ──►  CE @ each shift
    │
    ▼  argmax horizon indices → decode_from_indices (zero activations)
Pred MSE on RGB frames 10–13  (λ=0.5)
```

**Defaults:** quantized context · `dual_stream` parent · `streams_only` re-entry · factorized attn on mid/fine · AR inference tracks rolling embed buffer.

---

## Repository layout

```
src/Open_MAGVIT2/
  modules/diffusionmodules/norm.py          # FrameWiseGroupNorm
  modules/diffusionmodules/improved_video_model.py  # spatial _shape_key, adaGN v2
  modules/vqvae/hierarchical/               # tap_keys, layer_string, shape_audit, checkpoint_v2
  modules/predictor/                        # schedule, masks, batch_prep, orchestrator, stage, …
  models/video_hier_vqgan.py                # VAE
  models/video_hier_predictor.py            # Lightning predictor

configs/Open-MAGVIT2/gpu/
  shapes3d_sqvae2_64_S_v2.yaml              # production VAE v2
  shapes3d_sqvae2_64_S_v2_smoke.yaml        # 2-epoch CPU smoke
  shapes3d_sqvae2_64_S_v2_predict.yaml      # predictor (set vae_ckpt after VAE train)

scripts/
  validate_v2_vae_gate.py                   # pre-predictor validation gates

tests/
  test_encoder_v2.py                        # EV2-* encoder tests
  predictor/                                # schedule, masks, stage, orchestrator, AR, …
```

---

## Prerequisites

```powershell
cd C:\Users\leoni\Documents\projects\SEED-Voken
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # or project install instructions
$env:PYTHONPATH = (Get-Location).Path
```

**GPU:** CUDA recommended for production training. Smoke configs run on CPU.

**Data:** Shapes3D cache at `data/shape_cache` (auto-created by `ShapeVideoDataset`).

---

## Pre-flight checklist (before training)

Run these in order after pulling the audit-fix branch:

```powershell
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

# 1. Unit tests (expect ≥95 passed)
python -m pytest tests/ -q

# 2. Encoder v2 validation gates
python scripts/validate_v2_vae_gate.py

# 3. VAE smoke (optional, ~2 min CPU)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_smoke.yaml
```

**Before predictor training:**
- [ ] VAE v2 full train completed (`val_ema/mse` acceptable)
- [ ] `python scripts/validate_v2_vae_gate.py --ckpt <vae_best.ckpt>` passes
- [ ] `vae_ckpt` set in `shapes3d_sqvae2_64_S_v2_predict.yaml`
- [ ] pytest green

**Optional legacy weight init:**
```powershell
python scripts/migrate_jq0ynj5m_to_v2.py --old_ckpt path/to/jq0ynj5m.ckpt --out checkpoints/v2_partial.ckpt
```

---

## Step 0 — Run tests

```powershell
$env:PYTHONPATH = "C:\Users\leoni\Documents\projects\SEED-Voken"
python -m pytest tests/ -q
# Expected: 109 passed
```

Targeted suites:

```powershell
python -m pytest tests/test_encoder_v2.py -q
python -m pytest tests/predictor/ -q
```

---

## Step 1 — Validate encoder v2 gates (no checkpoint)

```powershell
python scripts/validate_v2_vae_gate.py
```

Checks:
- Spatial tap shapes @ T=13: `h8_w8→T=4`, `h16_w16→T=7`, `h32_w32→T=13`
- Prefix activation stability: `encode(T=9)` ≈ `encode(T=13)[:,:,:9]` (target < 1e-5)
- Roundtrip encode/decode smoke

---

## Step 2 — VAE v2 smoke (CI / sanity)

```powershell
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_smoke.yaml
```

2 epochs, 4 train batches, CPU, no W&B.

---

## Step 3 — VAE v2 full training (required before predictor)

```powershell
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml
```

| Setting | Value |
|---------|-------|
| Monitor | `val_ema/mse` (min) |
| Checkpoints | `checkpoints/vqgan/shapes3d_sqvae2_64_S_v2/` |
| W&B project | `seed-voken-shapes3d` / run `sqvae2_64_S_v2` |
| Sequence length | 13 |
| Tap keys | spatial (`h8_w8`, `h16_w16`, `h32_w32`) |

**Gate before predictor:** `val_ema/mse` comparable to jq0ynj5m @ T=9 trend.

Re-validate with checkpoint:

```powershell
python scripts/validate_v2_vae_gate.py --ckpt checkpoints/vqgan/shapes3d_sqvae2_64_S_v2/<run>/best.ckpt
```

---

## Step 4 — Migrate old jq0ynj5m weights (optional)

The jq0ynj5m checkpoint uses **temporal tap keys** and **old GroupNorm** — not directly compatible.

Partial conv-weight load:

```python
from src.Open_MAGVIT2.modules.vqvae.hierarchical.checkpoint_v2 import load_conv_weights_partial
import torch

old = torch.load("path/to/jq0ynj5m.ckpt", map_location="cpu")["state_dict"]
new_model = ...  # VideoHierVQModel v2
merged, n_loaded, n_skipped = load_conv_weights_partial(old, new_model.state_dict())
new_model.load_state_dict(merged, strict=False)
# Norms re-init; full T=13 retrain still recommended
```

---

## Step 5 — Predictor training

1. Edit `configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_predict.yaml`:

```yaml
model:
  init_args:
    vae_ckpt: checkpoints/vqgan/shapes3d_sqvae2_64_S_v2/<your_run>/best.ckpt
```

2. Train:

```powershell
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_predict.yaml
```

| Setting | Value |
|---------|-------|
| VAE | frozen |
| Loss | `λ_ce=1.0` + `λ_pred_mse=0.5` |
| Optimizer | AdamW, lr=1e-4 |
| Attention | coarse=full rev · mid/fine=factorized |
| Inference val | parallel + autoregressive logged |

Checkpoints: `checkpoints/predictor/shapes3d_sqvae2_64_S_v2_predict/`

---

## Step 6 — Inference modes

| Mode | API | Behavior |
|------|-----|----------|
| Train / parallel | `orchestrator.forward_train` / `forward_parallel` | GT context embeds, `streams_only` |
| Autoregressive | `orchestrator.forward_autoregressive` | Rolling finest embed buffer grows; no RGB re-encode |

From Lightning:

```python
model.forward_batch(video, inference_mode="autoregressive")  # or "parallel" / "train"
```

---

## Configuration reference

### VAE v2 (`shapes3d_sqvae2_64_S_v2.yaml`)

```yaml
hierarchy:
  tap_key_format: spatial
  sequence_length: 13
  blocks_sq: "h8_w8_x1,h16_w16_x1,h32_w32_x1"
quantizer:
  size_dict: [1536, 768, 384]
  dim_dict: [32, 32, 32]
```

### Predictor (`shapes3d_sqvae2_64_S_v2_predict.yaml`)

```yaml
predictor:
  n_layers: 4
  n_heads: 8
  dim: 32
  temporal_windows: [-1, 3, 1]
  attention:
    coarse: {type: full}
    mid:    {type: factorized, t_window: 3}
    fine:   {type: factorized, t_window: 1, spatial_window: 8}
loss:
  lambda_ce: 1.0
  lambda_pred_mse: 0.5
inference:
  mode: autoregressive
  commit: argmax
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Tap key mismatch | Old temporal checkpoint on v2 model | Full v2 retrain or partial conv load |
| Prefix instability | Old GroupNorm encoder | Use v2 encoder (FrameWiseGroupNorm) |
| Predictor OOM on fine stage | Full space-time attn | Ensure `fine.type: factorized` |
| `vae_ckpt: null` | Predictor init without trained VAE | Complete Step 3 first |
| AR NaN with extended ctx | Empty cross-attn rows on horizon tokens | Fixed via `sanitize_cross_attn_mask`; dynamic self-mask on finest AR re-init |

---

## Known limitations (post audit-fix)

| Item | Status | Notes |
|------|--------|-------|
| Full parallel oracle | Partial | Shift 0 uses full GT embeds; k>0 still uses stream carry |
| Full AR exposure bias | Partial | Finest stage k>0 re-inits from rolling embed; mid/top use `streams_only` |
| Bot CE query frame | By design | All bot shifts query frame 8 (`streams_only` + fixed `query_t`) |
| Factorized fine stage | By design | Not reversible; uses `spatial_window=8` local spatial attn |
| Index prefix cross-T | Expected | Activations stable (<1e-5); quantized indices may differ |
| EfficientRevBackProp | Deferred | Rev inverse unused in backward; higher memory on coarse stage |
| Production VAE ckpt | **Pending** | Required before predictor training (`vae_ckpt: null`) |

See **`docs/PATH_C_AUDIT.md`** for the original audit; many items above were fixed in the audit-fix pass.

---

## Quick command cheat sheet

```powershell
# Activate
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

# Test
python -m pytest tests/ -q

# Validate gates
python scripts/validate_v2_vae_gate.py

# VAE smoke → full
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_smoke.yaml
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml

# Predictor (after setting vae_ckpt)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_predict.yaml
```
