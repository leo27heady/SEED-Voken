# Path C Architecture Audit Report

**Date:** 2026-06-10 (post refactor)  
**Scope:** Path C stack — encoder v2, SQ-VAE-2 @ T=13/T=17, hierarchical predictor  
**Test status:** 119+ passing (`python -m pytest tests/ -q`)  
**Validation gate:** prefix stability < 1e-5, shape audit PASS

---

## Executive summary

| Area | Verdict | Notes |
|------|---------|-------|
| Encoder/decoder v2 | **Correct** | FrameWiseGroupNorm, spatial keys, prefix-stable |
| VAE configs & gates | **Operational** | epoch-67 ckpt wired; resume scripts available |
| PyramidSchedule math | **Correct** | `from_encoder_audit()` + hardcoded v2/v4 profiles |
| Envelope orchestrator | **Correct** | DFS order, `forward_parallel(parallel_mode)`, CE breakdown |
| Masks | **Correct** | Shift-aware RF via `rf_frames_for_shift` (MSK-06 golden) |
| Reversible blocks | **Correct** | REV-01 invertibility, REV-02 gradcheck; RevBackProp opt-in |
| Predictor stages | **Correct** | Zero-init output head, dual_stream parent, factorized + spatial_window |
| Inference (AR) | **Partial** | Finest rolling_ctx re-init; mid/top use `streams_only` carry |
| Lightning / training | **Correct** | grad clip 1.0, expanded val logging, OOM guard on parallel |
| Repo scope | **Path C only** | IBQ, flat LFQ/AR, ablation configs removed |

**Bottom line:** Training path is sound and observable. Remaining gaps are **partial AR exposure bias** (by `streams_only` design) and optional **VAE recon quality** (resume training to plateau).

---

## Fixes applied (2026-06 refactor)

| Item | Status |
|------|--------|
| `loss/mse_per_pixel` logging | Done — compare to per-pixel intuition |
| `parallel_mode: full \| context` | Done — full = oracle shift-0 embeds; context = train-like |
| Zero-init `output_head` + PRED-CE-01 | Done — CE starts near `log(K)` |
| Remove `fused_soft` parent mode | Done — `dual_stream` (default) or `fused_hard` only |
| EfficientRevBackProp | Done — opt-in `use_reversible_backprop: true` on coarse stage |
| NanGuardCallback | On v2, v4, 32_lite VAE configs |
| Validation logging | `val/pred_mse_*`, per-shift CE, parallel OOM fallback |
| Factory / quantizer | sqvae2 + Gaussian SQ only |
| Tests | REV-02, PRED-GPU-01, PAR-MEM-01, PRED-CE-01 added |

---

## Part 1 — Encoder/Decoder v2

**Status: CORRECT**

- FrameWiseGroupNorm + spatial tap keys (`h8_w8`, …).
- Prefix activation diff < 1e-5 (EV2-03 cross-T index prefix documented as non-stable for quantized indices).
- `checkpoint_v2.load_conv_weights_partial()` + `scripts/migrate_jq0ynj5m_to_v2.py`.

---

## Part 2 — VAE @ T=13 / T=17

**Status: CONFIG + METRICS CORRECT; TRAINING RESUMABLE**

- Production: `shapes3d_sqvae2_64_S_v2.yaml` (T=13, 3-stage).
- Fast dev: `shapes3d_sqvae2_32_S_lite.yaml` (32px, same hierarchy).
- Next gen: `shapes3d_sqvae2_64_S_v4.yaml` (T=17, 4-stage) — incompatible ckpts with v2.

**MSE scale:**

| Metric | Formula | @ 64px T=13 |
|--------|---------|-------------|
| `loss/mse` | sum of squared errors / batch | ~10²–10³ when recon is OK |
| `loss/mse_per_pixel` | `mse / (C·T·H·W)` | ~0.001–0.003 target range |
| ARELBO distortion | `dim_x * log(mse) / 2` | large; use for loss, not dashboards |

Resume: `scripts/resume_vae_training.ps1` / `.sh` with `ckpt_path=...`.

---

## Part 3 — Predictor core

### Schedule & masks

- `PyramidSchedule.from_encoder_audit()` builds stages from yaml.
- Cross-attn masks use `rf_frames_for_shift` — MSK-06 golden hash in tests.

### BatchPrep

- Single `encode_tokens` pass; context embeds through `context_end[s]`.
- `forward_parallel(parallel_mode="full")` uses full native-T embeds at shift 0 (oracle CE upper bound).

### PredictorStage

- k=0: quantized context + shift embed.
- k>0: stream carry (`streams_only`).
- Output head **zero-initialized** — logits start uniform; hidden states still differ by shift (STG-06).

### ParentCondition

| Mode | Status |
|------|--------|
| `dual_stream` | Default — even shifts → O1, odd → O2 |
| `fused_hard` | Argmax parent indices → embed |
| `fused_soft` | **Removed** |

### Attention

| Stage | Backend |
|-------|---------|
| coarse | Reversible coupling (+ optional RevBackProp) |
| mid | Factorized, `t_window=3` |
| fine | Factorized, `t_window=1`, `spatial_window=8` |

---

## Part 4 — Inference

| Mode | Behavior | Gap |
|------|----------|-----|
| `train` | Teacher-forced context, envelope DFS | — |
| `parallel` + `full` | Oracle shift-0 embeds | Upper-bound CE; may OOM on fine stage |
| `parallel` + `context` | Train-like context length | Safer for val on limited GPU |
| `autoregressive` | Rolling finest embed buffer on k>0 | Mid/top still `streams_only` — partial exposure bias |

INF-02: with zero-init head, parallel vs AR differ in **logits shape** (full native T vs context window), not necessarily argmax indices.

---

## Part 5 — Test coverage

| Group | Highlights |
|-------|------------|
| EV2 | Encoder v2, prefix stability, tap keys |
| RF / ENV / TGT | Schedule math, envelope order |
| MSK | Causal + shift-aware cross |
| REV | Invertibility, gradcheck, RevBackProp smoke |
| STG / ORC / BAT | Stage shifts, orchestrator CE, batch prep |
| INF / PAR | AR rollout, parallel memory modes |
| PRED | CE baseline, e2e checkpoint (gpu mark) |
| VQ | sqvae2-only hierarchical tests + `mse_per_pixel` |

**Total:** 119+ tests (run full suite before training).

---

## Remaining priorities

### P0 — Before trusting predictor metrics

1. Resume VAE to `val_ema/mse_per_pixel` plateau (`scripts/resume_vae_training.*`).
2. Gate predictor on `validate_v2_vae_gate.py --ckpt <best>`.

### P1 — Research quality (optional)

3. Full AR exposure bias (feed committed indices into mid/top, not only finest rolling_ctx).
4. `inference.commit: sample` (stochastic decode) — argmax only today.

### P2 — Nice to have

5. More ORC golden tests (ORC-02–07).
6. EV2-03 strict cross-T **quantized** index prefix test (activations already stable).

---

## Data flow (verified)

```
RGB (B,3,T,H,W)
  └─► VAE.encode_tokens ──► levels[s].indices
        └─► embed [:context_end+1] ──► context_embed[s]

Envelope DFS (s,k):
  parent = parent_by_stage[s-1]
  PredictorStage.execute_shift → CE at query_frame
  stream_carry[s,k] = (O1, O2)

pred_indices → decode_from_indices(zero acts) → pred MSE on RGB[:,:,t_context:t_total]
```

---

## Conclusion

Path C is **implementation-complete** for the locked training spec. Observability (per-pixel MSE, CE baselines, parallel modes) and repo slimming (sqvae2-only) are done. Treat predictor experiments as **blocked on VAE recon quality** until resume training plateaus.

---

## Addendum 2026-06-11 — review corrections (Path A)

The deep review (DEEP_REVIEW_2026-06-11.md) falsified several "Correct" verdicts
above. Status after the `path-a/elbo-pyramid` branch:

| Review finding | Original audit verdict | Fix | Status |
|---|---|---|---|
| ELBO KL mean-reduced over positions and K (~1e5-1e6x too weak) → codebook collapse | "VAE configs Operational" | KL sums in `gaussian_sq.py`; `kl_beta`/`kl_warmup_steps`/`grad_clip`; tests `test_elbo_scaling.py` | Fixed |
| All levels down-fused to 3x4x4 latent; mid layer dominates recon (ablation dMSE +0.234 vs +0.020 coarse) | not audited | `temporal_up` u2 + `decoder_source: finest_state` + `dec_ddconfig`; guard `validate_state_chain`; tests `test_pyramid_topdown.py` | Fixed (Run 2 to confirm) |
| `codebook_proj` frozen at random init (`@torch.no_grad` in batch prep) | "Predictor stages Correct" | embedding moved into grad context; test `test_grad_flow.py` | Fixed |
| pred-MSE has no gradient, in the loss + ckpt monitor; per-pixel double division | "Lightning/training Correct" | logging-only (`log_pred_mse`), monitor `val/loss_ce_ar`, per-pixel keys dropped | Fixed |
| `parallel(context)` identical to train mode (not an oracle) | "forward_parallel Correct" | `val/*_tf` aliases + doc note; legacy keys kept one release | Renamed |
| Fine-stage cross q/k get exact-zero grads (1-to-1 spatial mask, single overlapping parent frame at the query) | "Masks Correct" (MSK-06) | `cross_spatial_window` (3x3 neighborhood); canary test in `test_grad_flow.py` | Fixed (opt-in) |
| AR finest re-entry never trained; horizon temporal_pos untrained → `val/loss_ce_ar` grows 22→35 | "Inference Partial" | NOT yet fixed — Path C re-entry training (next phase) | Open |
| Native access-violation crash in full pytest run (test helper built FULL attention over the 13x32x32 fine stage → multi-GB attention matrix under memory pressure; multi-threaded OpenMP made it flakier) | "119+ passing" | factorized mid/fine stages in `test_predictor_logging._build_stack` + single-threaded torch in `tests/conftest.py` | Fixed |
