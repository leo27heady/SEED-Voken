# Path A implementation plan — ELBO fix + temporal pyramid + finest-state decoding

> **Status (2026-06-11, branch `path-a/elbo-pyramid`):** Phases 0–4 and 7 are
> **implemented and verified** (ELBO sums confirmed numerically in the smoke run:
> per-layer KL −398/−2430/−15900 = −positions·lnK at init; gate script reproduces
> the baseline failure G1–G3; pyramid roundtrip exact; predictor wiring check
> passes with live codebook_proj + cross q/k grads). Phase 5 (GPU runs 1–2 +
> gating) and Phase 6 (predictor retrain) are ready to launch — commands in the
> README. Extra fix landed beyond plan: `cross_spatial_window` (dead fine-stage
> cross q/k under the legacy 1-to-1 mask, found by the Phase-2 regression test).

**Date:** 2026-06-11
**Basis:** [DEEP_REVIEW_2026-06-11.md](./DEEP_REVIEW_2026-06-11.md) (findings §1–§2, fix list §4, Path A §5)
**Goal:** a tokenizer whose levels have the intended semantics (coarse = stable global scene, fine = detail), healthy codebook usage, and temporally predictable codes — gated by measurable token-quality metrics — plus the P0 predictor fixes that don't depend on the retrain.

**Corrections vs the review's original Path A sketch (rethink results):**

1. `InjSQBlock` upsamples **spatially only** (`Upsampler(..., block_size=(1,2,2))`). Naively switching to `token_grid: pyramid` would put *all* levels at T=3, destroying the 1/2/4 shift hierarchy and silently desynchronizing `PyramidSchedule` (which derives `native_t` from the T-halving chain). The pyramid work therefore **requires temporal upsampling** in the top-down path. Good news: the existing `Upsampler` with `block_size=(2,2,2)` produces T 3→5→9 exactly (2T−1 after its frame-drop), mirroring `causal_output_length`'s inverse.
2. KL β **warm-up defaults off**. With correct sums, KL_max ≈ Σ positions·ln K ≈ 19k nats vs distortion ≈ 59k — same order of magnitude; SQ-VAE's self-annealing is designed to work at full strength from step 0. β stays configurable as an escape hatch.
3. Tokenizer changes land as **two controlled runs** (ELBO-fix-only, then ELBO+pyramid) so each effect is attributable.

---

## Phase 0 — Hygiene and baseline freeze (½ day)

| Step | Change | Files |
|---|---|---|
| 0.1 | New branch `path-a/elbo-pyramid` off `experiment/predictor-integration`. | — |
| 0.2 | Deflake pytest: add `torch.set_num_threads(1)` in a root `tests/conftest.py` (fixes the Windows OpenMP access-violation crash; verified the suite passes single-threaded: 142/142). | `tests/conftest.py` |
| 0.3 | Pin seed: `seed_everything: 1234` (not `true`) in all active configs — dissertation reproducibility. | `configs/.../shapes3d_sqvae2_32_S_lite*.yaml` |
| 0.4 | Freeze baseline numbers: commit the review scripts' outputs (ablation table, token stats, W&B trends) so post-fix runs have a fixed comparison point. Keep `scripts/{layer_ablation_review,token_stats_review,grad_flow_review,analyze_wandb_review}.py`. | `docs/wandb_analysis/baseline_lite_2026-06-11.md` (new) |
| 0.5 | Delete dead code: `src/Open_MAGVIT2/modules/predictor/objectives/` (referenced by nothing in `src/`; update the one test that imports it or move the kept logic). | `objectives/`, `tests/predictor/test_predictor_loss.py` |

---

## Phase 1 — ELBO correctness (1 day, the highest-leverage diff)

### 1.1 KL sums in `GaussianSQQuantizer` (`gaussian_sq.py`)

Replace the mean-reductions with sums over positions and K (batch mean only):

```python
# discrete: sum over (T,H,W) positions and K, mean over batch
kld_discrete = (
    (prob_pos * (log_prob_pos - log_prob_pri)).sum(dim=(1, 2, 3, 4))
).mean()

# continuous (flg_loss_continuous layers): SQ-VAE weighted-MSE form —
# sum over (C,T,H,W), mean over batch
if self.flg_loss_continuous:
    precision_sum = 1.0 / torch.clamp(var_q_pos.sum(), min=1e-10)
    kld_continuous = (
        (z - z_to_decoder).pow(2).sum(dim=(1, 2, 3, 4)) * (0.5 * precision_sum)
    ).mean()
```

Notes:
- `log_prob_pri` paths (`zero`/`uniform`/`learned`) need no change — only the reduction.
- `usage_reg` (perplexity-target MSE) is now dwarfed by the summed KL; re-check its weight if ever re-enabled (default 0 — fine).
- Keep `calc_distance` fp32 promotion as is.

### 1.2 β control in the Lightning model (`video_hier_vqgan.py`)

- Add `loss_cfg.kl_beta` (default `1.0`) and `loss_cfg.kl_warmup_steps` (default `0`).
- In `training_step`, compute `beta = kl_beta * min(1.0, (step + 1) / max(kl_warmup_steps, 1))` and pass `kl_weights * beta` (or `beta` alone when `kl_weights is None`) into `compute_hier_elbo_loss`. Log `train/kl_beta`.
- `compute_hier_elbo_loss` needs no signature change (β folds into `kl_weights`); just allow a scalar tensor broadcast or build the tensor in the model.

### 1.3 Gradient clipping for manual optimization

Lightning ignores `gradient_clip_val` in manual-optimization modules. In `training_step`, after `manual_backward(loss)` and before `opt.step()`:

```python
self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
```

Make the value configurable (`loss_cfg.grad_clip`, default 1.0).

### 1.4 Tests (new `tests/test_elbo_scaling.py`)

- **KL hand-check:** tiny tensor (B=2, K=5, grid 1×2×2), compare `kld_discrete` against a literal Σ q(log q − log p) computed with loops. Both `zero` and `uniform` priors.
- **Magnitude sanity:** on a random batch through the full `VideoHierVQModel` (lite config, CPU), assert `0.01 < kl_total / |distortion| < 10` at init (catches any future reduction regression).
- **β schedule:** warmup 100 steps → β at step 0 ≈ 0.01, at ≥100 = kl_beta.

---

## Phase 2 — Predictor P0 fixes (independent of retrain; 1 day)

### 2.1 Resurrect `codebook_proj` (`batch_prep.py`)

Remove the `@torch.no_grad()` decorator from `encode_and_schedule`; wrap only the VAE call:

```python
def encode_and_schedule(self, vae, video, stages):
    with torch.no_grad():
        tokens = vae.encode_tokens(video, flg_quant_det=True)
    ...  # embedding via stages[s].embed_indices now records grad for codebook_proj
```

Same treatment for `build_full_context_embed` (consistency; val runs under Lightning's no-grad anyway).

### 2.2 Stop paying for a gradient-free MSE every step (`video_hier_predictor.py`)

- Wrap `_decode_horizon_mse` in `torch.no_grad()` (it can never produce gradients — indices are argmax).
- Compute it only every N training steps (`loss.pred_mse_every_n_steps`, default 50 = `log_every_n_steps`); always compute in validation.
- **Remove it from the optimized loss** (`total = lambda_ce * out.loss_ce` only) — keep logging it. `lambda_pred_mse` becomes a pure logging toggle; rename config key to `log_pred_mse: true` (accept the old key with a deprecation warning to avoid breaking the v2/v4 predict configs).

### 2.3 Fix model selection (config + metrics)

- `ModelCheckpoint.monitor: val/loss_ce_ar` (pure CE; no non-differentiable MSE mixed in). Update `filename` template accordingly.
- Fix the double division: `pred_mse_per_pixel` — `loss_mse` is already a pixel mean; drop the `/ n_pix` lines in `metrics.py` (keep the raw value, rename log keys to `pred_mse` only, or divide the *sum* if a per-pixel scale is wanted — pick one, document it).
- Rename the misleading val mode: `val/loss_ce_parallel` → `val/loss_ce_tf` ("teacher-forced/stream-carry"; it is train-mode-on-val when `parallel_mode: context`). Keep old keys for one release if W&B continuity matters — simplest: log both, note in README.

### 2.4 Tests (extend `tests/predictor/`)

- **Gradient-flow regression** (`test_grad_flow.py`, adapted from `scripts/grad_flow_review.py`): perturb zero-init params, one forward+backward in train mode, assert every `requires_grad` parameter of `predictor_stages` has a non-None, nonzero-somewhere grad — **explicitly including `codebook_proj`** — except the intentionally frozen `codebook_embed`.
- Assert `out.loss_mse is None or not out.loss_mse.requires_grad` stays true (documents the design).
- Assert train-mode and AR-mode CE identical at k=0 for all stages on a fixed seed (mode-consistency canary).

---

## Phase 3 — Temporal pyramid + finest-state decoding (2–3 days, the core architecture change)

### 3.1 Temporal upsampling in the top-down path (`video_inj_topdown.py`)

- `InjSQBlock.__init__`: new arg `up_block_size: Tuple[int,int,int] = (1, 2, 2)`; use it for `self.spatial_up = Upsampler(z_channels, block_size=up_block_size)` (rename attr to `state_up`; keep `spatial_up` as alias for ckpt compat or migrate state dict keys — simplest is keep the name).
- `SQVAE2TopDown.__init__`: derive each u2 layer's block size from the **tap T chain**: read `hierarchy.sequence_length` and the per-level tap shapes via the existing audit helpers, or accept an explicit `hierarchy.temporal_up: [2, 2]` list (explicit beats inference; validate it against the shape audit). For the lite config: `(2,2,2)` on both u2 layers → z_state T: 3→5→9 (Upsampler's drop rule gives 2T−1 ✓), H/W: 4→8→16.
- `_compute_prior_fields` uses `block.spatial_up` — automatically consistent once the block is parametrized.
- `decode_from_indices` InjSQ branch — already calls `block.spatial_up`; no change beyond the rename.

### 3.2 Finest-state output (`video_inj_topdown.py` + `video_hier_vqgan.py`)

- New hierarchy option `decoder_source: latent | finest_state` (default `latent` = current behavior, full backward compat).
- `SQVAE2TopDown.forward` / `forward_progressive` / `decode_from_indices`: when `finest_state`, return `z_state` (the post-final-injection state at the finest grid) instead of the down-fused `z_latent`. Keep computing `z_latent` only if cheap/needed — it isn't, so skip it entirely in `finest_state` mode (the per-layer `fuse_to_latent` calls go away).
- `forward_progressive` in `finest_state` mode: partial outputs live on intermediate grids; `align_spatial(partial_state, finest_thw)` before returning so a single decoder can render every stage row.
- Validate: `validate_hierarchy_taps` / `shape_audit` must accept the u2 chain with temporal ups — extend the audit to check `tap T == expected chain T` per level and fail loudly on mismatch (this is the guard against correction #1 regressing).

### 3.3 Decoder for the finest grid (`video_hier_vqgan.py` + `improved_video_model.py`)

- New optional `dec_ddconfig` on `VideoHierVQModel` (default `None` → use `ddconfig`, current behavior).
- For the lite pyramid config: input `z` is `(B, 16, 9, 16, 16)`; decoder needs **one spatial doubling, no temporal change** → `dec_ddconfig: {z_channels: 16, ch: 64, ch_mult: [1, 2], num_res_blocks: 2, num_groups: 8, out_ch: 3, in_channels: 3, resolution: 32, double_z: false}`. (`ch_mult[0]==1` → the single `Upsampler` is `(1,2,2)` — verified against the `Decoder` constructor logic.)
- `get_last_layer()` already points at `decoder.conv_out` — unchanged.
- Note in config comments: decoder FLOPs rise (most blocks now run at 9×16×16 instead of 3×4×4); `ch: 64`, `num_res_blocks: 2` is the starting point — tune down if step time is unacceptable.

### 3.4 Predictor compatibility (verification only — no code change expected)

With temporal ups restored, token grids are **identical to native mode**: (3,4,4), (5,8,8), (9,16,16) → `PyramidSchedule.from_encoder_audit`, masks, shifts, and all predictor code are untouched. Add an assertion test (3.5) so this stays true.

### 3.5 Tests (new `tests/test_pyramid_topdown.py`)

- Build `SQVAE2TopDown` with the pyramid config; feed fake taps with T 3/5/9; assert per-level `indices` grids `(3,4,4)/(5,8,8)/(9,16,16)` and `z_out` shape `(B,16,9,16,16)`.
- Roundtrip: `decode_from_indices(level_indices)` ≡ forward's `z_out` under `flg_quant_det=True` (same tolerance pattern as existing golden tests).
- Decoder smoke: `dec_ddconfig` decoder maps `(B,16,9,16,16) → (B,3,9,32,32)`.
- Schedule invariance: `PyramidSchedule.from_encoder_audit` on the pyramid config equals the one from the native lite config (stage specs + native_t).
- Progressive: 3 partial outputs, all decodable to `(B,3,9,32,32)` after alignment.
- Run the existing `e2e_checkpoint`/golden suites to confirm `decoder_source: latent` defaults are bit-identical (no regression for old ckpts).

---

## Phase 4 — Configs and gate tooling (1 day)

### 4.1 New configs (`configs/Open-MAGVIT2/gpu/`)

| File | Purpose | Key deltas vs `shapes3d_sqvae2_32_S_lite.yaml` |
|---|---|---|
| `shapes3d_sqvae2_32_S_lite_elbofix.yaml` | **Run 1**: isolate the ELBO fix | unchanged architecture; pinned seed; `loss_cfg: {kl_beta: 1.0}`; new `default_root_dir`/`dirpath`/wandb `name: sqvae2_32_S_lite_elbofix` |
| `shapes3d_sqvae2_32_S_lite_pyr.yaml` | **Run 2**: ELBO + pyramid + finest-state | `token_grid: pyramid`, `blocks_sq: "h4_w4_x1,h8_w8_u2,h16_w16_u2"`, `temporal_up: [2, 2]`, `decoder_source: finest_state`, `dec_ddconfig` (§3.3), same `size_dict/dim_dict` (controlled comparison), wandb `name: sqvae2_32_S_lite_pyr` |
| `shapes3d_sqvae2_32_S_lite_pyr_smoke.yaml` | 2-epoch CPU smoke for CI/dev | tiny `dataset_size`, `max_epochs: 2`, `accelerator: cpu` |
| `shapes3d_sqvae2_32_S_lite_pyr_predict.yaml` | predictor on Run-2 ckpt (Phase 6) | `vae_config` → pyr yaml; `vae_ckpt` → Run-2 best; `monitor: val/loss_ce_ar`; `log_pred_mse: true` |

Codebook sizes stay `[4096, 2048, 1024]` for runs 1–2 (attribution). A **Run 3** size ablation (`[512, 1024, 2048]`, coarse smallest) is pre-planned but only if gates show coarse perplexity saturating far below K.

### 4.2 Token-quality gate script (`scripts/token_quality_report.py`)

Consolidate `layer_ablation_review.py` + `token_stats_review.py` into one CLI:

```
python scripts/token_quality_report.py --config <vae yaml> --ckpt <path> [--n-videos 128] [--json out.json]
```

Reports per layer and checks the **Path A gates**:

| Gate | Threshold (lite, rotating shapes) | Current baseline |
|---|---|---|
| perplexity / K | ≥ 0.10 every layer | 0.011 / 0.020 / 0.020 |
| token persistence P(unchanged) | coarse > mid > fine AND coarse ≥ 0.5 | 0.028 / 0.123 / 0.105 (inverted) |
| ablation Δ-MSE ordering | coarse ≥ mid ≥ fine | 0.020 / 0.234 / 0.031 (mid-dominant) |
| recon `val_ema/mse_per_pixel` | ≤ 0.0027 (no worse than baseline) | 0.0027 |

Exit nonzero if a gate fails (usable in scripts); print a markdown table for the dissertation.

### 4.3 W&B fetch helper

Keep `scripts/fetch_wandb_review.py` + `analyze_wandb_review.py`; parametrize run paths via CLI args instead of the hardcoded dict.

---

## Phase 5 — Training runs and gating (GPU time; ~1 day each on the lite setup)

```powershell
# from repo root, venv active, $env:PYTHONPATH set
# Run 1 — ELBO fix only (architecture unchanged)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_elbofix.yaml

# gate it
python scripts/token_quality_report.py `
  --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_elbofix.yaml `
  --ckpt <best ckpt from run 1>

# Run 2 — pyramid + finest-state (includes ELBO fix)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_pyr.yaml
python scripts/token_quality_report.py --config ...lite_pyr.yaml --ckpt <best ckpt from run 2>
```

Decision rules:

- **Run 1 passes perplexity gate but persistence still inverted** → expected (fusion topology unchanged); proceed to Run 2 regardless; Run 1 is the ablation datapoint.
- **Run 2 passes all gates** → Phase 6.
- **Run 2 fails perplexity after the KL fix** → escalate per review §5 Path D (EMA codebook / dead-code restarts, then FSQ/LFQ swap) — pre-scoped, separate plan.
- **Run 2 passes usage but coarse persistence < 0.5** → add the coarse-layer temporal-smoothness KL (review §5 A.5) as Run 2b before considering Path D.
- Train to plateau this time (`val_ema/mse_per_pixel` flat over ≥3 evals) — the epoch-67 baseline was still improving when interrupted.

---

## Phase 6 — Predictor retrain on the new tokenizer (after gates pass)

Scope here is only what Path A unblocks (the deeper AR/re-entry redesign is review §5 Path C — separate plan):

1. Point `..._pyr_predict.yaml` at the Run-2 checkpoint; predictor architecture unchanged (grids identical — §3.4).
2. Expected effects to verify in W&B: coarse CE drops well below the old marginal-entropy wall (~3.5) as codes become temporally stable; `train/token_acc_stage_0` should leave the 8–17 % regime; the AR k>0 divergence will *still* be present (Path C scope) but should start lower because `codebook_proj` now trains.
3. Re-run `scripts/grad_flow_review.py` once against the new predictor to confirm `codebook_proj` grads are alive in the real config.

---

## Phase 7 — Docs, README, commands (½ day, lands with the code PR)

### `README.md`

- Quick start: add `$env:OMP_NUM_THREADS` note **removed** (fixed by `tests/conftest.py` instead); keep `python -m pytest tests/ -q`.
- Configs section: add the four new yamls with one-line descriptions; mark `shapes3d_sqvae2_32_S_lite.yaml` as *legacy baseline (pre-ELBO-fix; kept for comparison)*.
- Training section: add the Run-1/Run-2 commands and the token-quality gate command (the §5 block); state the gate thresholds table.
- Add a "Pipeline" line: tokenizer → `token_quality_report` gate → predictor.

### `docs/PATH_C_README.md`

- Update the "locked spec" notes: `decoder_source: finest_state` variant, temporal-up pyramid, new monitor `val/loss_ce_ar`, `log_pred_mse` rename, the `parallel→tf` metric rename.
- Update the repository-layout block (new configs, `token_quality_report.py`, removed `objectives/`).

### `docs/PATH_C_AUDIT.md`

- Append a 2026-06 addendum table: the five review findings, their fixes, and the commit/PR that closed each (the current audit marks areas "Correct" that this review falsified — the addendum keeps the audit honest for the dissertation record).

### `docs/Open-MAGVIT2-hierarchical-vq.md`

- Document the corrected ELBO reductions (sum semantics, β knob) and the `decoder_source` / `temporal_up` hierarchy options with the lite-pyr example block.

---

## Sequencing & effort summary

| Phase | Depends on | Effort | Deliverable |
|---|---|---|---|
| 0 hygiene | — | ½ d | branch, deflaked tests, frozen baseline |
| 1 ELBO | 0 | 1 d | corrected KL + β + clip + tests |
| 2 predictor P0 | 0 | 1 d | live `codebook_proj`, clean monitor, no dead decode |
| 3 pyramid | 1 | 2–3 d | temporal u2 + finest-state + dec_ddconfig + tests |
| 4 configs/gates | 1, 3 | 1 d | 4 yamls + `token_quality_report.py` |
| 5 runs | 4 | GPU days | gated checkpoints (Run 1, Run 2) |
| 6 predictor retrain | 5 | GPU day | predictor on healthy tokens |
| 7 docs | 1–4 | ½ d | README + 3 docs updated |

Phases 1+2 are independent and can be one PR each or a combined "P0 correctness" PR; Phase 3 should be its own PR (architecture change with golden-test evidence of default-path bit-compatibility).

## Risks / rollback

- **Pyramid run quality regresses vs native** → `decoder_source: latent` default preserves the old path exactly; Run 1's ELBO-fixed native checkpoint remains a working fallback tokenizer.
- **Summed KL overwhelms distortion early (posterior stays uniform, recon stalls)** → `kl_warmup_steps` (e.g. 2 epochs ≈ 2000 steps) or `kl_beta < 1`; log `train/kl_beta` so runs are interpretable.
- **Decoder at 9×16×16 too slow on the 1-GPU box** → drop `dec_ddconfig.ch` 64→32 or `num_res_blocks` 2→1; decoder cost is the knob, tokens are unaffected.
- **Old checkpoints** load fine: all new behavior is behind config flags with legacy defaults (`decoder_source: latent`, `temporal_up` absent, `kl_beta` only changes loss going forward).
