# Session summary — Path 3 predictor refactor, FSQ predictor blob, factorized head (2026-06-15/16)

> Standalone log of what was done, what we learned, and what remains. Companion:
> [PROJECT_MEMORY.md](./PROJECT_MEMORY.md) (§0.1/§0.2 hold the durable version),
> [PLAN_V2_FSQ_PREDICTOR.md](./PLAN_V2_FSQ_PREDICTOR.md) (the master plan; §0.5 = status).

---

## 1. What was accomplished

1. **Path 3.0/3.1 predictor refactor (behavior-preserving).** Tag `pre-path3-baseline`.
   Added `enums.py`, `shift.py` (+ `schedule.envelope_shifts()`), frozen
   `PredictorConfig`/`StageConfig` (`config.py`), unified `CrossConditioning`
   (`cross_conditioning.py`) with legacy-checkpoint `_load_from_state_dict` remap, and a
   single `RolloutPolicy` hierarchy (`rollout.py`) collapsing
   `EnvelopeOrchestrator._forward_envelope` → `_run_envelope(batch, policy)`. Guarded by a
   **golden CE fixture** (`tests/predictor/test_golden_refactor.py` + `_golden/`) asserting
   bit-equality across train/parallel(full|context)/autoregressive.
2. **Gate recalibration (quantizer-aware)** in `scripts/token_quality_report.py`. SQ-era G2
   (exact-index persistence) and G3 (coarse≥mid≥fine ablation) are invalid for the FSQ
   pyramid on rotating shapes; for FSQ/LFQ, G2 → temporal-MI hierarchy, G3 → no-dead-layer.
   Rightsized FSQ run `dr8vrkt8` then PASSES the gate (recon 0.00013/px). **Caveat found
   later:** that MI is a sparse-count *overestimate* (see §3).
3. **FSQ→predictor wiring fix.** `quantizer_builder.resolve_quantizer_sizes` derives
   `size_dict`/`dim_dict` from FSQ `levels` (the predictor build used to KeyError on FSQ
   configs). Added optional `predictor.codebook_dim` override. New config
   `shapes3d_fsq_32_S_lite_pyr_rightsized_predict.yaml`.
4. **Factorized per-channel FSQ prediction head** (`predictor.output_mode: factorized_fsq`).
   Per-channel softmaxes; summed per-channel CE (= ln K at uniform); commit = per-channel
   argmax → recompose via the FSQ basis. Composite mode is the default and stays golden
   bit-equal. **Implemented + verified, but does NOT solve the blob** (see §3).

**Test state:** full suite **213 passed, 1 skipped** (GPU e2e), peak ~7.9 GiB, ~4 min.

---

## 2. Problems

- **P1 — The predictor produces blobs (the headline problem).** On the FSQ tokenizer, the
  predictor outputs a featureless gray blob for ALL modes (train/parallel/autoregressive),
  while `vae_recon` (the tokenizer decoding GT tokens) is sharp. So the tokenizer is fine;
  the predicted *tokens* are degenerate. Run `6x3dvkv8`.
- **P2 — The predictor converges to the per-position marginal.** Token accuracy ≈ chance
  (coarse 1.3% / mid 0.9% / fine 8.8% composite), predicted entropy ≈ uniform; it captures
  ~10% of the gate's claimed MI. Decoding a near-marginal token field gives the average
  blob.
- **P3 — Coarse/mid FSQ tokens are intrinsically hard to predict frame-to-frame.** Held-out
  same-position bigram ≈ random for coarse/mid; per-channel held-out ≈ random for coarse/mid
  (fine has real per-channel signal). The factorized head left **mid dead-flat at the
  marginal for 4000 steps** (zero learning).
- **P4 — The gate didn't catch P1.** Its temporal-MI is a sparse-count overestimate on
  high-entropy codes, so it passed a prediction-hostile tokenizer.
- **P5 — Recon vs predictability tension.** The only ways tokens became predictable in this
  project (old SQ baseline coarse CE 3.5; SQ fine here) were via codebook **collapse** =
  reconstruction loss. FSQ is collapse-free → recon-optimal → prediction-hostile.

---

## 3. Insights (what we actually learned, with evidence)

- **The tokenizer is healthy; the problem is downstream prediction.** `vae_recon` sharp,
  recon 0.00013/px, codebooks well-used. The blob is a *prediction* failure.
- **Two causes; #2 dominates.**
  - **#1 composite-index/head mismatch (real, fixed, modest):** FSQ index = mixed-radix of
    per-channel codes; a monolithic K-way head needs all channels right at once (composite
    top-1 ≈ product → ~1%) even though per-channel signal exists (coarse ~20-30%). Factorized
    head removes this penalty → pixel pred-MSE 0.185→~0.16, fine layer learns.
  - **#2 low coarse/mid predictability (dominant, unsolved):** with the factorized head,
    **mid per-channel acc stayed at the marginal (~0.20) for 4000 steps; coarse barely moved
    (0.20→0.25)**; coarse/mid value-space MSE was *worse* than a constant. The decoded image
    is still a blob ([factorized_recon_compare.png](./wandb_analysis/factorized_recon_compare.png)).
- **Why coarse/mid are unpredictable:** (a) the shape *rotates* → content moves across grid
  positions, so same-position prediction is useless and the predictor isn't learning
  motion-compensation (neither factorized@4k nor composite@38k steps); (b) **FSQ boundary
  chaos** — a smoothly-moving latent crosses the fine lattice boundaries, so per-channel codes
  flip near-randomly even between adjacent frames; (c) aggressive coarse/mid temporal
  downsampling (3/5 token-frames for 9 video-frames) makes consecutive coarse/mid frames far
  apart in content.
- **No output head can fix unpredictable targets.** The factorized head is the correct FSQ
  architecture and a prerequisite, but it is **necessary-not-sufficient**.
- **SQ doesn't escape it.** Probe of the trained SQ (native elbofix): coarse held-out bigram
  ≈ random (like FSQ); mid only mildly above marginal; the one big win (fine 43.6%) was pure
  **collapse** (95/1024 codes). The historical "SQ coarse CE 3.5" came from *severe* collapse
  (53/4096 codes) — a broken tokenizer, not inherent predictability.
- **The gate metric must measure HELD-OUT predictability**, not sparse-count MI.

---

## 4. Next steps (prioritized)

The bottleneck is **token-space dynamics (tokenizer + temporal topology + motion)**, not the
predictor head. The head work (factorized) is kept but the next experiments must target
predictability.

1. **Predictability-aware tokenizer (the main lever).** Retrain the FSQ tokenizer with a
   **temporal-coherence penalty** (encourage consecutive frames' codes/values to be similar —
   the `temporal_kl_weight` idea adapted for FSQ: e.g. `w·‖value_t − value_{t-1}‖` or a
   code-persistence reward) and/or **less aggressive coarse/mid temporal downsampling**
   (3/5/9 → e.g. 5/7/9 so consecutive coarse/mid frames are closer in content). Accepts a
   little recon for much more predictable tokens — justified: the dissertation contribution
   is the predictor.
2. **Held-out-predictability gate.** Before spending predictor GPU, screen the tokenizer with
   a train/test-split next-frame predictability metric (per-channel and/or value-space), and
   demote the sparse-count MI to informational. This would have caught P1 up front.
3. **Predictor motion modeling — RoPE (Path 3.2).** Relative positions to let attention track
   the rotation/motion; also fixes the untrained-absolute-position AR divergence.
4. **Dense supervision (Path 3.4a).** ~5–8× more signal/step at zero extra forward compute.
5. **Re-run the predictor** on a predictability-improved tokenizer with the factorized head +
   RoPE, and only then judge AR divergence (the original "train-what-you-test" goal). Today
   even teacher-forced/parallel blob, so the AR gap is secondary until prediction works at all.

### Possibility of going back to SQ-VAE
Legitimate but **not a fix for the blob**, and it costs more than it buys:
- **What SQ buys:** a single learned index (the monolithic K-way head is the natural fit — no
  composite/product penalty; the factorized FSQ head already neutralizes this for FSQ); and a
  cleaner "categorical prediction over one learned vocabulary" story with the codebook copied
  as warm-started input embeddings.
- **What SQ does NOT buy:** inherent coarse/mid predictability at healthy usage (≈ FSQ, per
  the probe). Its predictability advantage is collapse = recon loss. It also re-introduces the
  collapse-tuning + soft/hard-gap headaches FSQ was chosen to avoid.
- **Verdict:** decide SQ-vs-FSQ on the *tokenizer* axis (recon / story / collapse tolerance),
  NOT as a predictor fix. If you do go SQ, pair it with #1 (tune for **moderate** usage +
  temporal coherence) — plain SQ will blob just like FSQ. The cheapest route to a predictable
  tokenizer is #1 on **either** family; FSQ keeps the better recon and avoids collapse tuning.
  Recommendation: **stay on FSQ + add temporal coherence**, unless the dissertation
  specifically needs the single-codebook story.

---

## 5. Remaining predictor implementation per PLAN_V2 (NOT yet implemented)

Done this session: P3.0 hygiene, P3.1 structural refactor (§5), gate recalibration, FSQ→predictor
wiring, factorized head (new, not in the original plan). **Gated by GATE-T = tokenizer
frozen + pre-tokenized; in practice now gated by the predictability work above.** Remaining:

| Plan | Item | Status / notes |
|---|---|---|
| §6.2 (H2a) | **Mask-polarity SDPA refactor** — rewrite `mha_self_attention` to unpack q/k/v + non-causal SDPA path; flips mask polarity for all callers. Its own bit-equality gate (`test_mha_polarity_equivalence`) **before** RoPE. | NOT STARTED (prereq for RoPE) |
| §6.1/6.3/6.5 | **Factorized 3D RoPE** (`predictor/attention/rope.py`), per-axis budgets, cache grows for AR; run at head_dim≥16. GATE-P1: parallel-CE ≤ baseline. | NOT STARTED |
| §6.4 | **Stability kit** — QK-RMSNorm, pre-RMSNorm, SwiGLU (each behind a flag, default legacy). | NOT STARTED |
| §7.1 (M1) | **FlexAttention** (`predictor/attention/flex.py`) + **mandatory SDPA-additive fallback**; Windows `torch.compile(flex)` smoke first (likely falls back → re-scope GATE-P2 to SDPA-additive). | NOT STARTED |
| §7.2 (M2) | **KV-cache** (`predictor/attention/kv_cache.py`) prefill/decode, two-stream for reversible; equivalence test incl. reversible block. `commit_embed` is the AR seam (already partly set up: `rollout.Autoregressive` owns commit). | NOT STARTED |
| §7.3 (L1) | **Pre-tokenized dataset** — cache frozen-VAE indices keyed by ckpt×config×dataset hash; `schedule_from_indices`; gate `log_images`/`log_pred_mse` off (no activations). Big speedup (VAE encode dominates step time — measured 565 ms/step). | NOT STARTED |
| §7.4 (H5) | **bf16 for predictor attention only** (must NOT wrap the frozen VAE encode — would corrupt gated tokens). | NOT STARTED |
| §7.5 | **torch.compile** the VAE for token-cache build / logging-decode only (not predictor stages). | NOT STARTED |
| §8.1 (H3) | **P3.4a dense causal supervision** — supervise every t→t+1 transition (~5–8× signal/step, zero extra forward); mandatory `_lasttok` CE gate (GATE-P3a). | NOT STARTED |
| §8.1 (B3) | **P3.4b scheduled sampling** — `ScheduledSampling(Autoregressive)` subclass (the seam exists: `rollout.py`); p-anneal 1→0.5; write commit_embed into the KV-cache. GATE-P3b (val_ce_ar tracks tf) = the plan's MAIN RESULT. | NOT STARTED |
| §8.3 | **P3.5 MaskGIT** within-frame iterative decode; **P3.6 VAR** next-scale. Optional, behind flags. | NOT STARTED |
| §10 R8 | **Reversible `rev_back_prop` gradcheck** (incl. parent grads) — BLOCKING before publishing any reversible result. | NOT STARTED |
| §5.6 M4 | **`video_hier_vqgan.py` dead-code removal** (GAN/disc/perceptual + VAR LR-anneal bodies) — deferred to after the live tokenizer run, on the tokenizer/P0 track. | DEFERRED |

**New items surfaced this session (not in the plan):**
- Factorized FSQ head — DONE (kept; insufficient alone).
- Temporal-coherence tokenizer objective — RECOMMENDED, NOT STARTED (Path-2/tokenizer track).
- Held-out-predictability gate metric — RECOMMENDED, NOT STARTED.

---

## 6. Artifacts, runs, commands

- **W&B:** tokenizer `dr8vrkt8` (rightsized FSQ, gate-passing, recon 0.00013/px);
  predictor `6x3dvkv8` (composite head, blob, 38k steps).
- **Verification artifacts:** `docs/wandb_analysis/factorized_recon_compare.png` (blobs),
  `factorized_verify2.json` (4000-step metrics), `gate_dr8vrkt8.json`.
- **Config:** `configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr_rightsized_predict.yaml`
  (now `output_mode: factorized_fsq`).
- **Tests:** `python -m pytest tests -q` → 213 passed, 1 skipped (run when GPU/RAM free;
  ~8 GiB peak; use targeted predictor tests while the GPU is busy).
- **Branch:** `refactor/v2-predictor-and-fsq`, tag `pre-path3-baseline`. All work uncommitted
  pending review.
