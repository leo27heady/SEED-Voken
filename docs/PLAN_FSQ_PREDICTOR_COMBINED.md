> **SUPERSEDED for implementation by [PLAN_V2_FSQ_PREDICTOR.md](PLAN_V2_FSQ_PREDICTOR.md)**
> (the implementation-ready master: full designs, sequencing DAG, gates, risk register,
> and the adversarial-review corrections). Keep this doc for the run-history narrative
> §0.0–§0.2 (Run 1 / Run 2 / Run 2c results) which the master references.

# Combined Plan — FSQ tokenizer (Path 2) + Predictor modernization (Path 3)

> Authored 2026-06-13 after the `cwisgodl` Run-1 gate. Supersedes the Path-2/3
> sketches in PROJECT_MEMORY §8. Companion: [PROJECT_MEMORY.md](./PROJECT_MEMORY.md).
> Scope: a robust, stable, efficient route that swaps the quantizer family for a
> collapse-free one **and** modernizes the predictor, sequenced so the two never
> change at the same time (tokenizer is always frozen + gated before predictor work).

---

## 0. Where we are (Run 1 = `cwisgodl`, the attribution baseline)

At step 109k / epoch 104 / 13.2 h, every closed-form ELBO anchor matches and
**SQ-VAE self-annealing is unambiguously working** — the signal the mean-reduced
baseline never produced:

- var L1 60 → **0.93**, L2 60 → **4.1** (annealing to the data-noise floor);
  L3 pinned at the **e⁵ = 148.4** clamp (dead, as designed).
- KL signs/magnitudes exact (kl_total −17.2k vs distortion +55k); L1/L2 raw KL
  rising toward 0 (sharpening), L3 frozen at the uniform bound −15 950.
- val mse/px **0.00272 and still at its minimum** at the last point — no overfit,
  EarlyStopping not yet triggered, but gains are marginal (−5 % over the last 4 h).

**Gate verdict (epoch=99 ckpt): G4 PASS, G1/G2/G3 FAIL — exactly as predicted.**
- G1: L2 healthy (ppl/K 0.26); **L1 ppl/K 0.046 is oversized-vocab, not collapse**
  (355 unique / 4096) → hard evidence for right-sizing coarse K. L3 dead (uniform).
- G3 ablation: L2-dominant still (ΔMSE 0.228 vs L1 0.028, L3 0.000) → finding #2
  persists on native; the pyramid is the structural fix.
- G2 persistence inverted (fine 0.42 > coarse 0.012) → finding #4 (coarse chaotic).

**Decisions:**
1. **Stop Run 1, start Run 2 (`lite_pyr`) now.** Run 1's scientific job (attribution
   baseline + gate) is done; the failures are structural and more steps won't move
   them. The epoch=99 ckpt + `gate_run1.json` are the record. On one GPU, continuing
   only delays Run 2.
2. **Run 2 is the prerequisite for everything below** — it validates the pyramid +
   `finest_state` topology with the *known* SQ quantizer, and becomes the **SQ-vs-FSQ
   ablation baseline** the dissertation wants. Do not skip it.

---

## 0.1 Run 2 result (`jijbhzdi`, pyramid + finest_state) — topology fixed, mirror failure exposed

Gate (epoch=39): **G4 PASS (recon 0.00078 — best of any run), G1/G2/G3 FAIL.** The
finest_state decoder did exactly its job and revived L3, but doing so **starved the
coarse/mid layers** — the textbook hierarchical-VAE "all information amortizes to the
layer the decoder reads most directly" pathology, and the exact mirror of the native
imbalance (finding #2).

| layer | role now | var (60→) | active codes | ppl/K | abl ΔMSE | keep-only MSE |
|---|---|---|---|---|---|---|
| L1 coarse | global base | **51.5** (not annealing) | ~14 | 0.002 | +0.246 | 0.235 |
| L2 mid | **vestigial** | **51.9** (not annealing) | ~20 | 0.003 | **+0.054** | 0.762 |
| L3 fine | **workhorse** | **2.87** (annealing) | ~708↑ | 0.249 | **+0.356** | 0.037 |

Ablation order is now **L3 > L1 > L2** (fine-dominant) — G3 wants the reverse.
Mechanism, confirmed in the trajectory: only L3 has a strong recon gradient
(decoder reads its state directly), so only L3's variance anneals (60→2.9) and its
codebook fills (1→708). L1/L2 keep var ≈ 51 with a near-uniform *soft* posterior
(ppl_frac 0.79/0.93) but argmax-collapse to ~15–20 codes — they carry almost no
information. **The high volatility the user sees is a symptom of this**: Gumbel noise
on under-determined coarse posteriors + the still-shifting L1/L2↔L3 balance makes the
distortion bounce (mse/px swings 0.0003↔0.0011 between logged steps).

**Root cause is information-theoretic, not a bug:** at 32 px the finest 9×16×16 grid
(2304 tokens) alone has ample capacity to represent the whole 9-frame clip, so there
is **no pressure to use the coarse layers at all** unless the objective imposes a
scale decomposition. The hierarchy here is currently cosmetic.

**The fix is already implemented:** `progressive_coding: true` (the pyr config has it
`false`). `forward_progressive` + `finest_state` decodes each partial state — after
L1, after L1+L2, after L1+L2+L3 — through the decoder and the loss averages their
distortions ([video_inj_topdown.py:636-641](src/Open_MAGVIT2/modules/vqvae/hierarchical/video_inj_topdown.py),
[hier_elbo_loss.py:32-44](src/Open_MAGVIT2/modules/vqvae/hierarchical/hier_elbo_loss.py)).
This forces the after-L1 state to reconstruct coarse structure on its own, the after-L2
state to improve it, etc. — i.e. it *imposes* the coarse→fine decomposition the data
doesn't force. Expected to fix both the imbalance (coarse layers get a real recon
gradient → variance anneals → codes fill, MI rises) and the volatility (annealed coarse
posteriors stop injecting Gumbel noise). This is the Laplacian-pyramid / progressive-
growing objective and the standard hierarchical-VAE anti-collapse remedy.

**CRITICAL — does the planned Path 2 (FSQ) fix this? NO.** FSQ cures *codebook
collapse* (few active codes), not *layer starvation* (a layer with no recon gradient).
On this topology FSQ would make L1/L2 use ~all their codes uniformly while still
contributing ~nothing to reconstruction — it converts "collapsed to 15 codes" into
"high-entropy but vestigial," which cosmetically passes G1 while remaining dead on MI
and ablation. **Layer balance must be solved first, on the SQ model, with progressive
coding; FSQ is layered on top of an already-balanced topology.** The predictor refactor
(Path 3) is downstream and cannot help either — starved coarse tokens carry no signal
for any predictor to learn.

**New immediate step — Run 2c:** `lite_pyr` + `progressive_coding: true`
(+ consider per-layer free-bits / `temporal_kl_weight` if coarse persistence still
lags). Gate it. This inserts **before** Path 2 in §4.

## 0.2 Run 2c result (`lle1hzce`, pyramid + finest_state + progressive_coding) — balance fixed, soft/hard gap exposed

`progressive_coding: true` did exactly what §0.1 predicted **and** revealed the next layer of the problem.

**Fixed (✓):** all three layers now anneal (var L1 60→1.05, L2→0.47, L3→1.75 — Run 2
had L1/L2 stuck ~51) and fill codebooks (active L1 27→202↑, L2→680, L3→735). Progressive
MSE is a clean coarse→fine decomposition (L1-only 503 → +L2 53 → +L3 4.7 summed).
Volatility dropped (distortion CV 0.034 vs Run 2's ~0.33). **Starvation solved.**

**New problem (✗) — soft-vs-hard quantization gap:** `val_ema/mse_per_pixel = 0.135`
(full recon, **argmax**, EMA) vs train **soft** full-recon 0.00017 — a ~800× gap. This
is **not EMA lag** (LitEma decay = 0.999, ≈700-step halflife; Run 2 reached val_ema
0.0007 by 25k, Run 2c is 0.135 at 87k = 190× worse with 3.5× the steps). Mechanism:
progressive coding forces each residual partial to reconstruct, so the model leans on the
continuous Gumbel-soft blend (`z_q = Σ p_k·cb_k`) to place each layer precisely; argmax
snaps each of 3 stacked residual layers to a single code and the errors **compound**. The
temperature floor 0.3 (τ ≈ 0.40 at 92k, never < 0.3) keeps training permanently soft.
Non-progressive runs tolerated τ=0.3 (codes co-adapt for the final argmax sum); the
3-stage residual structure does not.

**Decisive consequence — neither SQ run is usable alone:** Run 2 has great argmax recon
(0.0008) but starved tokens; Run 2c has balanced tokens that **don't decode in argmax
mode** (which is exactly what the predictor consumes). We need *both* properties.

**This crystallizes the combined target → FSQ + progressive coding.** FSQ uses
straight-through hard rounding, so train == eval quantization → **no soft/hard gap by
construction**, while progressive coding supplies the layer balance. They compose
cleanly: progressive fixes starvation, FSQ fixes the quantization gap (and codebook
collapse). The earlier §0.1 caveat still holds — FSQ *alone* won't fix starvation — but
FSQ *with* progressive coding is now the concrete next tokenizer. (SQ band-aid if staying
on SQ: anneal τ floor to ~0.05–0.1 **and** weight the progressive stages toward the full
stage — `compute_hier_elbo_loss` currently averages them uniformly — but this is fragile
versus FSQ's structural fix.)

**Immediate action:** run the gate on the best Run-2c ckpt. Prediction: recon ≈ 0.13
(confirming the argmax gap) **with** good per-layer ppl/active-codes/ablation (confirming
balance). That splits the diagnosis cleanly and justifies moving to FSQ+progressive.

> Note: the t=1 fuzziness in the **progressive L1-only** row is a separate, cosmetic
> visualization artifact (trilinear temporal alignment of the 3-frame coarse state maps
> output frame 1 → coarse frame 0); the full recon is unaffected. Details below in the
> chat analysis / memory `finding-pyramid-layer-starvation`.

---

## 1. Strategy & guiding principles

The two paths attack different layers and must be **decoupled in time**:

- **Path 2 (FSQ)** removes the *cause* of tokenizer instability (learned-codebook
  collapse / posterior-variance degeneracy) by construction. It is a quantizer swap
  behind the existing `LayerQuantizer`/`QuantizerResult` contract — the pyramid
  topology, encoder/decoder, ELBO loss plumbing, and gate all stay.
- **Path 3 (predictor)** is an internal **strangler rewrite** behind the stable
  `PreparedBatch`/`PredictorOutput`/metric-key/`PyramidSchedule` contracts.

**Invariants that make this safe (non-negotiable):**
1. Tokenizer and predictor never change in the same run. VQ is frozen + gated, then
   the predictor trains against pre-tokenized indices (field standard: MAGVIT-2,
   VideoPoet, Cosmos).
2. Every phase has a **green-test bar** and a **single success metric**; a phase that
   regresses its metric is reverted, not patched forward.
3. Behavior-preserving refactors land **before** behavior-changing ML, each verified
   against the contract tests (defaults must reproduce old numbers bit-for-bit).
4. The dissertation pillars live in the **predictor** (categorical prediction +
   entropy maps, hierarchy, in-stage reversibility) — none depend on the quantizer
   family, so swapping SQ→FSQ costs no thesis claim and *gains* an ablation chapter.

---

## 2. Path 2 — FSQ quantizer

### 2.1 Why FSQ
SQ-VAE keeps codes alive with machinery: Gumbel sampling, temperature decay,
posterior-variance self-annealing, the zero-prior entropy term, the `log_param_q_max`
clamp, and (fallback) learned priors. That machinery is the entire source of the
instability you have been fighting, and it is also what produced the L3 degeneracy.
**FSQ** ([Mentzer et al. 2023](https://arxiv.org/abs/2309.15505)) gets near-uniform
code usage with **zero auxiliary losses and no learned codebook** — it is what
MAGVIT-v2/LFQ-lineage and NVIDIA Cosmos (current SOTA discrete video tokenizer)
use. LFQ (binary, with entropy penalty) is the codebook-flavored alternative; FSQ is
primary because it has no tunable loss at all.

### 2.2 Math
Per layer, pick levels `L = [L_1..L_d]` (each in {4..8}); implicit vocab `K = ∏ L_i`.
For projected channel `h_i`:

```
z_i = round_ste( (L_i - 1)/2 · tanh(h_i) )           # in {0..L_i-1}, straight-through
index = Σ_i z_i · basis_i,   basis = cumprod([1, L_1, L_2, ...])   # mixed-radix
value_i = z_i / ((L_i-1)/2) - 1   ∈ [-1, 1]          # decode
```

`aux_loss = 0`. Usage is ~uniform by construction → no collapse, no dead codes,
no annealing schedule. The thesis's per-token entropy/confidence map still comes from
the **predictor's** softmax over `K`, unchanged.

**Level choices** (start by matching Run-2 K for a clean SQ↔FSQ ablation, then exploit
FSQ's freedom to right-size — coarse is oversized per §0):

| layer | role | match-K levels (K) | right-sized levels (K) |
|---|---|---|---|
| L1 coarse 3×4×4 | chaotic, few positions | `[8,8,8,8]` (4096) | `[8,8,8]` (512) |
| L2 mid 5×8×8 | dominant | `[8,8,8,4]` (2048) | `[8,8,6,6]` (2304) |
| L3 fine 9×16×16 | gets recon grad via finest_state | `[8,8,4,4]` (1024) | `[8,8,8,5]` (2560) |

### 2.3 Contract & integration (file-by-file)
The seam already exists; `LayerQuantizer.forward`'s `var_q_pos` is `Optional` in
[base.py:26](src/Open_MAGVIT2/modules/vqvae/hierarchical/base.py) — only
`GaussianSQQuantizer` made it required.

1. **New `modules/vqvae/hierarchical/fsq.py`** — `FSQLayerQuantizer(LayerQuantizer)`:
   - `__init__(levels, in_channels)`: `self.dim_dict = len(levels)`,
     `self.size_dict = prod(levels)` (so `_codebook_size`/`level_metadata` work
     unchanged), `register_buffer("_levels")`, `register_buffer("_basis")`,
     `in_proj/out_proj = Conv3d(1×1)` iff `in_channels != dim_dict` (mirror
     [gaussian_sq.py:82-87](src/Open_MAGVIT2/modules/vqvae/hierarchical/gaussian_sq.py)).
     `self.prior = "zero"` attribute so the topdown's `prior == "learned"` guards skip.
   - `forward(z, *, var_q_pos=None, flg_train, flg_quant_det, z_pri=None, var_q_pri=None)`:
     **ignore** `var_q_pos`/`z_pri`; project → bound+round-STE → indices → values →
     `out_proj`. Compute `perplexity` from the index histogram (same formula as SQ).
     Return `QuantizerResult(z_q, aux_loss=z.new_zeros(()), perplexity, indices,
     log_stats={active_codes, usage_fraction, size_dict})` — **omit `posterior_var`**
     (already guarded at [hier_elbo_loss.py:72](src/Open_MAGVIT2/modules/vqvae/hierarchical/hier_elbo_loss.py)
     and [video_hier_vqgan.py](src/Open_MAGVIT2/models/video_hier_vqgan.py)).
   - `decode_indices(indices)`: index → per-dim `z_i` (mixed-radix unpack) → values →
     `out_proj`. Required by `decode_from_indices` (predictor path).
   - `set_temperature`: inherit the no-op default.
2. **`quantizer_builder.py`**: dispatch `qtype in {"fsq","lfq"}`; thread a `levels`
   kwarg (and reuse the existing `lfq_sample/lfq_batch` for LFQ). Remove the
   sq-only `raise`.
3. **`video_inj_topdown.py`**: no structural change. It already calls the quantizer
   with `var_q_pos` and gated `prior` kwargs; FSQ ignores them. Confirm `dim_dict`
   plumbing accepts the small FSQ d (it does — `dim_dict` is per-layer list).
   `_has_sq_layers` already guards `log_param_q_scalar` creation, so an all-FSQ
   config allocates no SQ-only params.
4. **Loss/Lightning**: nothing. `aux_loss=0` → `kl_total=0` → `total=distortion`.
   `_update_temperature`/`_current_kl_beta` become no-ops in effect.
5. **Configs**: `shapes3d_fsq_32_S_lite_pyr.yaml` (clone of `lite_pyr`) with
   `quantizer.type: fsq`, `quantizer.levels: [[...],[...],[...]]`, no
   `temperature`/`prior`/`log_param_q_*`. Keep `decoder_source: finest_state` +
   `temporal_up: [2,2]`.

### 2.4 Gate & tests
- **Gate semantics shift**: G1 (ppl/K ≥ 0.10) passes trivially under FSQ (uniform by
  construction) → it stops being diagnostic; **G2/G3/G4 become the real signal**
  (persistence/topology/recon), which is what you actually want to measure. Add a note
  to `token_quality_report.py` that G1 is informational for FSQ configs.
- **Tests**: `tests/test_fsq_quantizer.py` — index↔code↔value round-trip exactness;
  STE gradient flows to `in_proj` (not to the rounding); `aux_loss == 0`;
  `decode_indices` matches forward's `z_q`; `size_dict == prod(levels)`;
  `level_metadata` reports correct K; e2e a 2-epoch CPU smoke
  (`shapes3d_fsq_32_S_lite_pyr_smoke.yaml`).

### 2.5 Risks
- **Low FSQ d compresses the 16-ch tap hard** (e.g. 16→4). If recon regresses vs SQ,
  raise d with more low-level dims (`[5,5,5,5,5,5]`=15625, d=6) or widen the tap. The
  match-K levels above keep d=4; the smoke recon vs Run-2 SQ recon is the check.
- **No rate term** → you lose per-layer rate accounting. Frame it as the explicit
  SQ-VAE-2-vs-FSQ ablation (a *stronger* dissertation story than one quantizer).

---

## 3. Path 3 — predictor modernization (strangler rewrite)

Contracts to preserve verbatim (the public API; checkpoints/configs/W&B keys):
`PreparedBatch`, `PredictorOutput`, `PyramidSchedule`, metric keys in
[metrics.py](src/Open_MAGVIT2/modules/predictor/metrics.py), and the
`VideoHierPredictorModel.__init__` signature.

### Phase 3.0 — hygiene (behavior-preserving, do first)
- Remove dead scaffolding from [video_hier_vqgan.py](src/Open_MAGVIT2/models/video_hier_vqgan.py):
  GAN/discriminator/perceptual branches (`use_gan` false everywhere; ~60 untested
  lines in `training_step`) and the inherited VAR LR-annealing block
  (`wp/wp0/wpe/sche_type/max_iter/wp_iter/resume_lr` + `lr_annealing`, run with
  `scheduler_type:"None"`). ~200 lines, every survivor exercised.
- Replace control-flow `assert`s in `orchestrator.py`/`schedule.py` with `ValueError`
  (asserts vanish under `python -O`; these validate inputs).
- Drop the `val/*_parallel` legacy aliases (the one-release grace is up).

### Phase 3.1 — structural refactor (behavior-preserving; defaults reproduce old numbers)
The enabler for everything after. Verified by the existing predictor tests passing
unchanged + a parallel-mode CE-equality check vs the pre-refactor commit.
- **`RolloutPolicy` strategy objects** replacing the mode-branching in
  `_forward_envelope` ([orchestrator.py:121-161](src/Open_MAGVIT2/modules/predictor/orchestrator.py)):
  `TeacherForced`, `StreamCarry`, `Autoregressive` (and later `ScheduledSampling`).
  Each answers exactly two questions — *what is the context for shift (s,k)?* and
  *what masks apply?* The envelope loop stays single and shared. This is the
  precondition for Phase 3.4.
- **`Shift` dataclass** (`stage, k, parent_key, is_reinit`) replacing `(s,k)` tuples
  across orchestrator/schedule/masks/metrics.
- **Enums** for `parent_mode`, `commit_mode`, `shift_ce_weights`, `attention_type`,
  `parallel_mode` (currently validated at five different layers, or not at all).
- **One frozen `PredictorConfig`** validated once at model init, replacing the
  `temporal_windows` value threaded separately to orchestrator/masks/batch_prep/model
  ([config.py](src/Open_MAGVIT2/modules/predictor/config.py)).
- **Unify cross-conditioning**: the dual_stream/fused_hard branch is duplicated in
  [factorized_layer.py:78-93](src/Open_MAGVIT2/modules/predictor/attention/factorized_layer.py)
  and `reversible_block.py`. Extract a `CrossConditioning` module — this also creates
  the seam for the `factorized_rev` wrapper (true mid/fine reversibility, needed if
  reversibility is a headline claim).

### Phase 3.2 — RoPE + attention stability (behavior-changing; verify parallel-CE ≤ baseline)
- **Factorized 3D RoPE** replacing the per-stage absolute `spatial_pos`/`temporal_pos`
  Parameters ([stage.py:61-62](src/Open_MAGVIT2/modules/predictor/stage.py)). Separate
  rotary bands for t/h/w (CogVideoX/Cosmos recipe), applied to q,k inside attention.
  **This deletes the untrained-positional-slot failure mode as a class** (root of
  finding #6, the val-AR-CE divergence) instead of patching it, and makes rollouts
  length-generalizing. `shift_embed` stays as a small additive shift-id.
- **Stability kit** (each a known free win): QK-norm (RMSNorm on q,k), pre-RMSNorm,
  SwiGLU MLP, and **bf16** instead of fp16-mixed (no loss-scale machinery; the forced
  fp32 CE islands mostly become unnecessary).

### Phase 3.3 — attention efficiency (behavior-preserving math; speed/memory only)
- **FlexAttention** (torch ≥ 2.5) with `mask_mod` predicates replacing the
  materialized boolean masks in [masks.py](src/Open_MAGVIT2/modules/predictor/masks.py):
  self = frame-causal + temporal window; cross = RF-overlap + spatial window. Deletes
  `build_all_masks`/`PredictorMaskCache`/`sanitize_cross_attn_mask` and the slow-path
  MHA fallback that today catches **every** non-pure-causal mask
  ([utils.py:31](src/Open_MAGVIT2/modules/predictor/attention/utils.py) only routes
  pure-causal to SDPA). Fallback: SDPA + additive float mask if Flex unavailable.
- **KV-cache** for AR re-entry — `rolling_ctx` currently re-attends the full buffer
  each shift, O(K²) ([orchestrator.py:187-190](src/Open_MAGVIT2/modules/predictor/orchestrator.py));
  matters the moment Phase 3.4 extends horizons.
- **Pre-tokenize the dataset** per gated tokenizer ckpt (the frozen 3D-encoder forward
  dominates step time). Also decouples the predictor from VAE code entirely — the
  Path-4 escape hatch and the cleanest place the SQ↔FSQ swap becomes invisible to the
  predictor (it only ever sees indices + grid shapes + K).

### Phase 3.4 — train what you test (the actual quality fix; this is "Path C")
- **Dense causal supervision**: supervise every t→t+1 transition in the predicted
  region, not just `target_token_index` per shift (~5–8× signal/step; trains the
  temporal positions/RoPE phases that were dead).
- **Scheduled sampling**: a fourth `RolloutPolicy`; teacher-force from GT with prob `p`
  annealed 1→0.5, else re-embed the model's own committed token. Trains the finest AR
  re-entry path so `val/loss_ce_ar` stops the 22→35 divergence.
- Success metric: `val/loss_ce_ar` tracks `val/*_tf` (gap closes); coarse CE breaks
  below the ~3.5 marginal-entropy wall (only possible if the FSQ/pyramid tokenizer
  made coarse predictable — couples back to Path 2).

### Phase 3.5 — MaskGIT within-frame decoding (optional; operationalizes pillar #2)
Confidence-ordered iterative refinement (4–8 steps) as an alternative `commit` path in
`_commit_tokens`. Fixes within-frame conditional independence (irrelevant for
deterministic cubes, required for multimodal/real data) and makes the per-token
**entropy/confidence map drive the decode order** — a clean thesis argument.

### Phase 3.6 — VAR next-scale ablation (optional research bet)
Condition stage `s` on the concatenated upsampled token maps of *all* coarser stages
and predict the whole scale at once ([VAR, NeurIPS 2024](https://arxiv.org/abs/2404.02905),
extended in time). A strong baseline against the stream-carry design that locates the
work in the hottest AR-vision lineage. Only after 3.4.

---

## 4. Combined sequencing (with gates between every stage)

```
[done] Run 2 (SQ pyramid) ──► gate: L3 revived, coarse/mid STARVED (§0.1)
                                                          │
                                                          ▼
  Run 2c: lite_pyr + progressive_coding:true ──► gate (layer balance + volatility)
                                                          │  must pass G1/G3 before FSQ
                                                          ▼
  Phase 3.0 hygiene + COMMIT branch ──► Phase 3.1 refactor (tests green)
                                                          │
                          ┌───────────────────────────────┘
                          ▼
  Path 2: FSQ quantizer ──► Run 2-FSQ (fsq pyramid) ──► gate (G2/G3/G4)
                          │   SQ-vs-FSQ ablation recorded; pick the gated tokenizer
                          ▼
  Pre-tokenize gated tokenizer (3.3)  ──►  FREEZE tokenizer
                          ▼
  Phase 3.2 RoPE+stability ─► 3.3 FlexAttn+KV ─► verify parallel-CE not regressed
                          ▼
  Phase 3.4 train-what-you-test ──► predictor val_ce_ar closes gap  ◄── main result
                          ▼
  (optional) 3.5 MaskGIT ──► 3.6 VAR ablation
```

Rationale for this order:
- **Run 2 before any refactor** — cheap (~1 GPU-day), validates the topology fix with
  the known quantizer, and is the ablation anchor. Don't entangle it with new code.
- **Hygiene + commit before refactor** — the branch is still uncommitted with two GPU
  runs riding on it; this is the single biggest operational risk in the project.
- **Refactor (3.1) before FSQ** — a clean quantizer seam + test harness makes the FSQ
  swap a 1-file add rather than surgery; but FSQ itself is independent of the predictor
  refactor, so 3.1 and Path 2 can proceed in parallel if desired (they touch disjoint
  trees: `modules/predictor/*` vs `modules/vqvae/hierarchical/*`).
- **FSQ + gate before predictor ML (3.2+)** — predictor changes must train against a
  frozen, gated tokenizer; never debug both at once.
- **3.2/3.3 (verifiable, behavior-preserving-ish) before 3.4 (the real change)** — so
  a 3.4 regression is attributable to the training objective, not to RoPE/attention.

---

## 5. Risk register & rollback

| Risk | Trigger | Mitigation / rollback |
|---|---|---|
| FSQ recon worse than SQ | smoke recon > Run-2 SQ recon | raise FSQ `d` (more low-level dims) or widen tap; else keep SQ-pyramid + `learned_chain` prior for L3 |
| FSQ G2/G3 still fail | gate after Run-2-FSQ | it's a *topology* result then, not a quantizer one → finest_state weighting / per-stage recon loss; FSQ at least removes G1 from the equation |
| RoPE regresses parallel-CE | Phase 3.2 verify step | RoPE behind a config flag; fall back to absolute PEs (kept one release) |
| FlexAttention unavailable / numerics differ | import error or CE drift | SDPA+additive-mask fallback path (mandatory, not optional) |
| Scheduled sampling destabilizes | val CE diverges in 3.4 | anneal `p` slower (1→0.7), shorter horizon first, dense-supervision-only ablation |
| rev_back_prop incorrectness | before any published reversible result | the gradcheck-vs-full-graph-with-parent-grads test (backlog) is **blocking**; loud warning on enable until it passes |
| Refactor changes numbers | 3.1 contract tests | defaults must reproduce pre-refactor CE bit-for-bit; if not, the refactor is wrong, revert |

---

## 6. On a from-scratch rewrite (Path 4) — not now
Most of what a rewrite buys is captured cheaper by §2+§3 behind the existing
interfaces, and a rewrite resets the verified assets (gate, ELBO tests, shape-audit
guards, RSS-safe test harness) that are the most valuable things in the repo. The
trigger that flips this: committing to real video (UCF-101/K600/SSv2) at ≥128px — then
the Lightning-CLI scaffolding and 32px test geometry become drag, and Path 4 becomes
"port the verified pieces into a clean trainer," done as a strangler over the
pre-tokenized-index interface (§3.3), not a greenfield restart.
