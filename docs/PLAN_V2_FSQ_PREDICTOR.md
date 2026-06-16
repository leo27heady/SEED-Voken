# Next-Version Master Plan — FSQ Tokenizer (Path 2) + Predictor Modernization (Path 3)

> **Implementation-ready.** Authored 2026-06-14 after Run 2c (`lle1hzce`) + the verified
> t=1 trace + a 6-agent design/critique pass. Supersedes the §4 sketch in
> [PLAN_FSQ_PREDICTOR_COMBINED.md](PLAN_FSQ_PREDICTOR_COMBINED.md) (keep that doc for the
> run-history narrative §0.0–§0.2). Companion: [PROJECT_MEMORY.md](PROJECT_MEMORY.md).
> Branch `path-a/elbo-pyramid`. All paths are repo-root-relative.
>
> Goal: an **up-to-date, training-stable, speed/perf-efficient** next version. Two tracks
> that never change in the same run: **Path 2** (quantizer family, frozen+gated first) and
> **Path 3** (predictor, strangler rewrite behind stable contracts).

---

## 0. State going in

- **Run 2c proved** `progressive_coding` fixes the coarse/mid **starvation** (all 3 layers
  anneal, codebooks fill, clean coarse→fine decomposition) **but** exposed a ~**800×
  soft-vs-hard (Gumbel-train vs argmax-eval) quantization gap** on SQ-VAE (val argmax
  recon 0.135 vs train soft 0.00017; *not* EMA — LitEma decay 0.999).
- **Decision:** next tokenizer = **FSQ + progressive coding + causal temporal-align fix**.
  FSQ's straight-through rounding makes train == eval (gap gone by construction);
  progressive gives layer balance. They compose.
- **t=1 fuzzy frame: root cause verified empirically** (§1). Cosmetic + mild
  training-target artifact in the progressive-L1 row; **full recon is unaffected**.

---

## 0.5 IMPLEMENTATION STATUS (updated 2026-06-15)

- **Path 2 (FSQ + temporal-align): DONE** (commit `6c6e5c7`). FSQ quantizer (§4),
  builder dispatch + `_levels_map` (H1), `temporal_align_mode: causal`
  `_upsample_state_to_finest` (§1.3), gate G1-informational note (M3), tests. Runs:
  `bllr9fzo` (match-K, gate-ready, recon 0.00019) and `dr8vrkt8` (rightsized
  512/1024/256, recon 0.000255 @ ep18, in progress) — the optional size ablation.
- **Path 3.0/3.1 (predictor hygiene + structural refactor): DONE, behavior-preserving**
  (uncommitted on `refactor/v2-predictor-and-fsq`, tag `pre-path3-baseline`). Landed:
  `enums.py` (§5.3), `shift.py` + `envelope_shifts` (§5.2), `PredictorConfig`/
  `StageConfig` (§5.4), `CrossConditioning` + state-dict remap (§5.5/H4),
  `RolloutPolicy` collapse of the envelope loop (§5.1/B4). Golden CE bit-equality
  (`test_golden_refactor.py` + `_golden/` fixture) holds across all 4 modes through
  every step; 114/114 predictor tests pass. See PROJECT_MEMORY §0.1 for details.
- **Corrections to THIS plan, found in implementation** (full reasoning in
  PROJECT_MEMORY §0.1):
  - **§5.6 predictor `_lr_annealing` removal — DO NOT DO** (it is *live* in
    `training_step`, not dead; removal changes the LR schedule). "asserts→ValueError"
    is moot (no asserts in that file). KEPT.
  - **§5.6 `video_hier_vqgan.py` dead-code (M4) — DEFERRED** to after the encoder run
    (live tokenizer module; remove on the P0/tokenizer track to avoid resume risk).
  - **§5.5 "reversible_block.py:50-67"** is `set_parent`, not param defs; duplication
    is indirect via F/G (`ParallelPredictorResidual`).
  - **§5.7 ordering**: record the golden FIRST (at baseline), assert after each step.
  - **§5.1 interface** simplified — orchestrator keeps generic mechanics; policy owns
    only context-source / `is_reinit` / `on_output` / `final_ar_len`.
- **NOT started (gated by GATE-T + frozen+pre-tokenized tokenizer):** §6 RoPE,
  §7 Flex/KV/pre-tokenize/bf16, §8 dense supervision + scheduled sampling.

---

## 1. The t=1 fuzzy-frame artifact — verified root cause & fix

### 1.1 What it is (evidence)
Traced with the trained ckpt (`epoch=27-step=87500`, saved grid
`docs/wandb_analysis/l1_t1_trace.png`):

- `forward_progressive` renders each partial by trilinearly upsampling the coarse state
  to the finest grid: `align_spatial(p, finest_thw)` =
  `F.interpolate(mode="trilinear", align_corners=False)`
  ([video_inj_topdown.py:25-30](src/Open_MAGVIT2/modules/vqvae/hierarchical/video_inj_topdown.py), used at `:640`).
- Measured 3→9 temporal weights: out **t0 = 1.0·c0**, out **t1 = 1.0·c0** (identical),
  t2 = .67c0+.33c1, t4 = c1, t7/t8 = c2. Aligned latent **|f0−f1| = 1.3e-9** (bit-identical).
- Coarse frame **c0 is the causal-boundary frame** (encoder RF = input frame 0 only).
  So out-t1 shows frame-0 content where GT frame 1 has already moved; the causal decoder
  fed a duplicated frame adds a boundary smear.
- Decoded per-frame MSE: **t0 = 0.0087, t1 = 0.0761 (8.7×)**, t2…t8 ≈ 0.008 — despite
  identical latent at t0/t1. ⇒ the **uniform trilinear** mapping is inconsistent with the
  encoder's **non-uniform causal** downsampling. Full recon (all layers) t1 = 0.0001.
- Alternative interpolations are worse (`align_corners=True` blurs all frames; `nearest`
  wrecks middle frames) → the fix must be causal-aware, not a different interp mode.

### 1.2 Scope (important)
The **full reconstruction is already correct** on every path that matters:
`forward` and `decode_from_indices` (the path the **gate's G4 scores**) build the finest
`z_state` via the **learned causal `Upsampler`** (`block.spatial_up`,
[video_inj_topdown.py:666](src/Open_MAGVIT2/modules/vqvae/hierarchical/video_inj_topdown.py)),
not trilinear. The trilinear bug exists **only** in `forward_progressive`'s post-hoc
partial alignment, which feeds (a) the progressive viz rows and (b) the averaged
progressive distortion in `compute_hier_elbo_loss`. So the fix changes the L1/L2 partial
rendering + their loss contribution; it does **not** change G4, the encoder indices, or
the predictor's frozen tokens.

### 1.3 Fix (Option A — chosen)
Replace the trilinear partial alignment with a **replay of the learned `Upsampler`
chain** so partials follow the exact causal 3→5→9 map the forward pass uses.

```python
# video_inj_topdown.py — new method on SQVAE2TopDown
def _upsample_state_to_finest(self, z_state, from_layer):
    """Replay learned u2 Upsamplers from blocks[from_layer+1..end] on a partial
    z_state → finest grid with the SAME causal temporal map (T->2T-1, frame-drop)
    the forward chain used. NOT trilinear. Identity when from_layer == last."""
    z = z_state
    for j in range(from_layer + 1, self.num_layers):
        block = self.blocks[j]
        if isinstance(block, InjSQBlock):
            z = block.spatial_up(z)
    return z
```
In `forward_progressive` (`:632-641`) keep `partial.append(z_state.clone())` at **native
grid**, then replace the trilinear loop:
```python
if self.decoder_source == "finest_state":
    finest_thw = partial[-1].shape[2:]
    partial = [p if p.shape[2:] == finest_thw
               else self._upsample_state_to_finest(p, from_layer=i)
               for i, p in enumerate(partial)]
    return partial[-1], partial
```
`compute_hier_elbo_loss` and `decode_progressive` are unchanged. `decoder_source: latent`
(non-pyramid) path untouched.

Option B (native-T multi-scale Laplacian loss: decode each partial at native T vs
causally-downsampled GT) is the *more principled* objective but changes loss scale and
needs a viz-time remap — keep documented as a **later ablation**, not the default.

### 1.4 Tests + guard (incorporates review item B1)
- `tests/test_progressive_temporal_align.py`:
  1. craft acts with c0≠c1; assert L1 partial finest frames t0≠t1 (`abs().max() > 1e-4`)
     — the bug made them bit-equal.
  2. assert `partial[-1]` (full) `allclose` the `forward` finest `z_state` — fix must not
     touch the final stage.
  3. **loss-scale guard:** total progressive distortion shifts < ~10% vs trilinear on a
     fixed batch (coarse stage gets easier; re-baseline G-thresholds only if intentional).
- **B1 guard:** `assert decode_from_indices(forward.indices) full-recon ≈ forward full-recon`
  (the gate path == the training full path) — proves G4 measures the trained recon.
- `temporal_align_mode: causal|trilinear` config flag, **default `trilinear`** (old
  behavior) so the change is opt-in and old ckpts/tests are byte-identical; the FSQ run
  config sets `causal`.

---

## 2. Invariants (non-negotiable)

1. **Tokenizer and predictor never change in the same run.** VQ is trained → gated →
   **frozen** → pre-tokenized; predictor ML trains on frozen indices.
2. Behavior-preserving refactors land **before** behavior-changing ML; defaults reproduce
   old numbers **bit-for-bit**, proven by a golden test.
3. Every phase: one entry criterion, **one** success metric, a green-test bar, a rollback.
4. Dissertation pillars live in the **predictor** (categorical prediction + entropy maps,
   hierarchy, in-stage reversibility) — quantizer-family-agnostic. SQ↔FSQ is an ablation.
5. Test suite peak RSS ≤ 12 GiB (conftest watchdog); never full-attention on >2.5k-token
   grids in tests; new compile/Flex paths **off by default in tests**.

---

## 3. Sequencing DAG (with review corrections applied)

```
TOKENIZER TRACK (strict)                                   PREDICTOR TRACK
P0  commit + hygiene ───────────────────────────────►  (P3.0/P3.1 may run in PARALLEL,
 │   (incl. video_hier_vqgan.py dead-code, see M4)      gated by predictor tests, NOT by
 ▼                                                      the tokenizer)
P1  FSQ quantizer + tests  ⟂ P3.0 hygiene (predictor/*) │
 ▼                                                      P3.0 predictor hygiene
P2  causal temporal-align fix (all assembly paths, B1)   ▼
 ▼                                                      P3.1 structural refactor
RUN-FSQ  (match-K AND high-d-fine in ONE batch, B2)      (RolloutPolicy, Shift, enums,
 ▼                                                       PredictorConfig, CrossConditioning)
[GATE-T] G2/G3/G4  (G1 informational)                    │  golden CE bit-equality
 │ PASS                                                  ▼  ── all the above gated by tests,
 ▼                                                       independent of GATE-T ──
FREEZE → pre-tokenize ───────────────────────────────► P3.2 RoPE + stability     [GATE-P1]
                                                         ▼   parallel-CE ≤ baseline
                                                        P3.3 efficiency (Flex/KV/bf16) [GATE-P2]
                                                         ▼   speed↑, CE unchanged
                                                        P3.4a dense supervision  [GATE-P3a]
                                                         ▼   _lasttok CE tracks baseline
                                                        P3.4b scheduled sampling [GATE-P3b]
                                                         ▼   val_ce_ar gap closes ◄ MAIN RESULT
                                                        (opt) P3.5 MaskGIT → P3.6 VAR
```

**Hard interlock:** nothing in P3.2+ starts until GATE-T passes and the tokenizer is
frozen+pre-tokenized. P3.0/P3.1 (pure `predictor/*` refactor) may overlap the tokenizer
track. **Correction M4:** the `video_hier_vqgan.py` (tokenizer Lightning module)
dead-code removal moves into **P0/tokenizer track**, NOT the parallel predictor-hygiene
phase, so a predictor commit can't destabilize a live tokenizer run.

---

## 4. Path 2 — FSQ quantizer + temporal fix

### 4.1 `FSQLayerQuantizer` (new `modules/vqvae/hierarchical/fsq.py`)
Math (Mentzer 2023): per channel `i`, half-range `m_i=(L_i−1)/2`;
`bounded = m_i·tanh(h_i)`; `ẑ = round_ste(bounded)`; `value = ẑ/m_i ∈ [−1,1]`;
`code c_i = round(bounded)+m_i ∈ {0..L_i−1}`; `index = Σ c_i·basis_i`,
`basis = cumprod([1,L_1,…])`. `aux_loss = 0`, no temperature/prior/posterior_var.

Key signatures (full body in workflow output / to be written):
```python
def round_ste(z): return z + (torch.round(z) - z).detach()

class FSQLayerQuantizer(LayerQuantizer):
    def __init__(self, levels: List[int], in_channels=None):
        self.dim_dict = len(levels); self.size_dict = prod(levels)
        self.prior = "zero"               # topdown learned-prior guards skip
        # register _levels, _half=(L-1)/2, _basis; Conv3d in/out_proj iff in_ch!=d
    def forward(self, z, *, var_q_pos=None, flg_train=True, flg_quant_det=False,
                z_pri=None, var_q_pri=None) -> QuantizerResult:  # all extras ignored
        # -> QuantizerResult(z_q, aux_loss=z.new_zeros(()), perplexity(from hist),
        #    indices, log_stats={active_codes,usage_fraction,size_dict})  # NO posterior_var
    def decode_indices(self, indices) -> z_q   # exact inverse, used by decode_from_indices
```
Why integration is a no-op elsewhere: `var_q_pos` already `Optional`
([base.py:26](src/Open_MAGVIT2/modules/vqvae/hierarchical/base.py)); `prior=="zero"`
skips the `z_pri` kwargs in `InjSQBlock`/`SQResSQBlock`; `_has_sq_layers=False` keeps
`log_param_q_scalar` a buffer (no SQ params in the optimizer); `aux_loss=0` →
`kl_total=0` → `total=distortion`; `posterior_var` omission is guarded at
[hier_elbo_loss.py:72](src/Open_MAGVIT2/modules/vqvae/hierarchical/hier_elbo_loss.py);
`size_dict=prod(levels)` keeps `_codebook_size`/`level_metadata` working.

### 4.2 Builder dispatch (`quantizer_builder.py`) — **correction H1**
Add `fsq`/`lfq` dispatch and thread a per-layer `levels`. In the topdown, build
`_levels_for(i)` that returns `None` for non-fsq layers and validates
`len(levels_cfg) == #fsq_layers` (not `num_layers`) so a mixed `per_layer: [sq,fsq,fsq]`
config can't `IndexError`. Add a mixed-dispatch test.

### 4.3 Levels — **correction B2: run two configs in the FIRST batch**
The soft/hard gap is gone *by construction* (a **unit-test** assertion:
`forward(flg_train=True) ≈ forward(flg_quant_det=True)`), but **G4 recon passing is a
separate, empirical bet** on the low-d bottleneck (16-ch tap → d=4). Do **not** conflate
them. Run **match-K and high-d-fine in the same first GPU batch**:

| layer | match-K (ablation continuity) | high-d-fine (recon insurance) |
|---|---|---|
| L1 coarse 3×4×4 | `[8,8,8,8]` (4096, d4) | `[8,8,8]` (512, d3) |
| L2 mid 5×8×8 | `[8,8,8,4]` (2048, d4) | `[8,8,6,6]` (2304, d4) |
| L3 fine 9×16×16 | `[8,8,4,4]` (1024, d4) | `[8,8,8,5,5]` (12800, d5) |

If match-K G4 regresses vs SQ (0.00078), high-d-fine is the answer; else widen the tap.

### 4.4 LFQ (deferred, behind dispatch)
Binary FSQ (`levels=[2]*log2K`) + MAGVIT-2 entropy penalty. Documented alternative; **not**
run first (its entropy weight is exactly the tunable knob the project is deleting). Note
the `_codebook_size` `hasattr(quantizer,"lfq")` branch
([video_inj_topdown.py:74](src/Open_MAGVIT2/modules/vqvae/hierarchical/video_inj_topdown.py))
expects `.lfq.codebook_size`; give LFQ a `size_dict` so the first branch wins, or don't
ship until tested.

### 4.5 Config, gate, tests
- `configs/.../shapes3d_fsq_32_S_lite_pyr.yaml`: clone of `lite_pyr`; `quantizer.type: fsq`,
  `levels: [...]`, remove `temperature/prior/log_param_q_*/temporal_kl_weight`; keep
  `progressive_coding: true`, `decoder_source: finest_state`, `temporal_up: [2,2]`; add
  `hierarchy.temporal_align_mode: causal`. Plus `_smoke.yaml` (CPU) and `_rightsized.yaml`.
- **Gate:** G1 (ppl/K) is **informational** under FSQ (≈uniform by construction — note
  it's *approximately* uniform: `torch.round` half-to-even biases extreme bins, **M3**);
  **G2/G3/G4 are the real signal.** G4 now equals train recon (no gap). One-line note in
  `token_quality_report.py` when `quantizer.type ∈ {fsq,lfq}`.
- `tests/test_fsq_quantizer.py`: round-trip exactness; STE grad to `in_proj` (not through
  round); `aux_loss==0`; `decode_indices==forward.z_q`; `size_dict==prod`; no
  `posterior_var`; **train==eval recon** (`allclose`); index range; builder dispatch incl.
  mixed; CPU e2e smoke.

---

## 5. Path 3.0/3.1 — hygiene + structural refactor (behavior-preserving)

**Tag the baseline:** `git tag pre-path3-baseline <HEAD>`. Hard constraint: no
parameter-owning module is renamed → no `state_dict` key changes (except the deliberate
`CrossConditioning` remap, handled below).

### 5.1 RolloutPolicy strategy objects (`predictor/rollout.py`) — **correction B4**
There is exactly **one** `RolloutPolicy` hierarchy (the structural seam). The scheduled-
sampling *schedule* (p-anneal/mix) is **not** a separate class — it becomes the body of
`ScheduledSampling(Autoregressive)` in P3.4b. **Delete the idea of a second
`rollout_policy.py`.**

Interface: a stateful, per-forward object (not an `nn.Module`) answering two questions +
two hooks:
```python
class RolloutPolicy:
    def context_for_shift(self, shift) -> ShiftContext: ...   # embeds / stream state
    def masks_for_shift(self, shift, ctx, parent) -> ShiftMasks: ...  # shared default
    def record(self, shift, out, ctx): ...        # stream-carry bookkeeping
    def commit_output(self, shift, out, q_logits): ...  # AR rolling-buffer hook (no-op default)
    def final_ar_len(self, finest_s) -> Optional[int]: ...
# subclasses: TeacherForced, Oracle(parallel), Autoregressive, ScheduledSampling (P3.4b)
```
The envelope loop collapses to one shared `_run_envelope(batch, policy)`; the three public
entrypoints (`forward_train/forward_parallel/forward_autoregressive`) become 1-line
factory calls. `make_policy(mode, batch)` is the only mode-string switch left.

### 5.2 `Shift` dataclass (`predictor/shift.py`)
Frozen `Shift{stage,k,parent_stage,parent_k,is_reinit}` with `is_root/is_first/prev_k/
as_tuple()`. **Contract bridge:** internally everything is `Shift`; `_keys_to_tuples`
converts to `Dict[(s,k),…]` at the orchestrator boundary so `PredictorOutput.logits/
ce_breakdown` and metric keys stay `Tuple[int,int]`. `schedule.envelope_shifts()` wraps
`envelope_order()` (kept byte-identical → golden snapshot test passes).

### 5.3 Enums (`predictor/enums.py`)
`str,Enum` mixins for `ParentMode/CommitMode/ShiftCEWeights/AttentionType/ParallelMode`
(== raw strings, so YAML still parses). Validation consolidates into
`PredictorConfig.__post_init__`; `Enum(value)` raises `ValueError` on bad strings (also
satisfies asserts→exceptions for these paths).

### 5.4 Frozen `PredictorConfig` (`config.py`)
One typed `PredictorConfig` (+`StageConfig`) validated once at model init, replacing the
`temporal_windows` value threaded to 4 consumers. `EnvelopeOrchestrator.__init__` keeps
the **old kwarg signature** for back-compat (so `stack_factory.py` + ~15 tests run
unmodified) and gains an optional `config=`.

### 5.5 CrossConditioning (`predictor/cross_conditioning.py`) — **correction H4**
Unify the duplicated dual_stream/fused_hard cross-attn (factorized_layer.py:78-93 +
reversible_block.py:50-67 + parallel_residual.py). **State-dict risk:** attribute names
change (`cross_q_f`→`cross_f.q`, …) → add `_load_from_state_dict` remap hooks. **Required
CPU test** `test_cross_cond_remap_cpu` (random-weight stack → save → remap-load →
bit-equality) so the remap is covered in CI, not only on the GPU box. Also creates the
seam for a future `factorized_rev` (true mid/fine reversibility).

### 5.6 Dead-code removal
- `video_hier_vqgan.py` (**in P0/tokenizer track per M4**): GAN/discriminator/perceptual
  branch, VAR LR-annealing (`wp/wp0/wpe/sche_type/max_iter/lr_annealing`). **Keep the
  kwargs in the signature** (accepted-but-ignored) so YAML `init_args` still instantiate;
  delete only the bodies. Grep-confirm each symbol unused first.
- `video_hier_predictor.py`: same LR-anneal removal; asserts→`ValueError`.
- `metrics.py`: drop `val/*_parallel` aliases. **Correction L2:** keep them one more
  release until the SQ↔FSQ W&B ablation is recorded (≈6 lines), then remove.

### 5.7 Verification (the core proof)
`tests/predictor/test_golden_refactor.py`: capture `forward_train/parallel/autoregressive`
CE + per-`(s,k)` `ce_breakdown` at `pre-path3-baseline` (fixed seed, `lite_stack`); assert
bit-for-bit after each refactor step. Migration order (each keeps suite green): Shift →
enums → record goldens → CrossConditioning(+remap test) → RolloutPolicy → PredictorConfig
→ dead-code. One commit per step; `git revert` any single step.

---

## 6. Path 3.2 — RoPE + stability stack

### 6.1 Factorized 3D RoPE (`predictor/attention/rope.py`)
Split `head_dim` into even per-axis budgets `D_t+D_h+D_w` (time gets the largest, e.g.
`(4,2,2)` for head_dim 8); rotate q,k by per-axis angles `θ_{a,i}=base_a^(-2i/D_a)`
applied to interleaved pairs. Relative ⇒ (1) no untrained absolute slots (kills the
`val_ce_ar` 22→35 divergence's positional half — finding #6), (2) length generalization
for AR rollout past `t_total`. `Rotary3D` caches cos/sin and **grows on demand** so AR
`t_len>t_total` works.

### 6.2 Integration point — **correction H2(a) (the dangerous one)**
RoPE applies to q,k after in-projection, before SDPA — which means rewriting
`mha_self_attention` to manually unpack q/k/v and adding a **non-causal SDPA path** (also
deletes the slow materialized-MHA fallback). This changes **mask polarity** for all
callers (`~mask`→allow-mask). **This refactor is behavior-preserving with RoPE OFF and
MUST be its own DAG step with a green bar BEFORE RoPE:** `test_mha_polarity_equivalence`
(new path with RoPE off == old `nn.MultiheadAttention(attn_mask=~mask)` within 1e-6).
Without this, a polarity off-by-one silently changes the absolute-PE path and gets
misattributed to RoPE.

### 6.3 head_dim floor — **correction H2(b)**
`dim=32,n_heads=4→head_dim=8` gives spatial axes only 1 rotary band over a 16-wide grid
(nearly positionless). **Run the first RoPE experiment at `head_dim≥16`** (bump fine-stage
`dim` to 64) so the no-regression test isn't confounded by under-resolved spatial RoPE.
Axis split is a config.

### 6.4 Stability/precision kit
QK-RMSNorm (norm **then** rotate), pre-RMSNorm blocks, SwiGLU MLP — each behind a flag
(`qk_norm`, `norm_type`, `mlp_type`), default off/legacy so resume is byte-identical until
a fresh-train gate pass. bf16 — see **correction H5** in §7.4 (must NOT wrap the VAE).

### 6.5 Config + verify
`predictor.pos_encoding: absolute|rope` (absolute kept one release). `test_rope.py`:
relative-invariance (score depends only on Δpos), length-generalization (cache grows, no
NaN), axis-budget. **Success metric (GATE-P1):** `val/*_tf` parallel-CE ≤ pre-RoPE
baseline on frozen FSQ tokens.

---

## 7. Path 3.3 — efficiency

### 7.1 FlexAttention + mandatory SDPA fallback (`predictor/attention/flex.py`)
Replace materialized boolean masks (`PyramidMaskBuilder`/`PredictorMaskCache`/
`sanitize_cross_attn_mask`) with `mask_mod` predicates:
- self: `frame-causal & (0 ≤ qt−kt < temporal_window)`;
- cross: `RF-interval-overlap & spatial-window` (RF groups are contiguous `range(...)`
  → encode as `[lo,hi]`; empty RF → empty interval → never attend, subsuming
  `sanitize`'s empty-row case; keep only the real-query-zero-keys row-guard).

`BlockMask` cache keyed by `(kind,s,k,Q,KV,device)` (covers AR's shrinking lengths without
slicing). **Mandatory fallback** (not optional): when Flex unavailable, build an
**additive float `-inf` mask and use fused SDPA** — strictly faster than today's
materialized-MHA fallback even without Flex. `test_flex_equivalence.py`: predicate↔legacy
mask bit-equality (pins the golden-hash invariant), and Flex↔SDPA `allclose` (GPU-gated).

**Correction M1 (Windows reality):** FlexAttention needs Triton/inductor, historically
flaky on native Windows. Before P3.3, run a 10-line `torch.compile(flex_attention)` smoke
on the actual GPU; **if it falls back, re-scope GATE-P2 to "SDPA-additive ≥ 1.5× over
materialized-MHA"** and treat Flex as opt-in/Linux-only. Correctness holds either way.

### 7.2 KV-cache for AR re-entry (`predictor/attention/kv_cache.py`)
Replace the O(K²) full-buffer re-attend ([orchestrator.py:187-190](src/Open_MAGVIT2/modules/predictor/orchestrator.py))
with prefill(k=0)/decode(k>0) incremental cache (pre-allocated, write-cursor, no
`torch.cat`). Two-stream (F/G) KV for the reversible path. **Correction M2:** the
equivalence test must include the **reversible block** (`test_kv_cache_equiv_reversible`),
not just full attention. **The cache's `commit_embed` parameter is the single AR-seam
contract** shared with scheduled sampling (see B3 below).

### 7.3 Pre-tokenized dataset (`data/shape_video.py`)
Cache **frozen-VAE indices** (not embeds — `codebook_proj` stays trainable, re-embeds
indices in-grad) keyed by `sha256(vae_ckpt) × vae_config_hash × dataset_hash`,
`int16` if `maxK≤32767`. `BatchPrep.schedule_from_indices(indices, stages)` skips the VAE
entirely (the Path-4 escape hatch). Built in `DataModule.prepare_data`.
**Correction L1:** under pre-tokenized data `activations/encoder_bottleneck` are `None` →
`log_images` ([video_hier_predictor.py:268-276](src/Open_MAGVIT2/models/video_hier_predictor.py))
would crash; default `log_pred_mse=False` and gate `log_images` off (or lazily re-encode a
tiny fixed viz batch). `test_token_cache.py`: `schedule_from_indices(cached)` ≡
`encode_and_schedule(vae,video)` bit-for-bit; cache key changes with ckpt bytes.

### 7.4 bf16 — **correction H5 (correctness, not just speed)**
The current `autocast(enabled=False)` wraps **all of `forward_batch`, including the frozen
VAE encode**. Running the gated fp32 VAE under bf16 changes argmax indices near decision
boundaries → the predictor would train on tokens that **don't match the gated tokenizer**,
breaking the frozen contract. **bf16 wraps ONLY the predictor stages' attention matmuls;
VAE encode stays fp32 (or is gone via pre-tokenization); CE stays fp32** (already `.float()`
at orchestrator.py:181). Assert pre-tokenized indices == fp32-VAE indices.

### 7.5 torch.compile
Compile the VAE encoder/decoder (static shapes) for the **token-cache build** + rare
logging-decode only. **Do not** compile predictor stages until `_forward_envelope` is
refactored (graph breaks from the custom-autograd reversible path). **Never compile in
tests** (`conftest.py` sets `PREDICTOR_FLEX=0`; RSS/time budget).

---

## 8. Path 3.4 — train-what-you-test (the main quality fix)

### 8.1 Split into 3.4a then 3.4b — **correction H3 (attribution)**
Dense supervision changes what per-shift CE *means* (averages over heterogeneous
transitions), so landing it with scheduled sampling makes the `val_ce_ar` result
unattributable. **Split:**
- **P3.4a — dense causal supervision:** supervise every t→t+1 transition in the
  predicted region (not just `target_token_index`), ~5–8× signal/step, **zero extra
  forward compute** (logits already computed). Add a **mandatory `_lasttok` CE key**
  (CE on the canonical target frame only) as the gate metric so it's comparable to the
  pre-dense baseline. Edits: `schedule.build_shift_supervision(dense=)`,
  `batch_prep` dense target gather, `orchestrator` (tensors just get bigger),
  `_collect_pred_indices` commits only the canonical frame. No leakage (targets are q+1,
  not attended) — assert in a test.
- **P3.4b — scheduled sampling:** `ScheduledSampling(Autoregressive)` (the §5.1 subclass),
  teacher-force prob p anneals 1→0.5, else re-embed the model's own committed token —
  trains the finest AR re-entry path. **Correction B3:** design it against the
  **post-KV-cache** orchestrator — it writes the (GT-or-pred) `commit_embed` into the
  cache (the §7.2 contract), **not** the legacy `rolling_ctx`+`torch.cat`. SS only rewires
  the finest stage's inter-shift seam; coarse/mid keep stream-carry.

### 8.2 Success metric (GATE-P3b, MAIN RESULT)
`val/loss_ce_ar` tracks `val/*_tf` (AR/TF gap < 1.5× at finest, vs ~9× today); the 22→35
divergence flattens/declines. Coarse CE can break the ~3.5 marginal-entropy wall only if
the FSQ tokenizer made coarse predictable (couples to GATE-T G2).

### 8.3 Optional research bets (after 3.4)
P3.5 MaskGIT within-frame iterative decode (operationalizes the entropy map — pillar #2);
P3.6 VAR next-scale ablation. Both alternative paths behind flags, default off.

---

## 9. Critical corrections from adversarial review (index)

| # | Severity | Correction | Where applied |
|---|---|---|---|
| B1 | BLOCKER | Verify temporal fix vs the path the gate scores; `decode_from_indices` full-recon already uses learned-up (correct) — add equality test, don't duplicate trilinear there | §1.2, §1.4 |
| B2 | BLOCKER | "gap closed" (unit-test, guaranteed) ≠ "G4 passes" (empirical low-d bet); run match-K + high-d-fine in one batch | §4.3 |
| B3 | BLOCKER | KV-cache (3.3) and scheduled sampling (3.4b) edit the same AR seam; unify via the `commit_embed` cache contract; SS designed post-KV | §7.2, §8.1 |
| B4 | BLOCKER | Two clashing `RolloutPolicy` classes → one hierarchy; SS schedule is a subclass body; delete `rollout_policy.py` | §5.1 |
| H1 | HIGH | `_levels_for(i)` must return None for non-fsq layers; validate len==#fsq layers | §4.2 |
| H2 | HIGH | mask-polarity refactor needs its own bit-equality gate before RoPE; run RoPE at head_dim≥16 | §6.2, §6.3 |
| H3 | HIGH | split dense-supervision from scheduled-sampling; gate on `_lasttok` key | §8.1 |
| H4 | HIGH | CrossConditioning state-dict remap needs a CPU equality test | §5.5 |
| H5 | HIGH | bf16 must NOT wrap the frozen VAE (corrupts gated tokens); predictor-attn only | §7.4 |
| M1 | MED | FlexAttention on Windows unverified; smoke-test, re-scope DoD to SDPA-additive if it falls back | §7.1 |
| M2 | MED | KV-cache equivalence test must cover the reversible block | §7.2 |
| M3 | MED | FSQ usage is *approximately* uniform (half-to-even rounding); keep G2/G3/G4 as real signal | §4.5 |
| M4 | MED | `video_hier_vqgan.py` dead-code removal → tokenizer track (P0), not parallel predictor phase | §3, §5.6 |
| L1 | LOW | `log_images` crashes under pre-tokenized data (no activations) — gate off / lazy re-encode | §7.3 |
| L2 | LOW | keep `_parallel` metric aliases until the SQ↔FSQ ablation is recorded | §5.6 |
| R8 | BLOCKER (publish) | `rev_back_prop` correctness unproven — gradcheck incl. parent grads blocks any reversible result | §10 |

---

## 10. Risk register

| # | Risk | Trigger | Mitigation / rollback | Blocking |
|---|---|---|---|---|
| R1 | FSQ low-d recon regression | RUN-FSQ G4 fail / smoke > 2× SQ | high-d-fine config (run in same batch, B2); widen tap; else SQ band-aid (τ→0.05 + stage-weighted progressive loss) | no |
| R2 | FSQ G2/G3 still fail | gate after RUN-FSQ | topology, not quantizer → finest_state/per-stage recon weighting | no |
| R3 | RoPE ckpt incompat | absolute ckpt won't load RoPE model | flag `pos_encoding`; tokenizer frozen first → cheap fresh predictor run | no |
| R4 | Flex unavailable/drift | import err / CE drift | mandatory SDPA-additive fallback; equivalence test; M1 re-scope | no |
| R5 | Scheduled-sampling instability | val_ce_ar diverges in 3.4b | slower p-anneal (1→0.7); shorter horizon; dense-only ablation | no |
| R6 | Test RSS > 12 GiB | conftest watchdog | shrink stack; Flex/compile off in tests; never full-attn >2.5k tokens | yes (suite) |
| R7 | Temporal fix loss-scale shift | progressive distortion >10% | `temporal_align_mode` default trilinear; opt-in; guard test | no |
| R8 | `rev_back_prop` unproven | any reversible publish | gradcheck incl. parent grads is **blocking**; loud warn on enable | yes (publish) |
| R9 | Refactor changes numbers | golden CE ≠ baseline | revert, don't patch forward | yes (phase) |
| R10 | Tokenizer+predictor change together | a config retraining both | structurally prevented (freeze+pretokenize before P3.2) | yes (process) |

---

## 11. Experiments + commands

```powershell
# venv: .\venv\Scripts\python.exe ; tests: python -m pytest tests -q

# P1/P2 smoke (CPU)
.\venv\Scripts\python.exe -m pytest tests/test_fsq_quantizer.py tests/test_progressive_temporal_align.py -q
.\venv\Scripts\python.exe main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr_smoke.yaml

# RUN-FSQ (~1 GPU-day) — match-K AND high-d-fine (B2), then gate each
.\venv\Scripts\python.exe main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr.yaml
.\venv\Scripts\python.exe main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr_rightsized.yaml
.\venv\Scripts\python.exe scripts/token_quality_report.py --config <fsq yaml> --ckpt <best> --json docs/wandb_analysis/gate_run_fsq.json

# FREEZE → pretokenize (~30 min) → predictor track
.\venv\Scripts\python.exe scripts/pretokenize_dataset.py --config <fsq yaml> --ckpt <gated best> --out ../../data/shape_tokens_fsq
.\venv\Scripts\python.exe main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr_predict.yaml
```

GATE-T decision tree: all pass + argmax≈train → freeze/pretokenize/predictor;
G4 fail → high-d-fine/widen tap; G2/G3 fail → loss weighting (topology, not quantizer);
argmax≫train → impossible under FSQ ⇒ STE/decode bug, fix don't proceed.

---

## 12. Definition of Done ("the next version")

1. Tokenizer (FSQ + progressive + causal temporal-align) GATE-T green: G2/G3/G4 pass, G1
   informational; argmax `val_ema/mse` within ~1.2× of train recon; progressive-L1 t=1 MSE
   within 2× of t=0. Gate JSON + SQ↔FSQ ablation archived.
2. Tokenizer frozen + dataset pre-tokenized; predictor step imports no VAE.
3. Predictor refactor behavior-preserving (golden CE bit-for-bit; contracts intact; old SQ
   ckpts load).
4. RoPE + stability: parallel-CE ≤ baseline; no `spatial_pos`/`temporal_pos` params.
5. Efficiency: Flex (or SDPA-additive fallback) + KV-cache; fine-stage ≥ 2× (or ≥1.5×
   fallback) faster; slow materialized-MHA path gone.
6. Main result: `val/loss_ce_ar` tracks `val/*_tf` (gap < 1.5× finest); 22→35 divergence
   closed; dense-only vs +SS ablated.
7. Tests green (≥162 + new), RSS ≤ 12 GiB, ≤ ~6 min.
8. Reversibility either not shipped or backed by the R8 gradcheck.
9. Docs: PROJECT_MEMORY §5/§7 + this plan's gate results + the dissertation ablation.
```
