# Path C Architecture Audit Report

**Date:** 2026-06-07  
**Scope:** Verify implementation vs `path_c_full_implementation_22ca5538.plan.md`  
**Test status:** 84/84 passing  
**Validation gate:** prefix stability 6.1e-6 (< 1e-5), shape audit PASS

---

## Executive summary

| Area | Verdict | Notes |
|------|---------|-------|
| Encoder/decoder v2 | **Correct** | FrameWiseGroupNorm, spatial keys, prefix-stable |
| VAE configs & gates | **Correct** (train pending) | Smoke passes; production ckpt not run |
| PyramidSchedule math | **Correct** | RF groups, envelope, targets match spec |
| Envelope orchestrator | **Mostly correct** | DFS order, parent lookup, CE slicing OK |
| Masks | **Partial** | Self-attn OK; cross-attn shift offsets unused |
| Reversible blocks | **Correct** | Invertibility verified; no RevBackProp hook |
| Predictor stages | **Mostly correct** | Dual-stream OK; factorized drops reversibility |
| Inference (AR) | **Incomplete** | Buffer tracked, not fed back into forward |
| Lightning / training | **Mostly correct** | Missing grad clip, shift_reentry wiring |
| Test coverage | **Good** (~84 tests) | ~40% of plan test IDs still missing |

**Bottom line:** Core math and data flow are sound for **training with teacher-forced context**. The largest gaps are **shift-aware cross masks**, **full autoregressive exposure bias**, and **operational VAE retrain**.

---

## Part 0 — Stage geometry @ T=13

### Spec

| Stage | Key | Native T | Shifts |
|-------|-----|----------|--------|
| 0 | h8_w8 | 4 | 1 |
| 1 | h16_w16 | 7 | 2 |
| 2 | h32_w32 | 13 | 4 |

Envelope: `(0,0)→(1,0)→(2,0)→(2,1)→(1,1)→(2,2)→(2,3)`

### Verification

- `PyramidSchedule.from_v2_64s()`: `native_t = [4,7,13]` ✓
- `context_end = [2, 4, 8]` → targets at indices `[3], [5,6], [9,10,11,12]` ✓
- `required_total_frames(9) = 13` ✓
- `envelope_order()` matches exactly (GLD-01 golden) ✓
- `parent_for(s,k) = (s-1, k//ratio)` aligns with DFS ancestry ✓

**Issue:** None for S=3, ratio=2, T=13.

---

## Part 1 — Encoder/Decoder v2

### 1.1 FrameWiseGroupNorm

**Status: CORRECT**

- Implemented in `norm.py`; applied to ResBlock norms, Encoder/Decoder `norm_out`.
- EV2-01 shape test passes.
- Prefix activation diff **6.1e-6** (within 1e-5 gate).

### 1.2 FrameWise AdaptiveGroupNorm

**Status: CORRECT**

- `FrameWiseVideoAdaptiveGroupNorm` pools style per-frame over (H,W) only.
- Style interpolated to match `x` spatial/temporal shape before pooling.
- EV2-07 passes.

### 1.3 Spatial tap keys

**Status: CORRECT**

- `_shape_key` → `h{H}_w{W}` (no T).
- `layer_string.py` accepts `h8_w8_x1`; legacy `t3_h8_w8_x1` shim with warning.
- `tap_key_format: spatial | temporal` supported in hierarchy config.
- EV2-04, EV2-05, EV2-06 pass.

### 1.4 Prefix-stability contract

**Status: MOSTLY CORRECT**

| Check | Result |
|-------|--------|
| Activation prefix T=9 vs T=13 | ✓ diff < 1e-5 |
| Quantized indices prefix | ⚠ **Not fully tested** — EV2-03 tests determinism only, not cross-length prefix equality |
| Hierarchical index prefix | ⚠ Expected to differ across T (only activations are prefix-stable) |

**Proposed fix:** Add explicit test:

```python
tok9 = model.encode_tokens(clip[:,:,:9])
tok13 = model.encode_tokens(clip)
assert torch.equal(tok13["levels"][2]["indices"][:,:9], tok9["levels"][2]["indices"])
```

(Same for stages 0,1 if native T allows.)

### 1.5 Checkpoint migration

**Status: PARTIAL**

- `checkpoint_v2.load_conv_weights_partial()` copies conv weights, skips norms.
- EV2-10 unit test passes.
- **Missing:** CLI script `scripts/migrate_jq0ynj5m_to_v2.py`.

---

## Part 2 — VAE retrain @ T=13

**Status: CONFIG CORRECT; TRAIN NOT DONE**

- `shapes3d_sqvae2_64_S_v2.yaml` matches plan (spatial keys, T=13, quantizer sizes).
- Smoke config runs 2 epochs on CPU.
- `validate_v2_vae_gate.py` passes on untrained weights.

**Blocker for predictor quality:** Production `val_ema/mse` checkpoint required. jq0ynj5m (T=9, temporal keys) is **not compatible** without retrain.

---

## Part 3 — Predictor core (schedule, masks, batch_prep)

### 3.1 PyramidSchedule

**Status: CORRECT**

Verified formulas:
- `causal_output_length(T) = (T-1)//2 + 1` composed per level.
- RF groups via `causal_receptive_groups` with k=3, s=2.
- RF-01 manual check: mid groups from T=13 include `{6,7,8}` at m5 ✓

**Gaps:**
- `from_encoder_audit(ddconfig, hierarchy_cfg, …)` **not implemented** — stages hardcoded in `VideoHierPredictorModel`.
- RF-05, RF-09, RF-10 tests partially covered; 1×1 top not tested.

**Proposed fix:** Add factory that reads `audit_encoder_taps` + quantizer config.

### 3.2 Masks

**Status: PARTIAL — important gap**

#### Self-attention: CORRECT

```python
frame_mask = (diff >= 0) & (diff < temporal_window)  # window=-1 → full causal
```

- MSK-01 causal verified.
- Bot `window=1` → block-diagonal in time (implicit via frame_mask).

#### Cross-attention: INCOMPLETE

```python
def build_cross_attn_mask(..., child_shift=0, parent_shift=0, ...):
    # child_shift, parent_shift ARE NEVER USED
```

Current mask = RF frame intersection × spatial `(r,c) → (r//2,c//2)`.

**Missing per plan (MSK-06, MSK-04):**
- Shift offsets on finest timeline when parent/child shifts differ.
- Multi-parent temporal overlap refinement (e.g. m3 → {t2,t3}).

**Impact:** Cross-attn may allow parent tokens from wrong shift branch when `(1,0)` vs `(1,1)` parent outputs differ in length/content. Currently mitigated because parent outputs share context-length token count per stage, but **shift-specific RF cropping is not applied**.

**Proposed fix:**

```python
def build_cross_attn_mask(..., child_shift, parent_shift):
    # Offset finest-frame sets by shift indices before intersection
    child_rf_shifted = shift_rf_frames(child_rf, child_shift, schedule, child_stage)
    parent_rf_shifted = shift_rf_frames(parent_rf, parent_shift, schedule, parent_stage)
    # Then intersect + spatial block
```

Add MSK-06 golden test comparing masks for `(s=2,k=1, ps=1,pk=0)` vs `(pk=1)`.

### 3.3 BatchPrep

**Status: CORRECT**

Flow:
1. `encode_tokens(video, flg_quant_det=True)` single pass ✓
2. Context embed = indices `[:, :context_end+1]` per stage ✓
3. Target indices from GT at `target_token_index(s,k)` ✓
4. All masks precomputed ✓

`build_full_context_embed()` exists for oracle eval but **not used** by parallel mode.

---

## Part 4 — Predictor neural modules

### 4.2 PredictorStage

**Status: MOSTLY CORRECT**

| Spec item | Implementation |
|-----------|----------------|
| k=0: `cat([proj(ctx), proj(ctx)])` + pos + shift | ✓ |
| k>0: `cat([O1,O2])` + shift, no GT | ✓ |
| Frozen codebook copy from VAE | ✓ |
| Reversible layers (coarse) | ✓ |
| Factorized (mid/fine) | ✓ but **not reversible** |

**Issue — query/CE geometry:**

All bot shifts use **same query frame** `query_t = context_end = 8`:

```python
query_t = min(ctx_end, target_token_index(s,k) - 1)  # always 8 for bot k=0..3
```

Disambiguation relies on `shift_embed` + stream carry. This matches `streams_only` + hierarchical-vq-transformers port semantics but is **non-standard AR** (typically query advances with horizon).

**Verdict:** Consistent with locked `streams_only` design; document clearly.

### 4.3 ParentCondition

**Status: PARTIAL**

| Mode | Status |
|------|--------|
| `dual_stream` | ✓ even→O1, odd→O2 |
| `fused_soft` | ⚠ fused tensor passed but no softmax before KV |
| `fused_hard` | ✗ not implemented (argmax→embed) |

### 4.4 EnvelopeOrchestrator

**Status: MOSTLY CORRECT**

- DFS envelope order ✓
- `parent_by_stage[ps]` lookup after each event ✓
- CE on sliced query logits vs flat targets ✓
- `pred_indices` splices argmax into GT tensor at horizon positions ✓
- Pred MSE delegated to `VideoHierPredictorModel._decode_horizon_mse` ✓

**Issue — parent token length:**

Parent `o1/o2` have length = **context tokens** (shift-0 path), not full native T. Cross-mask built for full native T then sliced:

```python
cross_m = masks.cross_attn[:n_tokens, :parent_n]
```

This is **consistent** if parent forward also uses context length at shift 0. Verified: mid shift 0 → 1280 child tokens, top → 192 parent tokens.

**Issue — unused supervision fields:**

`ShiftSupervision.query_positions` / `target_positions` precomputed but orchestrator recomputes positions inline instead of using them.

### 4.5 Attention backends

**Status: PARTIAL**

| Stage | Plan | Implemented |
|-------|------|-------------|
| coarse | full reversible | ✓ ReversibleCouplingBlock |
| mid | factorized, t_window=3 | ✓ FactorizedPredictorLayer |
| fine | factorized, t_window=1, spatial_window=8 | ⚠ t_window=1 only; **no spatial_window**; full spatial attn (1024 tokens/frame) |

**OOM risk:** Fine stage does global spatial attn per frame (1024×1024 per frame). Plan intended `spatial_window=8` local attention (~64 tokens). Current fine stage may OOM at batch_size>1 on consumer GPUs.

**Proposed fix:** Add windowed spatial MHA in `FactorizedSpaceTimeBlock`:

```python
# spatial: local (2*sw+1)^2 neighborhood per (r,c)
```

**Factorized + reversibility:** Plan allows non-reversible fine stage ("Bot attention: Factorized"). Reversibility is per-shift memory tool only — OK to drop on fine.

### 4.6 VideoHierPredictorModel

**Status: MOSTLY CORRECT**

| Item | Status |
|------|--------|
| Freeze VAE | ✓ |
| Copy codebooks | ✓ |
| CE + pred MSE | ✓ |
| AdamW | ✓ |
| Grad clip 1.0 | ✗ missing |
| `shift_reentry` config | ✗ not read |
| `inference.commit` (sample) | ✗ argmax only |

**Pred MSE path:** Correct per plan — zero activations, splice predicted horizon indices, MSE on RGB[:,:,9:13].

---

## Part 5 — Training config

**Status: CORRECT** (yaml matches plan)

Missing runtime wiring: `shift_reentry.mode: streams_only` is documentation-only; behavior is hardcoded in `PredictorStage.execute_shift`.

---

## Part 6 — Inference modes

| Mode | Plan | Implemented | Gap |
|------|------|-------------|-----|
| parallel | GT context; eval upper bound | Identical to train | **No oracle horizon embeds** |
| autoregressive | argmax→embed→update buffer; carry streams | Buffer grows; **not fed into k>0 forward** | **Exposure bias not active** |

### AR detail

Current AR:
- Clones `rolling_ctx`; after each finest shift appends `embed(argmax)`.
- Forward path for k>0 still uses `streams_only` (O1,O2 carry).
- `rolling_ctx` only affects `ar_context_len` metric.

Attempt to re-init from extended buffer caused NaN (attention mask / length mismatch) and was reverted.

**Proposed fix (safe AR rollout):**

1. After bot shift k, commit `pred_idx` into a **mutable index buffer** (not embed buffer).
2. For shift k+1 on finest, optionally refresh **only** the output head input from updated stream (keep streams_only in rev blocks).
3. Or: separate `forward_autoregressive_generate()` that runs bot shifts sequentially with `t_len+1` mask rebuild per step.

**Parallel upper bound fix:**

```python
def forward_parallel(self, batch):
    full_ctx = {s: stages[s].embed_indices(batch.gt_indices[s]) for s in ...}
    # Use full native-T embeds at shift 0; measure lower CE
```

---

## Part 7 — Reversibility audit

### ReversibleCouplingBlock

```
o2 = i2 + F(i1)    # F has self + cross attn
o1 = i1 + G(o2)
```

- REV-01: `inverse(forward(x)) ≈ x` at ε=1e-5 ✓
- Cross-attn in both F and G ✓ (same `_cross_kv` per layer parity)
- **REV-02 gradcheck:** not implemented
- **REV-03 EfficientRevBackProp:** not implemented — training uses standard autograd through forward only (rev inverse unused in backward)

**Impact:** Higher memory vs true RevNet training; functionally correct.

**Note:** Plan says "RevNet is per-shift memory tool only" — inverse across shifts correctly avoided.

---

## Part 7 — Test coverage vs plan

| Group | Plan IDs | Covered | Missing |
|-------|----------|---------|---------|
| EV2 | 10 | 9 | EV2-03 full prefix indices |
| RF | 10 | 7 | RF-05, RF-09, RF-10 |
| ENV | 5 | 5 | — |
| TGT | 6 | 6 | — |
| MSK | 8 | 2 | MSK-02,04-08 |
| REV/DS | 8 | 2 | REV-02-04, DS-03-04 |
| STG | 6 | 5 | STG-05 codebook match |
| ORC | 7 | 1 | ORC-02-07 |
| BAT | 5 | 4 | BAT-05 ckpt load |
| INF | 5 | 4 | INF-04 decode length |
| ATT | 3 | 3 | ATT-03 equiv full≈factorized |
| GLD | 2 | 1 | GLD-02 e2e loss tolerance |
| FUZ | 2 | 2 | — |

**Total:** ~84 tests vs 70+ planned IDs (~55% ID coverage, good smoke depth).

---

## Critical issue priority list

### P0 — Before production predictor training

1. **Train VAE v2 checkpoint** (`shapes3d_sqvae2_64_S_v2.yaml`).
2. **Set `vae_ckpt`** in predict yaml.

### P1 — Correctness / quality

3. **Shift-aware cross-attention masks** (MSK-06).
4. **Windowed spatial attention** on fine stage (`spatial_window=8`) to avoid OOM.
5. **Full AR exposure bias** — feed committed embeds or rebuild masks per bot shift.

### P2 — Plan completeness

6. Wire `grad_clip=1.0` in Lightning.
7. Implement `fused_hard` / `inference.commit=sample`.
8. `forward_parallel` oracle mode for eval upper bound.
9. `PyramidSchedule.from_encoder_audit()`.
10. CLI checkpoint migration script.

### P3 — Nice to have

11. EfficientRevBackProp for memory.
12. Remaining golden/fuzz tests.
13. EV2-03 cross-T index prefix test.

---

## Data flow diagram (verified)

```
RGB (B,3,13,H,W)
  └─► VAE.encode_tokens ──► levels[s].indices (B,Ts,Hs,Ws)
        └─► embed [:context_end+1] ──► context_embed[s]  (shift 0 input)

Envelope DFS (s,k):
  parent = parent_by_stage[s-1] from prior event on DFS path
  if k==0: execute_shift(context_embed[s])
  else:    execute_shift(stream_carry[s,k-1])
  CE: logits[s,k][:, query_frame] vs GT indices at target_token[s,k]
  stream_carry[s,k] = (O1, O2)

After envelope:
  pred_indices[s] = GT context + argmax horizon slots
  MSE: decode(pred_indices, zero_acts) vs RGB[:,:,9:13]
```

---

## Conclusion

The implementation is **architecturally faithful** for the core training path:
- Correct stage geometry, envelope order, RF-based targets.
- Sound encoder v2 prefix stability enabling variable T.
- Working reversible coarse stage + factorized mid/fine.
- End-to-end CE + pred MSE pipeline functional (84 tests).

**Not yet 100% complete** relative to plan:
- Cross-mask shift awareness (math gap).
- AR inference (logic gap — buffer not consumed).
- Fine spatial window (scalability gap).
- Production VAE weights (operational gap).

Recommend addressing **P0 + P1** before trusting predictor metrics for research decisions.
