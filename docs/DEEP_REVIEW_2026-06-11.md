# Deep review — hierarchical SQ-VAE-2 tokenizer + hierarchical video predictor

**Date:** 2026-06-11
**Scope:** full stack on the `lite` configs; W&B runs `idomjiua` (tokenizer) and `wgm3d46g` (predictor); code, math, architecture, training dynamics.
**Method:** line-level code review + empirical verification scripts (`scripts/layer_ablation_review.py`, `scripts/token_stats_review.py`, `scripts/grad_flow_review.py`, `scripts/analyze_wandb_review.py`) run against the epoch-67 tokenizer checkpoint and the live W&B histories.

---

## 0. Executive summary

The codebase is well-engineered (clean module boundaries, good test coverage, mask caching, NaN guards, fp32 CE, zero-init heads). The two *concepts* — hierarchical envelope prediction with per-stage vocab distributions, and reversible stream-carry within a stage — are sound and worth keeping. But **five verified defects currently make the system measure something different from what it is designed to do**:

| # | Finding | Severity | Verified by |
|---|---------|----------|-------------|
| 1 | ELBO KL term is ~10⁵–10⁶× too weak (mean instead of sum over positions and K) → tokenizer trains as a plain AE → codebook collapse (53/4096, 140/2048, 76/1024 active codes) | **P0** | code + W&B magnitudes |
| 2 | All hierarchy levels are fused **down** to the 3×4×4 bottleneck grid; fine detail is destroyed by trilinear downsampling; the **mid** layer carries nearly everything (ablating it: MSE 0.0025 → 0.236; coarse: → 0.023 only) | **P0** | layer ablation |
| 3 | Coarse token stream is temporally chaotic (2.8 % of tokens persist between coarse steps; knowing the previous token gives only 0.36 nats) → coarse stage is **unpredictable in principle**; the predictor is not the problem | **P0** | token statistics |
| 4 | Predictor AR-eval path is never trained: finest-stage re-entry re-embeds tokens through a **dead random projection** (`codebook_proj` gets no gradients — `@torch.no_grad` in `batch_prep`) and uses **never-trained** horizon `temporal_pos` slots → `val/loss_ce_ar` grows 22 → 35 monotonically | **P0** | grad-flow check + W&B |
| 5 | `forward_parallel(parallel_mode="context")` is computationally identical to `forward_train` → "parallel" val metrics are just train-mode-on-val; `loss_pred_mse` carries **zero gradient** (decodes argmax indices through the frozen VAE) → pure overhead each step + it skews the `val/loss_total_ar` checkpoint monitor | **P1** | grad-flow check + metric equality in logs |

Direct answers to your questions:

- **"Can the decoder bypass coarse codes via the fine encoder skip?"** No — there are **no encoder→decoder skip connections**. `decode()` consumes only `z_q` (and reuses it as the AdaGN style input). `decode_from_indices` takes `activations`/`encoder_bottleneck` **only to read shapes** (the predictor passes zeros and reconstruction works). The real imbalance is the opposite of your fear: the **mid layer** dominates, the coarse layer is nearly decorative, and the fine layer's 16×16 content is averaged down to 4×4 before the decoder ever sees it.
- **"Is val AR growth a bug?"** Yes, structural: train mode evolves stream states, AR mode (finest stage only) re-embeds committed tokens into an extended context — an input distribution the network never sees in training. The stream-carry path itself generalizes fine (`val_parallel/ce_s2_k3` = 1.27 vs AR 11.1).
- **"Progressive training / per-stage decoders?"** Yes — and progressive ELBO is **already implemented** (`progressive_coding: true` → `decode_progressive` + per-stage rows in `log_images`). It is the cheapest way to force each level to be standalone-meaningful. Details in §5, Path B.
- **"Should I drop VQ-VAE?"** Keep discrete tokens (your entropy-map rationale is correct and is a genuine advantage). Replace/augment the *quantizer family and fusion topology*, not the concept. See §5.

---

## 1. Tokenizer findings (SQ-VAE-2 side)

### 1.1 [P0] ELBO term scaling is wrong relative to SQ-VAE/HQ-VAE

`gaussian_sq.py`:

```python
kld_discrete = (prob_pos * (log_prob_pos - log_prob_pri)).mean(dim=(1, 2, 3, 4)).mean()
```

The categorical KL must be **summed over K** (it is an expectation Σₖ qₖ·log(qₖ/pₖ)) and, in the ELBO, **summed over latent positions** (each position is a latent variable). Original SQ-VAE: `torch.sum(prob * (log_prob - log_prob_prior), dim=(1,2)).mean()`. Using `mean` divides the KL by `T·H·W·K` (~0.6–6 × 10⁶ depending on layer) and — because K differs per layer (4096/2048/1024) — applies an *inconsistent* relative weight across layers.

Meanwhile the distortion is ARELBO `dim_x/2 · log(MSE_sum)` with `dim_x = 27 648`. The W&B magnitudes confirm: distortion ≈ 59 000–130 000, `kl_total` ≈ 0.02–23. **The KL is numerically irrelevant; SQ-VAE's self-annealing balance (which the whole method depends on) cannot operate.** Consequences observed:

- Codebook collapse: perplexity L1 4063→44, L3 996→20; active codes 53/4096, 140/2048, 76/1024 (1–7 % usage).
- `log_param_q` (posterior var) decays without the KL counter-pressure (60 → 0.97/2.3/3.6, still falling at interrupt) — the temperature/variance annealing is driven only by distortion gradients.
- `kl_layer_3_raw` grows 0.02 → 22.9 across training. Layer 3 is the only `flg_loss_continuous` layer; its term is `MSE(z_res, z_q)·0.5/Σvar` — as var decays, this blows up. This is the slow-motion destabilizer you felt as "training instability".

**Fix** (small, in `gaussian_sq.py`): sum over K and over (T,H,W), mean over batch only, for both `kld_discrete` and `kld_continuous`. Then re-tune: `kl_weights` (already plumbed through `loss_cfg`) becomes a meaningful β; start with β=1 and a short warm-up. Expect very different (healthier) perplexity dynamics.

### 1.2 [P0] Fusion topology: everything is squeezed through 3×4×4×16

`video_inj_topdown.py` (`token_grid: native`, `latent_key: h4_w4`): every layer's `z_q` is `fuse_to_latent` → trilinear **down**-sampled to the bottleneck grid and summed. The decoder's entire input budget is 3·4·4·16 = 768 floats per 9-frame clip, regardless of how many levels exist. The 9×16×16 fine tokens contribute only their 4×4 temporal-spatial average.

Empirical (epoch-67 ckpt, 16 val videos, random-code ablation):

| Ablation | MSE vs GT | Δ vs baseline 0.00254 |
|---|---|---|
| randomize **L1** (h4_w4, K=4096) | 0.0229 | +0.020 |
| randomize **L2** (h8_w8, K=2048) | **0.2363** | **+0.234** |
| randomize **L3** (h16_w16, K=1024) | 0.0340 | +0.031 |
| keep ONLY L1 | 0.305 | — |
| keep ONLY L2 | **0.053** | — |
| keep ONLY L3 | 0.286 | — |

The hierarchy is inverted/degenerate: L2's *residual* absorbed the global structure (L1's collapsed 54-code vocabulary explains little variance), and L3 is mostly redundant. The intended semantics — "coarse = slow global scene, fine = local detail" — does not hold, which directly poisons the predictor's coarse stage.

Secondary issue: in native mode the "residual" `act_proj(act_8x8) − upsample(z_latent)` subtracts tensors from **different feature spaces** (a mid-encoder 64-ch activation through a 1×1 conv vs the post-`conv_out` bottleneck space). It can be learned around, but the residual semantics is weak; HQ-VAE's injected top-down (your `InjSQBlock`/pyramid path) is the principled version.

### 1.3 [P0] Coarse codes are temporally chaotic → unpredictable by construction

Token statistics over 128 val videos (`scripts/token_stats_review.py`):

| Layer | P(token unchanged at same position, next step) | H(marginal) | H(next \| prev@same-pos) |
|---|---|---|---|
| L1 coarse | **0.028** | 3.83 nats | 3.47 |
| L2 mid | 0.123 | 3.71 | 3.13 |
| L3 fine | 0.105 | 2.96 | 2.58 |

For *smoothly rotating shapes*, a healthy coarse code should be the **most** temporally stable; here it is the least. The predictor's val coarse CE (3.54) sits between H(next|prev) and H(marginal) — i.e. the predictor already extracts most of what is extractable. **Improving the predictor cannot fix the coarse stage; the tokenizer must produce temporally stable codes first.** (Root causes: §1.1 collapse + §1.2 information-dense bottleneck tap + stochastic gumbel quantization with no temporal smoothness pressure.)

### 1.4 Smaller tokenizer issues

- **No gradient clipping**: `VideoHierVQModel` uses manual optimization; Lightning's `gradient_clip_val` is ignored in manual mode (and isn't set in the VAE config anyway). Add `self.clip_gradients(...)` around `manual_backward` if instability persists after 1.1.
- `avg_probs` computed in both quantizer branches but unused (dead variable); `_arelbo_distortion` clamps at 1e-8 fine.
- `seed_everything: true` without a fixed seed → new random seed each run; for dissertation reproducibility pin it.
- Encoder run was interrupted at epoch 68/2000 (KeyboardInterrupt; W&B shows state "failed" for both runs — neither crashed). `val_ema/mse_per_pixel` was still improving (0.00276 and falling); the checkpoint is *under*-trained.
- `Upsampler` temporal frame-drop (`[0] + [2:]`) is a hack that works for T = 4k+1 (9→5→3); document the invariant — it silently corrupts other T.

---

## 2. Predictor findings

### 2.1 [P0] The mode triangle: train / parallel / autoregressive

Verified semantics in `orchestrator.py`:

- **train**: shift 0 consumes GT context embeds; shifts k>0 consume **carried stream states** (o1,o2). No tokens are ever fed back. CE on the single target frame per shift.
- **parallel(context)** (your configured val "parallel"): **identical computation to train** — confirmed by code path and by exact metric equality (`val/ce_parallel_stage_{0,1}` ≡ `val/ce_ar_stage_{0,1}` and stage-0/1 token-acc equality in the logs). It measures nothing independent; it is train-mode CE on val data.
- **parallel(full)**: oracle (full GT sequence embedded) — not what you log.
- **autoregressive**: identical to train **except** the finest stage at k>0, which re-embeds committed (argmax) tokens, appends to the context, and re-runs full attention with the extended mask.

So exactly one path differs between training and AR-eval, and it is the one that explodes: `val/ce_s2_k0` 1.13 (= parallel, same input) vs `val/ce_s2_k1/k2/k3` → 7.4 / 8.9 / 11.1, all growing monotonically while train CE falls. Three compounding causes, all verified:

1. **`codebook_proj` is a dead parameter.** `BatchPrep.encode_and_schedule` is `@torch.no_grad()`, and context embeds are built inside it → the trainable 96→dim projection never receives a gradient (verified: `grad is None` after backward on all stages). The fine stage therefore embeds tokens through a **frozen random 96→16** map both in training and in AR re-entry. The network learns to read random features for the context, but the AR path *additionally* depends on this map for self-predicted tokens at new positions.
2. **Horizon `temporal_pos` slots are never trained** (verified: zero gradient for `temporal_pos[:, ctx_end+1:]`). AR re-entry adds tokens at those positions → random positional codes at eval only.
3. **Exposure/structure mismatch**: stream-carry evolution vs re-embedded-token attention are different computations; the network specializes to the former, so the gap grows with training. This is the "partial AR exposure bias" your Path-C audit already flags — but the magnitude (CE 1.27 vs 11.1 at k3) makes it a blocker, not a footnote.

Note the encouraging flip side: **the stream-carry path itself generalizes** (`val_parallel/ce_s2_k3` = 1.27, token-acc 63 %). Your dissertation mechanism works *within the envelope*; only the re-entry mechanism is untrained.

### 2.2 [P1] `loss_pred_mse` has no gradient and distorts model selection

`decode_from_indices(argmax indices)` through the frozen VAE → `out.loss_mse.requires_grad == False` (verified). Every training step pays a full VAE decode for a constant. Worse, it is added (×0.5) into `val/loss_total_ar`, the **ModelCheckpoint monitor**, so checkpoint ranking mixes a non-differentiable, AR-flavored MSE into the criterion. Also `pred_mse_per_pixel` double-divides (`loss_mse` is already a mean over pixels; it is then divided by `numel`), which is why the logged per-pixel values are ~0.

### 2.3 [P1] Supervision sparsity / sample efficiency

Each step encodes the full video through the 3-D encoder, runs 7 shift passes, and supervises **one frame transition per shift** (7 CE terms, only horizon transitions). The context transitions (e.g. fine frames 1→2, 2→3, 3→4) are never supervised, although the causal masks would allow supervising every position in one pass. You're paying ~9-frame compute for ~1-frame learning signal per stage. Dense causal supervision would also train the temporal_pos slots and dramatically increase effective dataset size per epoch.

### 2.4 [P1] Capacity allocation is inverted at the fine stage

`dim: [256, 64, 16]`, `n_layers: [6, 4, 2]`, heads `[8, 8, 4]` → fine stage: dim 16, head_dim 4, 2 layers, **t_window = 1** (no temporal self-attention at all beyond the current frame), spatial window 4, vocab 1024, and a rank-16 logit head over 1024 classes. The fine stage handles 4× more shifts and 16× more tokens than coarse with ~1/250 of the per-token compute. "More compute for coarse" is a fine philosophy for *representation*, but the **output head and embedding dims** must still be commensurate with the vocabulary (rank-16 logits put a hard floor on fine CE). Meanwhile the coarse stage burns 6 layers × dim 256 on a 2-token-context, 16-position problem that is information-theoretically hopeless with current tokens (§1.3).

### 2.5 [P2] Other predictor observations

- **Cross-attention is 1-to-1**: each child token may attend to exactly one aligned parent token (`_spatial_block` puts a single True per row). No parent neighborhood → narrow conditioning channel. A 3×3 parent window would cost little.
- **Reversibility actually applies only to the coarse stage** in the lite config (factorized stages use plain `FactorizedPredictorLayer`; `rev_layers` only exist for `type: full`), and `use_reversible_backprop` is off, so no memory is saved anywhere right now. Also in the factorized path the two streams **never interact** (block_f/block_g are independent until the head) — it's two parallel networks, double cost, no coupling benefit.
- **Custom rev-backprop + cross-stage conditioning is risky when enabled**: `ReversibleCouplingBlock.backward_pass` either detaches `cross_kv` (dropping parent gradients) or calls `.backward(retain_graph=True)` into the parent's live graph per layer per shift (repeated partial backprops). Before using it for dissertation results, add a gradcheck test that compares full-graph vs custom-backward gradients **including parent stages**.
- Within-frame tokens are conditionally independent given the features (all 256 fine tokens of a frame are emitted in one forward). For stochastic data this marginalizes multimodality; for your near-deterministic shapes it's acceptable for now, but plan MaskGIT-style iterative refinement or within-frame raster AR later.
- `objectives/predictor_loss.py` is referenced by nothing in `src/` (dead code; only tests import it).
- Predictor runs fully in fp32 (`autocast(enabled=False)`) — fine for stability; the `16-mixed` trainer flag only affects the frozen VAE encode. Consider bf16 for speed once stable.
- Train accuracy at coarse (17 % and still rising at interrupt, val 8 %) with val CE ≈ marginal entropy → the coarse stage is **memorizing** train tokens; expected given §1.3.

### 2.6 Infrastructure / metrics

- Both runs ended via KeyboardInterrupt (state "failed" on W&B is just how it records interrupts).
- The full-suite pytest crash (`Windows fatal exception: access violation` inside MHA softmax) is an OpenMP/Windows native issue, not a code bug — the same file passes alone and with `OMP_NUM_THREADS=1`. Set `OMP_NUM_THREADS=1` (or `torch.set_num_threads(1)`) in `pytest.ini`/CI to deflake.
- `persistent_workers=True` recommendation from Lightning is valid here (8 workers re-spawn every epoch + every val epoch).

---

## 3. What is genuinely good (keep these)

- **Envelope/schedule math is correct.** I verified `_context_end`, RF-group composition and the parent ordering: fine shifts {0,1} consume the parent state that predicted mid token 3 (RF {4,5,6}), shifts {2,3} the state for mid token 4 (RF {6,7,8}). The DFS order delivers exactly the right parent at the right time.
- Stream-carry as a mechanism **works and generalizes** (val parallel s2_k3 CE 1.27) — your continuous-evolution thesis transfers.
- Per-stage CE/accuracy breakdown logging (`ce_s{s}_k{k}`) is exactly what made this diagnosis possible; `ce_over_baseline` is a nice normalization.
- Zero-init output heads + residual branches, fp32 CE, mask pre-baking (`PredictorMaskCache`), NaN guard, golden/fuzz/invertibility tests — solid engineering hygiene.
- The codebase separation (schedule / masks / stage / orchestrator) made this review tractable; that's worth a sentence in the dissertation's engineering section.
- Entropy maps: with per-stage logits you indeed get calibrated uncertainty for free once CE is trained — the design goal is achievable as architected.

---

## 4. Prioritized fix list

**P0 — correctness (do before any new training run)**
1. `gaussian_sq.py`: KL sums over K and positions (discrete and continuous terms); keep batch mean. Re-tune `kl_weights` as β with warm-up.
2. `batch_prep.py`: move `codebook_proj` application out of `no_grad` (embed raw codebook vectors under `no_grad`, apply the trainable projection inside the grad context in `execute_shift` / `embed_token_ids`).
3. Decide the canonical rollout (recommendation: **stream-carry**, §5 Path C) and make val measure it; rename/fix `parallel(context)` (it duplicates train) and stop adding the non-differentiable MSE into the checkpoint monitor (monitor `val/loss_ce_ar` or a renamed stream-carry CE).
4. Gate `_decode_horizon_mse` behind `torch.no_grad()` + every-N-steps (it's logging, not loss).

**P1 — make the hierarchy real**
5. Restructure fusion so fine information survives (§5 Path A/B): pyramid `u2` top-down with the decoder consuming the finest-grid state, or progressive coding on the native path.
6. Dense causal supervision in train mode (every frame transition, not just horizon) — trains temporal_pos, ~5–8× more signal per step.
7. Rebalance stage dims (e.g. `[256, 128, 64]`), give fine `t_window ≥ 2`, widen cross-attn to a 3×3 parent neighborhood.
8. Fix `pred_mse_per_pixel` double division.

**P2 — efficiency / hygiene**
9. Pre-tokenize the dataset once (frozen VAE + cached, deterministic clips → cache `(indices_L1..L3)` per sample); predictor epochs become pure-transformer speed (the 3-D encoder forward is most of your step time).
10. Remove dead code (`objectives/`), pin seed, `persistent_workers: true`, `OMP_NUM_THREADS=1` for pytest.
11. Add the parent-gradient gradcheck for `EfficientRevBackProp` before relying on it.

---

## 5. Strategic paths forward

### Path A (recommended core): fix ELBO + pyramid latent, retrain tokenizer, gate on token stability

The single highest-leverage change. Steps:

1. Apply fix 4.1 (KL sums). Add a `kl_weights` warm-up (β: 0→1 over ~5 epochs).
2. Switch the lite config to the HQ-VAE pyramid your code already supports:
   `token_grid: pyramid`, `blocks_sq: "h4_w4_x1,h8_w8_u2,h16_w16_u2"` — coarse injected at 4×4, `InjSQBlock` upsamples state to 8×8 then 16×16.
3. **Feed the decoder the finest `z_state` (16×16) instead of the down-fused `z_latent`.** Concretely: return `z_state` from `SQVAE2TopDown.forward` in pyramid mode and use a decoder with one spatial upsample (16→32) — either a `ddconfig` variant (`ch_mult: [1, 2]`-style) or a thin head. This removes the 4×4 information bottleneck *and* gives the hierarchy its intended direction (coarse → carries global scene; fine → adds detail).
4. Retrain and gate on these metrics *before* touching the predictor:
   - per-layer perplexity_frac ≥ ~0.1 (vs current 0.01–0.02),
   - token persistence: coarse > mid > fine, coarse ≥ ~0.5 for this dataset (vs 0.028),
   - balanced ablation deltas (each layer's randomization should hurt, monotonically decreasing coarse→fine for global error),
   - recon `val_ema/mse_per_pixel` ≤ current 0.0027.
   (`scripts/token_stats_review.py` and `scripts/layer_ablation_review.py` compute these.)
5. Optional stabilizers if collapse persists after the KL fix: EMA codebook updates or dead-code re-initialization (standard VQ-VAE-2 practice), and a mild temporal smoothness prior on coarse logits (KL between q(z_t) and q(z_{t-1}) at the coarse layer only — cheap and directly targets §1.3).

### Path B (cheap parallel experiment): progressive coding — your own question, answered

`progressive_coding: true` already exists end-to-end (`decode_progressive`, averaged ARELBO over partial sums, `progressive_L1..L3` image rows). It forces the L1-only decode to look like the scene → directly attacks "coarse codes don't represent the actual thing", and gives you the per-stage visualizations you asked for without new decoders.

- Cost: ~S× decoder forward per step; averaging softens the pressure — consider weighting partial reconstructions (e.g. [0.5, 0.75, 1.0]).
- Per-stage *separate* decoders are the heavier alternative; only worth it if you need stage outputs at different resolutions for the dissertation figures. Start with progressive; add decoders later if needed.
- Note: progressive + native fusion still suffers the 4×4 bottleneck — combine with Path A's pyramid for full effect.

### Path C: predictor consistency — train what you test

Decide the canonical inference story. Two coherent options; I recommend C1 for the dissertation narrative, with C2 as the long-horizon extension.

**C1 — stream-carry as the rollout (matches your RevViT thesis).**
1. Make `forward_autoregressive` use stream carry for *all* stages (drop the finest-stage `ar_reinit`) → AR ≡ train within the envelope; the growing val gap disappears by construction.
2. Keep CE/entropy maps per shift (you already have them).
3. For *multi-envelope* rollout (beyond t_total): re-enter by re-encoding the committed token window as fresh context. This re-entry **must then be trained**: append a second envelope per training step on a shifted window (cheap if you pre-tokenize per fix 9 — targets are just later GT tokens), teacher-forcing the new context from GT with probability p and from committed predictions with 1−p (scheduled sampling, anneal p 1→0.5).
4. Fix `codebook_proj` (P0.2) first — re-entry embeds predicted tokens through it.

**C2 — conventional AR training (bigger change, more standard).**
Dense causal teacher forcing: feed the full native sequence with causal masks, supervise every t→t+1 transition at every stage, predict the horizon by standard KV-cache AR decode. This makes train/inference identical, trains all positions, and turns "parallel(full)" into the train mode. You lose the pure stream-carry story but gain the data efficiency of fix 6 automatically. (These can be combined: dense supervision within context + stream-carry across the horizon.)

### Path D: quantizer family swap (if collapse persists or you want robustness)

Keep the hierarchy and predictor; replace per-layer `GaussianSQQuantizer` with **FSQ** (finite scalar quantization) or **LFQ/BSQ** (already half-supported via `build_layer_quantizer`'s lfq path):
- FSQ: no codebook, no collapse, no commitment/KL machinery — levels like `[8,6,5]`→K=240 coarse, `[8,8,8]`→512 fine. ~50 lines per layer; the predictor only needs `codebook_size` and an embedding for token ids (which FSQ provides as implicit codes).
- This decouples "discrete + entropy maps" (your fundamental requirement) from "VQ codebook health" (your current pain). The SQ-VAE variance-annealing story is lost, but for the dissertation the hierarchy + reversibility + uncertainty maps are the claims — not the quantizer brand.
- Suggested experiment: same encoder/decoder + pyramid fusion, FSQ per level, compare token persistence and predictor CE against Path A's fixed SQ. Whichever yields stabler codes wins the slot.

### Suggested order

1. P0 fixes (1–4) — a day of work, mostly small diffs.
2. Path A retrain (tokenizer) + gate metrics; Path B flag flip can ride along as an ablation.
3. Path C1 on the new tokens (+ pre-tokenization, fix 9).
4. Path D only if A's gates fail; C2/MaskGIT/RoPE-time as later upgrades when moving to real data.

### SOTA references worth importing when scaling up

- MAGVIT-v2 / Open-MAGVIT2: LFQ with entropy penalties (sample-entropy ↓, batch-entropy ↑) — the principled anti-collapse objective if you stay codebook-based; token factorization for large K (4096 → 2×64) to shrink prediction heads.
- FSQ (Mentzer et al. 2023) — collapse-free discrete latents.
- MaskGIT / Phenaki-style iterative parallel decoding — fixes within-frame conditional independence at sampling time.
- Diffusion forcing / scheduled sampling — principled exposure-bias treatments for rollout.
- RoPE or ALiBi on the time axis — removes the untrained-absolute-position failure mode entirely (relative encodings generalize to unseen horizon lengths; important once t_total varies).
- VideoGPT/TECO-style "tokenize once, train predictor on cached tokens" pipelines — your fix 9.

---

## 6. Evidence appendix

- **W&B trends** (`scripts/analyze_wandb_review.py` output, 2026-06-11): encoder distortion 115 629→59 376 vs kl_total 0.017→22.9; perplexities collapsing as listed; predictor `val/loss_ce_ar` 22.99→34.93 while `val/loss_ce_parallel` 22.99→10.79; `val/ce_s2_k1..k3` diverging, `val_parallel/ce_s2_k3`=1.27.
- **Layer ablation** (epoch-67 ckpt): table in §1.2; per-layer fused-contribution RMS: L1 0.42, L2 1.18, L3 0.73 (z_latent 1.49).
- **Token statistics**: table in §1.3.
- **Gradient flow** (random-perturbed heads to escape zero-init): `codebook_proj.grad = None` all stages; `temporal_pos` horizon-slot grad = 0.0 all stages; `loss_mse.requires_grad = False`.
- **Tests**: 8/8 `test_predictor_logging.py` pass in isolation; full-suite native crash is an OpenMP/Windows issue (use `OMP_NUM_THREADS=1`).
