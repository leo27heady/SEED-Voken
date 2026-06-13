# PROJECT MEMORY — SEED-Voken hierarchical video prediction

> Living context document. Captures everything learned/decided/implemented through
> **2026-06-12** (deep review → Path A implementation → Run-1 analysis → test-memory fix).
> Read this first when returning to the project. Companion docs:
> [DEEP_REVIEW_2026-06-11.md](./DEEP_REVIEW_2026-06-11.md) (full findings + evidence),
> [PATH_A_IMPLEMENTATION_PLAN.md](./PATH_A_IMPLEMENTATION_PLAN.md) (plan, implemented),
> [PATH_C_AUDIT.md](./PATH_C_AUDIT.md) (audit + 2026-06 corrections addendum),
> [wandb_analysis/baseline_lite_2026-06-11.md](./wandb_analysis/baseline_lite_2026-06-11.md) (frozen baseline numbers),
> [wandb_analysis/elbofix_run1_health_2026-06-12.md](./wandb_analysis/elbofix_run1_health_2026-06-12.md) (Run-1 verdict).

---

## 1. Core idea & non-negotiables (dissertation)

Hierarchical video prediction with discrete tokens. Three pillars:

1. **Hierarchical processing** — distribute compute across coarse→fine stages
   (coarse shifts 1×, mid 2×, fine 4× per envelope; temporal grids 3/5/9 @ lite).
2. **Categorical prediction over a vocabulary** (not direct vector regression) —
   sharp outputs + **per-stage entropy maps** as a confidence signal. This is the
   reason for the VQ/discrete route and is a real differentiator; keep it.
3. **Reversibility within a stage** — the author's prior published work applies
   RevViT to continuous-in-time prediction (output re-enters as input until step T).
   The predictor's **stream-carry shift mechanism** is the embodiment here: at
   shift k>0 a stage consumes its own (o1, o2) stream states, no token feedback.

Negotiable: the specific quantizer family (SQ-VAE vs FSQ/LFQ). Non-negotiable:
hierarchy + discrete vocab + (eventual) in-stage reversibility.

Stack: fork of SEED-Voken/Open-MAGVIT2. Tokenizer = SQ-VAE-2/HQ-VAE
(papers 2401.00365, 2409.04410). Predictor = custom envelope orchestrator.
Data: synthetic rotating 3D cube assemblies (shapekit), 32×32, T=9, cached.
W&B: `leo27heady/seed-voken-shapes3d`. Machine: 1 GPU (~7 GB used @ BS 48), Windows 11,
RAM-constrained (tests must stay ≤ 12 GiB — enforced, see §6).

## 2. Repo map (the parts that matter)

```
src/Open_MAGVIT2/
  models/video_hier_vqgan.py        # VAE Lightning module (manual opt, EMA, kl_beta, grad clip, dec_ddconfig)
  models/video_hier_predictor.py    # Predictor Lightning module (frozen VAE, CE-only loss)
  modules/vqvae/hierarchical/
    video_inj_topdown.py            # SQVAE2TopDown: native xN + pyramid u2 blocks, temporal_up,
                                    #   decoder_source latent|finest_state, log_param_q_max clamp
    gaussian_sq.py                  # SQ quantizer — KL SUMMED over positions+K (fp32), Path A fix
    hier_elbo_loss.py               # ARELBO distortion dim_x/2*log(MSE) + sum of layer aux losses
    shape_audit.py                  # tap audit + validate_state_chain (pyramid T-chain guard)
    prior_net.py                    # GaussianPriorHead — learned conditional prior (implemented, unused yet)
  modules/predictor/
    schedule.py                     # PyramidSchedule: RF groups, context_end, envelope DFS — verified correct
    orchestrator.py                 # train / parallel(full|context) / autoregressive envelope modes
    stage.py                        # dual-stream stage; rev coupling (full attn) or factorized layers
    masks.py                        # causal self-attn + RF-overlap cross-attn; cross_spatial_window (3x3 fix)
    batch_prep.py                   # tokenize (no_grad) + embed (WITH grad — codebook_proj fix)
    metrics.py                      # CE breakdowns; val tf-aliases; pred-MSE logging-only
    rev_back_prop/                  # custom reversible backprop (opt-in; parent-grad caveat, §8)
scripts/
  token_quality_report.py           # THE GATE: per-layer stats + G1-G4 pass/fail, exit code
  run_tests_with_peak_rss.py        # measured test runs
  fetch_wandb_review.py / analyze_wandb_review.py / analyze_elbofix_run.py
  layer_ablation_review.py / token_stats_review.py / grad_flow_review.py  # review probes
tests/
  conftest.py                       # single-thread torch, KMP guard, gc, 12 GiB RSS watchdog,
                                    #   per-test peak attribution printed at session end
  predictor/stack_factory.py        # memory-safe shared test stacks (smoke VAEs, lite_stack for grad tests)
```

## 3. Verified findings (all empirically confirmed; chronological)

From the 2026-06-11 deep review of baseline runs `idomjiua` (tokenizer) / `wgm3d46g` (predictor):

1. **ELBO scaling bug** — KL was `mean` over (T,H,W,K) instead of summed →
   ~10⁵–10⁶× too weak vs ARELBO distortion (obs.: distortion ≈ 60k, KL ≈ 0.02–23)
   → tokenizer trained as plain AE → codebook collapse (active codes 53/4096,
   140/2048, 76/1024). **FIXED** (sums, fp32 under AMP).
2. **Mid-layer dominance / fine info destroyed** — all levels were down-fused to the
   3×4×4 bottleneck (768 floats/clip). Ablation @ epoch-67 ckpt: randomize L2 →
   MSE 0.0025→0.236; L1 → 0.023; L3 → 0.034. Intended semantics inverted. **FIXED**
   structurally by pyramid + finest_state (Run 2 to confirm empirically).
3. **No encoder→decoder skip connections exist** (user's worry disproved twice at
   code level). `decode_from_indices` uses activations/bottleneck for SHAPES only;
   predictor passes zeros. Decoder = pure function of tokens.
4. **Coarse tokens temporally chaotic** — P(unchanged next step) = 0.028 (vs 0.123 mid,
   0.105 fine); H(next|prev) 3.47 vs marginal 3.83 nats → predictor val coarse CE 3.54
   was already near the information-theoretic bound. Tokenizer, not predictor, is the
   coarse bottleneck.
5. **Dead `codebook_proj`** — blanket `@torch.no_grad` in batch_prep froze the trainable
   96→dim embedding projection at random init (fine stage compressed tokens through a
   frozen RANDOM 96→16 map). **FIXED** + regression test.
6. **val AR CE grew 22→35 monotonically** — finest-stage AR re-entry path
   (re-embed committed tokens, extended ctx, untrained horizon temporal_pos slots —
   verified zero grad) is never trained; stream-carry generalizes fine
   (`val_parallel/ce_s2_k3` = 1.27 vs AR 11.1). **OPEN — Path C work** (deliberate).
7. **`parallel_mode: context` ≡ train mode computation** — val "parallel" metrics were
   train-mode-on-val, not an oracle. Renamed to `val/*_tf` (legacy keys kept 1 release).
8. **`loss_pred_mse` carried zero gradient** (decodes argmax indices through frozen VAE)
   yet was in the loss AND the ckpt monitor; per-pixel keys double-divided. **FIXED**:
   logging-only (`log_pred_mse`, `pred_mse_every_n_steps`), monitor → `val/loss_ce_ar`.

Found during Path A implementation:

9. **Dead cross-attn q/k at the fine stage** — legacy 1-to-1 spatial cross mask gives the
   supervised query frame exactly ONE parent key → softmax over a single entry is
   constant → cross q/k get EXACT zero gradient (verified). **FIXED** via
   `predictor.cross_spatial_window: 1` (3×3 parent neighborhood); canary test keeps the
   legacy behavior documented (`tests/predictor/test_grad_flow.py`).
10. **Pyramid u2 was spatial-only** — naive `token_grid: pyramid` would put all token
    grids at T=3, silently destroying the 1/2/4 shift hierarchy. **FIXED**:
    `temporal_up: [2,2]` → Upsampler (2,2,2), T→2T−1 chain 3→5→9 (matches causal encoder
    exactly); guarded by `validate_state_chain` (hard error otherwise).

From Run 1 (`s1vvokkt`, elbofix, ~8.3k steps, 2026-06-12 — interrupted by disk-full):

11. **ELBO fix verified live**: layers start exactly at −positions·lnK
    (−399/−2438/−15945); KL ≈ −18.8k vs distortion ≈ 85–133k (same order). L1/L2 healthy
    (ppl/K 0.17 / 0.43 vs baseline 0.011/0.020; vars annealing down 60→47/55).
12. **L3 structural posterior collapse on the NATIVE config**: q(z₃) pinned uniform
    (ppl≈K), var₃ runaway 60→119. Mechanism: zero-prior KL = −H rewards entropy; L3's
    recon gradient ≈ 0 (down-fusion, finding #2) → nothing balances it; inflating var
    lowers BOTH KL terms. The honest ELBO correctly reports L3 as informationless in
    this topology. Mitigation: `quantizer.log_param_q_max: 5.0` (var ≤ e⁵≈148) added to
    both run configs. Real fix: Run 2 (finest_state gives L3 the dominant recon
    gradient); fallback: `prior.mode: learned_chain` (implemented, config-only switch).
13. **Test-suite memory catastrophe**: >70 GB RSS (paging, hours, native
    access-violation crashes in MHA softmax). Root cause: test stacks built FULL
    self-attention on the 64px geometry (fine stage 13×32×32 = 13312 tokens → ~2.8 GB
    attn matrix per MHA call; worst with autograd across the 7-shift envelope). **FIXED**
    (see §6): peak now **11.92 GiB**, suite 162 passed in **5:21**.

## 4. Path A implementation (branch `path-a/elbo-pyramid`, UNCOMMITTED as of 2026-06-12)

All phases 0–4 + 7 done and verified; full suite green.

- **ELBO**: KL sums (discrete + continuous, fp32) in `gaussian_sq.py`;
  `loss_cfg.{kl_beta, kl_warmup_steps, grad_clip}` on `VideoHierVQModel`
  (manual optimization ignores trainer `gradient_clip_val` — model clips itself now);
  `train/kl_beta` logged. Old test asserting grid-invariant KL rewritten to assert
  linear scaling. Tests: `tests/test_elbo_scaling.py` (hand-computed KL check,
  magnitude-ratio canary 0.01<|KL|/|distortion|<10 at init, β schedule).
- **Predictor P0**: codebook_proj grad fix (no_grad narrowed to `vae.encode_tokens`);
  pred-MSE logging-only + every-N + out of loss; monitor `val/loss_ce_ar`;
  `val/*_tf` aliases; per-pixel keys dropped. Tests: `test_grad_flow.py`
  (all-params-grad, codebook_proj, ctx temporal_pos, pred-MSE non-grad,
  AR≡train @ k=0, legacy dead-q/k canary).
- **Pyramid**: `InjSQBlock(up_block_size)`; `hierarchy.temporal_up` (validated);
  `hierarchy.decoder_source: latent|finest_state` (forward / forward_progressive /
  decode_from_indices all honor it; progressive partials aligned to finest grid);
  model-level `dec_ddconfig` (e.g. ch_mult [1,2] → one spatial doubling, T preserved);
  `quantizer.log_param_q_max` clamp via `_var_q_slice`. **Token grids identical to
  native mode → predictor code unchanged** (tested). Defaults preserve legacy behavior
  bit-for-bit; old ckpts load. Tests: `tests/test_pyramid_topdown.py` (grids, exact
  roundtrip, decoder shape, schedule invariance, chain guard, clamp, e2e).
- **Configs**: `shapes3d_sqvae2_32_S_lite_elbofix.yaml` (Run 1),
  `..._lite_pyr.yaml` (Run 2), `..._lite_pyr_smoke.yaml` (CPU 2-epoch),
  `..._lite_pyr_predict.yaml` (Phase 6: dims [256,128,64], fine t_window 2,
  cross_spatial_window 1, monitor val/loss_ce_ar; `vae_ckpt` must be set to the
  gated Run-2 best). Codebook sizes kept [4096,2048,1024] in Runs 1–2 for attribution;
  size rebalance = pre-planned Run 3 only if gates show saturation.
- **Gate**: `scripts/token_quality_report.py --config <yaml> --ckpt <best> [--json out]`
  — exit 0 only if all gates pass. Verified to reproduce the baseline numbers exactly
  (and correctly fail G1–G3 on the old ckpt → `wandb_analysis/gate_baseline_lite.json`).

  | Gate | Threshold | Baseline (fails) |
  |---|---|---|
  | G1 perplexity/K | ≥ 0.10 each layer | 0.011/0.020/0.019 |
  | G2 persistence | coarse>mid>fine AND coarse ≥ 0.5 | 0.028/0.123/0.105 (inverted) |
  | G3 ablation ΔMSE order | coarse ≥ mid ≥ fine | mid-dominant |
  | G4 recon MSE/px | ≤ 0.0028 | 0.0025 ✓ |

- **Hygiene**: seeds pinned 1234; dead `objectives/` module removed;
  baseline frozen in `wandb_analysis/baseline_lite_2026-06-11.md`; docs updated
  (README pipeline+commands, PATH_C_README Path-A section, PATH_C_AUDIT addendum
  correcting falsified "Correct" verdicts, hierarchical-vq doc new options).

## 5. Run history (W&B project `leo27heady/seed-voken-shapes3d`)

| Run | Config | Steps/state | Verdict |
|---|---|---|---|
| `idomjiua` | lite (pre-fix) | ep 68, interrupted | baseline; collapsed codebooks; ckpt `epoch=67-step=70856.ckpt` at `C:\Users\leoni\Documents\checkpoints\vqgan\shapes3d_sqvae2_32_S_lite\` |
| `wgm3d46g` | lite_predict (pre-fix) | ep 41, interrupted | baseline predictor; val AR divergence; coarse at marginal-entropy wall |
| `s1vvokkt` | lite_elbofix @50k | ~8.3k steps, killed (disk full) | superseded by pt5j4bxu; mechanics ✓ |
| `pt5j4bxu` | lite_elbofix @**10k** | 74.5k steps / 356 ep | **all 4 gates fail** — overfit (val min 0.0083 @45k then rising; train 0.0017) + L3 dead at var ceiling. Self-annealing works (L1/L2 active codes 330/980); MI improved (L1 1.44 nats vs 0.36 baseline). See `wandb_analysis/elbofix_run1_full_2026-06-12.md`. NOT a usable attribution run — rerun @50k+ES. ⚠ W&B run ids change on restart; same name reused — check `api.runs()` ordering. |

Dataset sizes were temporarily cut (10k/1k) for disk space — this caused the
pt5j4bxu overfit. **Restored to 50000/1000/1000 on 2026-06-12** (reuses original
caches, no new disk). Both run configs now have **EarlyStopping** (val_ema/mse,
patience 10 val-checks). New stabilization option implemented & tested:
`quantizer.temporal_kl_weight: [w...]` — temporal consistency prior
`w*KL(q_t || sg(q_{t-1}))` per layer (off by default; Run-2b values `[0.25,0.1,0.0]`
documented in the pyr config). Gate updates: MI column (H(marg)−H(next|prev));
`--max-recon-mse` default now 0.005 (0.0028 was the KL-free AE's point).
The shape cache lives at `../../data/shape_cache` (relative to repo).

## 6. Test suite (Windows-specific knowledge)

- **Run**: `python -m pytest tests -q` (~5:20, 162 tests) or measured:
  `python scripts/run_tests_with_peak_rss.py tests -q`.
- `tests/conftest.py` does, in order: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `KMP_DUPLICATE_LIB_OK=TRUE` **before importing torch**, `torch.set_num_threads(1)`,
  gc after every test, **12 GiB RSS watchdog** (fails the FIRST offender only —
  CPython retains arenas, so later tests would all appear over budget), and prints a
  **top-10 per-test lifetime-peak attribution table** at session end.
- History: multi-threaded OpenMP + full-attention 64px test stacks → native access
  violations inside MHA softmax + 70 GB paging. Never build full attention on
  >~2.5k-token grids in tests; use `tests/predictor/stack_factory.py`
  (`v2_stack` = no-grad-only 64px smoke stacks, `lite_stack` = 32px grad-safe).
- Current top legitimate peak raisers: GPU e2e test (+4 GB host, CUDA context),
  v2/v4 boolean-mask golden tests (~1.8 GB transient).

## 7. Next steps (ordered)

1. **Commit the branch** (still uncommitted!). Suggested split: "P0 correctness"
   (Phases 0–2) + "pyramid architecture" (Phase 3+) or a single commit.
2. **Free disk space**, restore `dataset_size` (50000/1000) in both run configs.
3. **Run 1 to plateau** (`lite_elbofix`) — attribution baseline; expect G2/G3 to fail
   (by design); L3 will stay collapsed (structural).
4. **Run 2** (`lite_pyr`) — the main event. Gate with `token_quality_report.py`.
   Decision tree: all pass → predictor; G1 fails → Path D (FSQ/LFQ swap);
   only G2 fails → Run 2b with coarse temporal-smoothness KL;
   L3 still near-uniform → flip its prior to `learned_chain` (config-only).
5. **Predictor retrain** (`lite_pyr_predict`, set `vae_ckpt`). Success signals:
   coarse CE breaks below the ~3.5 marginal-entropy wall; judge by `val/*_tf`
   metrics (AR k>0 divergence persists until Path C — known).
6. **Path C (next implementation phase): train what you test.**
   Canonical rollout = stream-carry (drop finest `ar_reinit` OR train re-entry):
   multi-envelope re-entry training with scheduled sampling (teacher-force new context
   from GT with prob p, anneal 1→0.5); dense causal supervision (supervise every
   t→t+1 transition, not just horizon — trains temporal_pos, ~5–8× more signal/step).
7. **Pre-tokenize the dataset** once per gated tokenizer ckpt → predictor epochs at
   pure-transformer speed (the frozen 3D encoder forward dominates step time).

## 8. Improvement backlog (with rationale; roughly by leverage)

- **Learned conditional prior** (`prior.mode: learned_chain`, `GaussianPriorHead` —
  already implemented & reviewed): removes the zero-prior uniform-pull degeneracy
  entirely; HQ-VAE's own remedy. First fallback for any layer collapse.
- **FSQ per level** (Path D): collapse-free by construction; keeps entropy maps;
  decouples "discrete + uncertainty" (the thesis) from "codebook health" (the pain).
  LFQ + entropy penalties is the codebook-flavored alternative (half-wired in
  `quantizer_builder`).
- **RoPE / relative temporal encoding** in the predictor: kills the untrained
  absolute-position failure mode; generalizes to longer rollouts. Do during Path C.
- **Reversible factorized coupling**: `ReversibleCouplingBlock` is residual-fn-agnostic
  → wrap `FactorizedSpaceTimeBlock` as F/G ("factorized_rev" attention type). Gives TRUE
  in-stage reversibility on mid/fine (currently only coarse is reversible; factorized
  streams never interact). Needed if reversibility is a headline dissertation claim.
- **MaskGIT-style iterative within-frame decoding**: fixes conditional independence of
  tokens within a frame (fine for deterministic shapes; required for multimodal real data).
- **Token factorization** (K=4096 → 2×64 subcodes, Open-MAGVIT2 style) if large coarse
  vocabs survive the size ablation.
- **Before publishing any `use_reversible_backprop=True` result**: add a gradcheck
  comparing full-graph vs custom-backward gradients INCLUDING parent stages —
  `backward_pass` either detaches `cross_kv` (drops parent grads) or re-backprops into
  the parent graph per layer per shift (correctness unproven).
- Smaller: per-shift CE weighting (later shifts), EarlyStopping callback (both baseline
  runs died to Ctrl+C with max_epochs 500–2000), bf16 for the predictor once stable,
  Upsampler frame-drop T≡1(mod 4) invariant is guarded only on the pyramid path.

## 9. Quirks / gotchas (save future-you some confusion)

- **Negative KL totals are CORRECT** with the zero prior (term = −H(q) + const);
  layers start at exactly −positions·lnK. Don't "fix" this.
- **ARELBO distortion magnitude ~10⁴–10⁵ is normal** (dim_x/2 · log MSE_sum, dim_x=27648).
- `val_ema/*` lags train heavily early (EMA over the whole module).
- W&B marks Ctrl-C-interrupted runs as "failed" — not crashes. Check `output.log`.
- `seed_everything: 1234` everywhere now (was `true` = random seed per run).
- Predictor runs fp32 internally (`autocast(enabled=False)`) regardless of the
  trainer's 16-mixed; that's deliberate (large-vocab CE stability).
- `val/loss_total_ar` still logged (= λ_ce·CE now) so old dashboards don't break;
  `val/*_parallel` keys are legacy aliases of `val/*_tf` — drop after one release.
- The encoder tap dict is keyed by spatial shape and OVERWRITTEN per level — `h4_w4`
  IS the post-conv_out bottleneck (16ch), by design.
- Predictor stage embeds = frozen copied VAE codebook + trainable `codebook_proj`.
- `simple-shape-dataset-toolbox/` is untracked in git status — the shapekit dataset
  generator submodule-ish directory; don't accidentally commit if it's meant separate.
