# Frozen baseline — `shapes3d_sqvae2_32_S_lite` (pre-Path-A)

Reference numbers for the Path A before/after comparison. Produced 2026-06-11 from:

- tokenizer ckpt `epoch=67-step=70856.ckpt` (W&B `idomjiua`, interrupted at epoch 68/2000, still improving)
- predictor W&B `wgm3d46g` (interrupted at epoch 41/500)
- scripts: `layer_ablation_review.py`, `token_stats_review.py`, `grad_flow_review.py`, `analyze_wandb_review.py`

## Tokenizer

| Metric | L1 h4_w4 (K=4096) | L2 h8_w8 (K=2048) | L3 h16_w16 (K=1024) |
|---|---|---|---|
| train perplexity (end) | 44.5 | 40.6 | 20.5 |
| perplexity / K | 0.011 | 0.020 | 0.020 |
| active codes (train epoch) | 53 | 140 | 76 |
| posterior var (end, still falling) | 0.97 | 2.27 | 3.56 |
| token persistence P(unchanged) | **0.028** | 0.123 | 0.105 |
| H(marginal) nats | 3.83 | 3.71 | 2.96 |
| H(next \| prev @ same pos) | 3.47 | 3.13 | 2.58 |
| ablation Δ-MSE (random codes) | +0.020 | **+0.234** | +0.031 |
| keep-only MSE | 0.305 | **0.053** | 0.286 |
| fused contribution RMS | 0.42 | 1.18 | 0.73 |

- baseline recon MSE (16 val videos, EMA weights): **0.00254**; `val_ema/mse_per_pixel` at interrupt: **0.00276**
- loss magnitudes at interrupt: distortion ≈ **59 000**, kl_total ≈ **23** (pre-fix mean-reduced KL)
- `loss/kl_layer_3_raw` grew 0.02 → 22.9 over training (continuous term vs decaying var)

## Predictor

- `train/loss_ce` 51 → 9.2; per-stage CE end: s0 2.84, s1 1.27, s2 0.96
- token acc end (train): s0 0.17, s1 0.56, s2 0.65; (val AR): s0 0.083, s1 0.52, s2 0.34
- `val/loss_ce_ar` **grew 22.3 → 34.9** (monotonic); `val/loss_ce_parallel` 15.8 → 10.8
- AR divergence localized to finest k>0: `val/ce_s2_k1/k2/k3` → 7.4 / 8.9 / 11.1 vs `val_parallel/ce_s2_k3` = **1.27**
- verified at this commit: `codebook_proj.grad is None` (all stages), horizon `temporal_pos` grad = 0, `loss_pred_mse.requires_grad = False`

## Path A gates (targets vs this baseline)

| Gate | Target | Baseline |
|---|---|---|
| perplexity / K (every layer) | ≥ 0.10 | 0.011 / 0.020 / 0.020 |
| persistence ordering | coarse > mid > fine, coarse ≥ 0.5 | inverted (0.028 / 0.123 / 0.105) |
| ablation Δ-MSE ordering | coarse ≥ mid ≥ fine | mid-dominant |
| `val_ema/mse_per_pixel` | ≤ 0.0027 | 0.00276 |
