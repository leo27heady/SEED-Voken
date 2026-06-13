<div align="center">
<h1>SEED-Voken — Path C (Shapes3D Hierarchical Video)</h1>
</div>

This repository is scoped to **Path C**: causal hierarchical **SQ-VAE-2** tokenizers on synthetic shape video, plus an **envelope hierarchical predictor** on a frozen VAE.

**Primary docs:**

- [docs/PROJECT_MEMORY.md](./docs/PROJECT_MEMORY.md) — **living context: all findings, decisions, run history, next steps — read first**
- [docs/PATH_C_README.md](./docs/PATH_C_README.md) — training playbook (VAE → predictor)
- [docs/PATH_C_AUDIT.md](./docs/PATH_C_AUDIT.md) — architecture audit & test status
- [docs/Open-MAGVIT2-hierarchical-vq.md](./docs/Open-MAGVIT2-hierarchical-vq.md) — SQ-VAE-2 tokenizer details
- [docs/DEEP_REVIEW_2026-06-11.md](./docs/DEEP_REVIEW_2026-06-11.md) — full-stack review (verified findings)
- [docs/PATH_A_IMPLEMENTATION_PLAN.md](./docs/PATH_A_IMPLEMENTATION_PLAN.md) — Path A plan (implemented on this branch)

## Pipeline

```
tokenizer training  →  token-quality gate  →  predictor training
(main.py fit)          (token_quality_report)   (main.py fit, frozen VAE)
```

The gate is mandatory: predictor work on a tokenizer that fails the gates
wastes GPU days (see the review — the coarse stage cannot beat the marginal
entropy of unstable codes).

## Quick start

```powershell
cd C:\Users\leoni\Documents\projects\SEED-Voken
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path

python -m pytest tests/ -q   # single-threaded torch + 12 GiB RSS watchdog (tests/conftest.py)
# memory-measured run: python scripts/run_tests_with_peak_rss.py tests -q
```

## Configs (Path C only)

```
configs/Open-MAGVIT2/gpu/
  # Path A (current)
  shapes3d_sqvae2_32_S_lite_elbofix.yaml   # Run 1: ELBO fix only (native arch, attribution run)
  shapes3d_sqvae2_32_S_lite_pyr.yaml       # Run 2: ELBO + temporal pyramid + finest-state decoder
  shapes3d_sqvae2_32_S_lite_pyr_smoke.yaml # 2-epoch CPU smoke for the pyramid path
  shapes3d_sqvae2_32_S_lite_pyr_predict.yaml # predictor on the gated Run-2 ckpt

  # legacy baselines (pre-ELBO-fix; kept for comparison)
  shapes3d_sqvae2_32_S_lite.yaml           # 32px fast dev (review baseline)
  shapes3d_sqvae2_32_S_lite_predict.yaml
  shapes3d_sqvae2_64_S_v2.yaml             # 64px VAE, T=13, 3-stage (+smoke, +predict)
  shapes3d_sqvae2_64_S_v4.yaml             # 64px VAE, T=17, 4-stage (+smoke, +predict)
```

## Training (Path A runs)

```powershell
# Run 1 — ELBO fix only (architecture unchanged vs baseline)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_elbofix.yaml

# Run 2 — temporal pyramid + finest-state decoding (includes ELBO fix)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_pyr.yaml

# Gate a checkpoint (exit 0 = all gates pass)
python scripts/token_quality_report.py `
  --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_pyr.yaml `
  --ckpt path\to\best.ckpt --json gate_result.json

# Predictor (set vae_ckpt in the predict yaml to the gated checkpoint first)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_pyr_predict.yaml
```

### Token-quality gates

| Gate | Threshold | Baseline (failing) |
|---|---|---|
| G1 perplexity / K | ≥ 0.10 every layer | 0.011 / 0.020 / 0.019 |
| G2 token persistence | coarse > mid > fine, coarse ≥ 0.5 | 0.028 / 0.123 / 0.105 |
| G3 ablation ΔMSE order | coarse ≥ mid ≥ fine | mid-dominant |
| G4 recon MSE/pixel | ≤ 0.0028 | 0.0025 (passes) |

Baseline numbers: [docs/wandb_analysis/baseline_lite_2026-06-11.md](./docs/wandb_analysis/baseline_lite_2026-06-11.md).

### Predictor metric notes (Path A, Phase 2)

- Checkpoints monitor `val/loss_ce_ar` (pure CE). The pixel `pred_mse` metrics are
  **logging only** — they decode argmax indices through the frozen VAE and carry
  no gradient.
- `val/*_tf` ("teacher-forced") metrics are the new names for the `val/*_parallel`
  metrics; with `parallel_mode: context` they are exactly the training computation
  on val data, not an independent oracle. Legacy `parallel` keys are still logged
  for one release.

## Citation (Open-MAGVIT2)

```
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}
```
