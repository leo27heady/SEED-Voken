<div align="center">
<h1>SEED-Voken — Path C (Shapes3D Hierarchical Video)</h1>
</div>

This repository is scoped to **Path C**: causal hierarchical **SQ-VAE-2** tokenizers on synthetic shape video, plus an **envelope hierarchical predictor** on a frozen VAE.

**Primary docs:**

- [docs/PATH_C_README.md](./docs/PATH_C_README.md) — training playbook (VAE → predictor)
- [docs/PATH_C_AUDIT.md](./docs/PATH_C_AUDIT.md) — architecture audit & test status
- [docs/Open-MAGVIT2-hierarchical-vq.md](./docs/Open-MAGVIT2-hierarchical-vq.md) — SQ-VAE-2 tokenizer details

## Quick start

```powershell
cd C:\Users\leoni\Documents\projects\SEED-Voken
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path

python -m pytest tests/ -q
python scripts/validate_v2_vae_gate.py
```

## Configs (Path C only)

```
configs/Open-MAGVIT2/gpu/
  shapes3d_sqvae2_64_S_v2.yaml           # 64px VAE, T=13, 3-stage
  shapes3d_sqvae2_64_S_v2_smoke.yaml
  shapes3d_sqvae2_64_S_v2_predict.yaml
  shapes3d_sqvae2_32_S_lite.yaml         # 32px fast dev
  shapes3d_sqvae2_32_S_lite_predict.yaml
  shapes3d_sqvae2_64_S_v4.yaml           # 64px VAE, T=17, 4-stage
  shapes3d_sqvae2_64_S_v4_smoke.yaml
  shapes3d_sqvae2_64_S_v4_predict.yaml
```

## Training

```powershell
# VAE (resume from checkpoint)
.\scripts\resume_vae_training.ps1 -CkptPath "path\to\best.ckpt"

# Predictor (after vae_ckpt is set in predict yaml)
python main.py fit --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2_predict.yaml
```

## Citation (Open-MAGVIT2)

```
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}
```
