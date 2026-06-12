"""One-shot wiring check: build the pyr predictor from its yaml (no ckpt) and
run one train-mode forward/backward on random video. Deleted after Phase 5."""
import sys

import torch
import yaml

sys.path.insert(0, ".")

from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

with open("configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_pyr_predict.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["model"]["init_args"]
cfg["vae_ckpt"] = None

torch.manual_seed(0)
model = VideoHierPredictorModel(**cfg)
model.train()
with torch.no_grad():
    for p in model.predictor_stages.parameters():
        if p.requires_grad and p.abs().sum() == 0:
            p.add_(torch.randn_like(p) * 0.02)

video = torch.randn(2, 3, 9, 32, 32).clamp(-1, 1)
loss, out, prep = model.forward_batch(video, inference_mode="train", compute_pred_mse=True)
loss.backward()

print("stages:", [(s.dim, len(s.layers or s.rev_layers)) for s in model.predictor_stages])
print("token grids:", {s: tuple(prep.gt_indices[s].shape[1:]) for s in prep.gt_indices})
print("loss_ce:", round(out.loss_ce.item(), 3), "| pred_mse:", round(out.loss_mse.item(), 5),
      "| mse requires_grad:", out.loss_mse.requires_grad)
proj_grads = [
    (s, m.codebook_proj.weight.grad is not None and m.codebook_proj.weight.grad.abs().sum().item() > 0)
    for s, m in enumerate(model.predictor_stages) if hasattr(m.codebook_proj, "weight")
]
print("codebook_proj grads alive:", proj_grads)
fine = model.predictor_stages[2]
qk = sum(p.grad.abs().sum().item() for n, p in fine.named_parameters()
         if ("cross_q" in n or "cross_k" in n) and p.grad is not None)
print("fine cross q/k grad_abs_sum:", round(qk, 6), "(must be > 0 with cross_spatial_window=1)")
print("AR forward:", end=" ")
with torch.no_grad():
    _, out_ar, _ = model.forward_batch(video, inference_mode="autoregressive", compute_pred_mse=False)
print("ok, ce:", round(out_ar.loss_ce.item(), 3))
print("WIRING OK")
