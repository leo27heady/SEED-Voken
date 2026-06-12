"""Verify gradient flow in predictor training path.

Checks:
1. Does codebook_proj receive gradients? (suspect: no, due to @torch.no_grad in batch_prep)
2. Do temporal_pos embeddings beyond the context receive gradients? (suspect: no)
3. Does loss_pred_mse contribute gradient? (suspect: no, indices are discrete)
"""
import sys

import torch

sys.path.insert(0, ".")

from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

torch.manual_seed(0)

model = VideoHierPredictorModel(
    vae_config="configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml",
    vae_ckpt=None,
    freeze_encoder=True,
    t_context=5,
    t_total=9,
    parent_conditioning="dual_stream",
    predictor=dict(
        dim=[256, 64, 16], n_layers=[6, 4, 2], n_heads=[8, 8, 4],
        temporal_windows=[-1, 3, 1],
        attention={"coarse": {"type": "full"},
                   "mid": {"type": "factorized", "t_window": 3},
                   "fine": {"type": "factorized", "t_window": 1, "spatial_window": 4}},
    ),
    loss=dict(lambda_ce=1.0, lambda_pred_mse=0.5, shift_ce_weights="uniform"),
    inference=dict(mode="autoregressive", commit="argmax", parallel_mode="context"),
)
model.train()

# Break the zero-init so gradients can flow past the output head and residual
# branches (mimics the state after a few optimizer steps).
with torch.no_grad():
    for p in model.predictor_stages.parameters():
        if p.requires_grad and p.abs().sum() == 0:
            p.add_(torch.randn_like(p) * 0.02)

video = torch.randn(2, 3, 9, 32, 32).clamp(-1, 1)
loss, out, prep = model.forward_batch(video, inference_mode="train")
print(f"loss_total={loss.item():.4f} loss_ce={out.loss_ce.item():.4f} "
      f"loss_mse={out.loss_mse.item() if out.loss_mse is not None else None}")
print("loss_mse requires_grad:", out.loss_mse.requires_grad if out.loss_mse is not None else None)
loss.backward()

for s, stage in enumerate(model.predictor_stages):
    ctx_end = model.schedule._context_end[s]
    cb = stage.codebook_proj
    cb_grad = None
    if hasattr(cb, "weight"):
        cb_grad = None if cb.weight.grad is None else cb.weight.grad.abs().sum().item()
    ip_grad = None if stage.input_proj.weight.grad is None else stage.input_proj.weight.grad.abs().sum().item()
    tp = stage.temporal_pos.grad
    tp_ctx = None if tp is None else tp[0, : ctx_end + 1].abs().sum().item()
    tp_horizon = None if tp is None else tp[0, ctx_end + 1 :].abs().sum().item()
    print(f"stage {s}: ctx_end={ctx_end} | codebook_proj grad={cb_grad} | "
          f"input_proj grad={ip_grad:.4f} | temporal_pos grad ctx={tp_ctx} horizon={tp_horizon}")
