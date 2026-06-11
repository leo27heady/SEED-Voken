"""Diagnose non-finite predictor loss on first training step (32_S_lite)."""

from __future__ import annotations

import torch
from torch.cuda.amp import autocast

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import DataModuleFromConfig
from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel


def _status(t: torch.Tensor, name: str) -> str:
    finite = torch.isfinite(t).all().item()
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"finite={finite} val={t.detach().float().item() if t.numel() == 1 else '...'}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = VideoHierPredictorModel(
        vae_config="configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml",
        vae_ckpt=r"C:\Users\leoni\Documents\checkpoints\vqgan\shapes3d_sqvae2_32_S_lite\epoch=67-step=70856.ckpt",
        freeze_encoder=True,
        t_context=5,
        t_total=9,
        parent_conditioning="dual_stream",
        shift_reentry={"mode": "streams_only"},
        predictor={
            "dim": [256, 64, 16],
            "n_layers": [6, 4, 2],
            "n_heads": [8, 8, 4],
            "temporal_windows": [-1, 3, 1],
            "attention": {
                "coarse": {"type": "full"},
                "mid": {"type": "factorized", "t_window": 3},
                "fine": {"type": "factorized", "t_window": 1, "spatial_window": 4},
            },
        },
        loss={"lambda_ce": 1.0, "lambda_pred_mse": 0.5, "shift_ce_weights": "uniform"},
        inference={"mode": "autoregressive", "commit": "argmax", "parallel_mode": "context"},
        learning_rate=1e-4,
    ).to(device)
    model.train()

    dm = DataModuleFromConfig(
        batch_size=32,
        num_workers=0,
        train={
            "target": "src.Open_MAGVIT2.data.shape_video.ShapeVideoDataset",
            "params": {
                "config": {
                    "size": 32,
                    "mode": "train",
                    "sequence_length": 9,
                    "dataset_size": 64,
                    "cache": True,
                    "shape_scene_type": "DIM_3",
                    "shape_min_cubes": 2,
                    "shape_max_cubes": 6,
                    "shape_angle_min": 15,
                    "shape_angle_max": 45,
                    "shape_temporal_patterns": [],
                    "shape_cache_dir": "../../data/shape_cache",
                }
            },
        },
        validation={
            "target": "src.Open_MAGVIT2.data.shape_video.ShapeVideoDataset",
            "params": {
                "config": {
                    "size": 32,
                    "mode": "validation",
                    "sequence_length": 9,
                    "dataset_size": 16,
                    "cache": True,
                    "shape_scene_type": "DIM_3",
                    "shape_min_cubes": 2,
                    "shape_max_cubes": 6,
                    "shape_angle_min": 15,
                    "shape_angle_max": 45,
                    "shape_temporal_patterns": [],
                    "shape_cache_dir": "../../data/shape_cache",
                }
            },
        },
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    video = batch["video"].to(device)
    print(_status(video, "video"))

    for use_amp in (True,):
        print(f"\n=== per-shift CE amp={use_amp} ===")
        model.zero_grad(set_to_none=True)
        ctx = autocast(dtype=torch.float16) if use_amp and device.type == "cuda" else torch.enable_grad()
        with ctx:
            prep = model.batch_prep.encode_and_schedule(model.vae, video, model.predictor_stages)
            out = model.orchestrator.forward_train(prep)
        import torch.nn.functional as F
        for s, k in model.schedule.envelope_order():
            logits = out.logits[(s, k)]
            sup = prep.supervision[s][k]
            tgt = prep.target_indices[s][k]
            q = logits[:, sup.query_positions].reshape(-1, logits.shape[-1])
            ce = F.cross_entropy(q, tgt.reshape(-1))
            ce32 = F.cross_entropy(q.float(), tgt.reshape(-1))
            k_size = model.schedule.stages[s].codebook_size
            print(
                f"  s={s} k={k} K={k_size} logits_finite={torch.isfinite(q).all().item()} "
                f"ce_fp16={ce.item():.4f} ce_fp32={ce32.item():.4f}"
            )

    for use_amp in (False, True):
        print(f"\n=== forward_batch amp={use_amp} ===")
        model.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with autocast(dtype=torch.float16):
                with autocast(dtype=torch.float16, enabled=False):
                    loss, out, prep = model.forward_batch(video, inference_mode="train")
        else:
            loss, out, prep = model.forward_batch(video, inference_mode="train")
        print(_status(out.loss_ce, "loss_ce"))
        if out.loss_mse is not None:
            print(_status(out.loss_mse, "loss_mse"))
        print(_status(loss, "loss_total"))

        if use_amp and device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            print(f"grad finite after backward: {all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}")
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                print(f"non-finite grads in {len(bad)} params, first: {bad[:5]}")
        else:
            loss.backward()
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f"non-finite grad params: {len(bad)}")
            if bad:
                print(f"first bad: {bad[:5]}")


if __name__ == "__main__":
    main()
