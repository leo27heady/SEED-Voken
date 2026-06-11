"""PRED-GPU-01: end-to-end forward with trained VAE checkpoint when available."""

from pathlib import Path

import pytest
import torch

from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

CKPT = Path(
    r"C:\Users\leoni\Documents\checkpoints\vqgan\shapes3d_sqvae2_64_S_v2\epoch=67-step=56712.ckpt"
)


@pytest.mark.skipif(not CKPT.is_file(), reason="trained VAE checkpoint not on disk")
@pytest.mark.gpu
def test_pred_gpu_01_e2e_checkpoint_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = VideoHierPredictorModel(
        vae_config="configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml",
        vae_ckpt=str(CKPT),
        t_context=9,
        t_total=13,
    ).cuda().eval()
    video = torch.randn(1, 3, 13, 64, 64, device="cuda")
    with torch.no_grad():
        loss, out, _ = model.forward_batch(video, inference_mode="train")
    assert torch.isfinite(loss)
    assert torch.isfinite(out.loss_ce)
