"""Performance and correctness tests for attention mask caching."""

from __future__ import annotations

import time
from unittest import mock

import pytest
import torch
import yaml

from src.Open_MAGVIT2.modules.predictor.attention.factorized import (
    FactorizedSpaceTimeBlock,
    build_spatial_window_mask,
)
from src.Open_MAGVIT2.modules.predictor.attention.utils import is_pure_causal_mha_mask
from src.Open_MAGVIT2.modules.predictor.masks import PredictorMaskCache, PyramidMaskBuilder
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule


def _build_spatial_window_mask_ref(
    n_spatial: int,
    spatial_window: int,
    device: torch.device,
) -> torch.Tensor:
    """Reference loop implementation for equivalence checks."""
    side = int(n_spatial**0.5)
    sw = spatial_window
    mask = torch.zeros(n_spatial, n_spatial, dtype=torch.bool, device=device)
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr in range(-sw, sw + 1):
                for dc in range(-sw, sw + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < side and 0 <= nc < side:
                        mask[i, nr * side + nc] = True
    return mask


@pytest.mark.parametrize("n_sp,sw", [(64, 2), (256, 4), (16, 1)])
def test_spatial_window_mask_vectorized_matches_reference(n_sp, sw):
    ref = _build_spatial_window_mask_ref(n_sp, sw, torch.device("cpu"))
    fast = build_spatial_window_mask(n_sp, sw, torch.device("cpu"))
    assert torch.equal(ref, fast)


def test_factorized_block_registers_spatial_mask_buffer():
    block = FactorizedSpaceTimeBlock(
        dim=16, n_heads=4, n_spatial=256, t_window=1, spatial_window=4,
    )
    assert hasattr(block, "spatial_attn_mask")
    assert block.spatial_attn_mask.shape == (256, 256)
    expected = ~build_spatial_window_mask(256, 4, torch.device("cpu"))
    assert torch.equal(block.spatial_attn_mask, expected)


def test_factorized_forward_does_not_rebuild_spatial_mask():
    block = FactorizedSpaceTimeBlock(
        dim=16, n_heads=4, n_spatial=256, t_window=1, spatial_window=4,
    )
    x = torch.randn(2, 3 * 256, 16)
    with mock.patch(
        "src.Open_MAGVIT2.modules.predictor.attention.factorized.build_spatial_window_mask",
        side_effect=AssertionError("mask rebuilt at runtime"),
    ):
        y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_is_pure_causal_mha_mask():
    n = 5
    causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    assert is_pure_causal_mha_mask(causal)
    assert not is_pure_causal_mha_mask(torch.eye(n, dtype=torch.bool))


def test_predictor_mask_cache_matches_builder():
    schedule = PyramidSchedule.from_v2_64s()
    windows = [-1, 3, 1]
    cache = PredictorMaskCache(schedule, windows)
    builder = PyramidMaskBuilder(schedule)
    built = builder.build_all_masks(windows, torch.device("cpu"))
    cached = cache.get_all(torch.device("cpu"))
    assert set(cached.keys()) == set(built.keys())
    for key in built:
        assert torch.equal(cached[key].self_attn, built[key].self_attn)
        if built[key].cross_attn is None:
            assert cached[key].cross_attn is None
        else:
            assert torch.equal(cached[key].cross_attn, built[key].cross_attn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_32_lite_predictor_forward_under_2s():
    """Regression: spatial mask must stay cached (was ~6s/step on mask rebuild)."""
    from pathlib import Path

    import yaml as _yaml

    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_predict.yaml"
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    args = cfg["model"]["init_args"]
    # Avoid loading huge ckpt in CI — random VAE weights suffice for timing.
    args = {**args, "vae_ckpt": None}
    model = VideoHierPredictorModel(**args).cuda().train()
    b = cfg["data"]["init_args"]["batch_size"]
    x = torch.randn(b, 3, 9, 32, 32, device="cuda")

    for _ in range(3):
        with torch.autocast("cuda"):
            model.forward_batch(x, inference_mode="train")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda"):
        model.forward_batch(x, inference_mode="train")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"forward_batch took {elapsed:.2f}s (expected <2s after mask cache)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_spatial_mask_build_is_fast_vectorized():
    device = torch.device("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        build_spatial_window_mask(256, 4, device)
    torch.cuda.synchronize()
    per_call_ms = (time.perf_counter() - t0) / 100 * 1000
    assert per_call_ms < 5.0, f"vectorized mask build {per_call_ms:.2f}ms (expected <5ms)"
