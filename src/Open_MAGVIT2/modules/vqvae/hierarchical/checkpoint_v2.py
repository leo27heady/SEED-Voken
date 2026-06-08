"""Partial checkpoint migration from temporal-key VAE to spatial-key v2."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

CONV_PREFIXES = (
    "encoder.conv_in",
    "encoder.down",
    "encoder.mid",
    "encoder.norm_out",
    "decoder.conv_in",
    "decoder.up",
    "decoder.mid",
    "decoder.conv_out",
)


def load_conv_weights_partial(
    old_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """Copy conv weights that exist in both checkpoints; skip norm/key mismatches."""
    loaded = dict(new_state)
    n_loaded = 0
    n_skipped = 0
    for key, val in old_state.items():
        if not any(key.startswith(p) for p in CONV_PREFIXES):
            n_skipped += 1
            continue
        if "FrameWise" in key or "GroupNorm" in key:
            n_skipped += 1
            continue
        if key not in loaded:
            n_skipped += 1
            continue
        if loaded[key].shape != val.shape:
            n_skipped += 1
            continue
        loaded[key] = val
        n_loaded += 1
    return loaded, n_loaded, n_skipped
