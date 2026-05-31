"""Encoder tap shape audit for SQ-VAE-2 configs."""

from typing import Any, Dict, List, Tuple

import torch

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Encoder


def audit_encoder_taps(
    ddconfig: Dict[str, Any],
    sequence_length: int,
    batch_size: int = 1,
    resolution: int = None,
) -> Dict[str, Tuple[int, ...]]:
    """
    Run a dry encoder forward and return tap key -> tensor shape (B, C, T, H, W).
    """
    if sequence_length % 4 != 1:
        raise ValueError(
            f"sequence_length must satisfy T % 4 == 1, got {sequence_length}"
        )
    cfg = dict(ddconfig)
    if resolution is not None:
        cfg["resolution"] = resolution
    enc = Encoder(**cfg)
    enc.eval()
    res = cfg.get("resolution", 128)
    x = torch.zeros(batch_size, 3, sequence_length, res, res)
    with torch.no_grad():
        _, activations = enc(x, return_intermediates=True)
    return {k: tuple(v.shape) for k, v in activations.items()}


def format_tap_audit(audit: Dict[str, Tuple[int, ...]]) -> str:
    lines = ["Encoder intermediate taps:"]
    for key in sorted(audit.keys()):
        shape = audit[key]
        lines.append(f"  {key}: C={shape[1]}, T={shape[2]}, H={shape[3]}, W={shape[4]}")
    return "\n".join(lines)
