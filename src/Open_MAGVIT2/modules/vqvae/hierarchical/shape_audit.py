"""Encoder tap shape audit for SQ-VAE-2 configs."""

from typing import Any, Dict, List, Tuple

import torch

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Encoder
from src.Open_MAGVIT2.modules.vqvae.hierarchical.tap_keys import (
    normalize_resolution_key,
    spatial_keys_in_audit,
)


def audit_encoder_taps(
    ddconfig: Dict[str, Any],
    sequence_length: int,
    batch_size: int = 1,
    resolution: int = None,
) -> Dict[str, Tuple[int, ...]]:
    """
    Run a dry encoder forward and return spatial tap key -> tensor shape (B, C, T, H, W).
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
    lines = ["Encoder intermediate taps (spatial keys):"]
    for key in sorted(audit.keys()):
        shape = audit[key]
        lines.append(f"  {key}: C={shape[1]}, T={shape[2]}, H={shape[3]}, W={shape[4]}")
    return "\n".join(lines)


def validate_hierarchy_taps(
    ddconfig: Dict[str, Any],
    sequence_length: int,
    resolution_keys: List[str],
    tap_channels: Dict[str, int] = None,
    tap_key_format: str = "spatial",
) -> Dict[str, Tuple[int, ...]]:
    """Fail fast if ``blocks_sq`` keys are absent from the encoder tap dict."""
    audit = audit_encoder_taps(ddconfig, sequence_length)
    normalized_keys = [normalize_resolution_key(k) for k in resolution_keys]
    missing = [k for k in normalized_keys if k not in audit]
    if missing:
        raise ValueError(
            "Hierarchy tap keys not produced by the encoder for this ddconfig / "
            f"sequence_length={sequence_length}. Missing: {missing}. "
            f"Available: {sorted(audit.keys())}.\n"
            f"{format_tap_audit(audit)}"
        )
    if tap_channels:
        for key, ch in tap_channels.items():
            nk = normalize_resolution_key(key)
            if nk in audit and audit[nk][1] != ch:
                raise ValueError(
                    f"tap_channels[{key!r}]={ch} but encoder produces C={audit[nk][1]}"
                )
    return audit


def validate_state_chain(
    audit: Dict[str, Tuple[int, ...]],
    resolution_keys: List[str],
    layer_upsample: List[bool],
    temporal_up: List[int],
) -> None:
    """Pyramid (u2) guard: the top-down z_state grid must land exactly on each
    tap grid, otherwise tokens leave the native grids and the predictor's
    temporal shift hierarchy silently breaks (Path A correction #1).

    Upsampler semantics: spatial x2 per u2; temporal factor 2 yields
    T -> 2T - 1 (frame-drop), factor 1 keeps T.
    """
    if not any(layer_upsample):
        return
    first = normalize_resolution_key(resolution_keys[0])
    t, h, w = audit[first][2], audit[first][3], audit[first][4]
    u2_idx = 0
    for key, up in zip(resolution_keys, layer_upsample):
        nk = normalize_resolution_key(key)
        if up:
            t_factor = temporal_up[u2_idx]
            u2_idx += 1
            t = 2 * t - 1 if t_factor == 2 else t
            h, w = h * 2, w * 2
        expected = (t, h, w)
        got = tuple(audit[nk][2:])
        if expected != got:
            raise ValueError(
                f"z_state grid {expected} does not match encoder tap {nk} grid {got}. "
                f"For taps that halve T per level set hierarchy.temporal_up: "
                f"[2, ...] (one entry per u2 layer); current temporal_up has "
                f"factor(s) {temporal_up}."
            )


def audit_spatial_taps_at_lengths(
    ddconfig: Dict[str, Any],
    sequence_lengths: List[int],
    resolution: int = None,
) -> Dict[int, Dict[str, Tuple[int, ...]]]:
    """Audit spatial keys at multiple clip lengths (prefix-stability checks)."""
    return {
        t: audit_encoder_taps(ddconfig, t, resolution=resolution)
        for t in sequence_lengths
    }
