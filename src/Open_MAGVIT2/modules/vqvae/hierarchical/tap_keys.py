"""Spatial vs temporal hierarchy tap key helpers."""

from __future__ import annotations

import re
from typing import Dict, Iterable

import torch


_TEMPORAL_KEY_RE = re.compile(r"^t\d+_(h\d+_w\d+)$")
_SPATIAL_KEY_RE = re.compile(r"^(h\d+_w\d+)$")


def spatial_key_from_tensor(h: int, w: int) -> str:
    return f"h{h}_w{w}"


def normalize_resolution_key(key: str) -> str:
    """Return spatial suffix ``h{H}_w{W}`` from spatial or legacy temporal keys."""
    key = key.strip()
    m = _SPATIAL_KEY_RE.match(key)
    if m:
        return key
    m = _TEMPORAL_KEY_RE.match(key)
    if m:
        return m.group(1)
    raise ValueError(
        f"Invalid resolution key {key!r}; expected h{{H}}_w{{W}} or t{{T}}_h{{H}}_w{{W}}"
    )


def resolve_activation_key(
    activations: Dict[str, torch.Tensor],
    resolution_key: str,
    *,
    tap_key_format: str = "spatial",
) -> str:
    """Resolve config key to an activation dict key."""
    spatial = normalize_resolution_key(resolution_key)
    if spatial in activations:
        return spatial
    if tap_key_format == "temporal":
        for k in activations:
            if normalize_resolution_key(k) == spatial:
                return k
    for k in activations:
        if normalize_resolution_key(k) == spatial:
            return k
    available = ", ".join(sorted(activations.keys()))
    raise KeyError(
        f"Activation key for {resolution_key!r} (spatial={spatial!r}) not found. "
        f"Available: {available}"
    )


def spatial_keys_in_audit(audit_keys: Iterable[str]) -> list[str]:
    return sorted({normalize_resolution_key(k) for k in audit_keys})
