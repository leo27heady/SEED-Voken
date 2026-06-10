"""Predictor config helpers (per-stage list broadcast)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union


def stage_list(
    cfg_val: Union[int, float, Sequence[Any]],
    num_stages: int,
    name: str,
) -> List[Any]:
    """Broadcast a scalar config value to all stages or validate list length."""
    if isinstance(cfg_val, (list, tuple)):
        if len(cfg_val) != num_stages:
            raise ValueError(
                f"predictor.{name} length {len(cfg_val)} != stages {num_stages}"
            )
        return list(cfg_val)
    return [cfg_val] * num_stages


def resolve_stage_attention(
    attention_cfg: Dict[str, Any],
    stage_idx: int,
    temporal_windows: List[int],
) -> Dict[str, Any]:
    """Resolve attention config for one stage (per_stage list or legacy coarse/mid/fine)."""
    per_stage = attention_cfg.get("per_stage")
    if per_stage is not None:
        if not isinstance(per_stage, (list, tuple)):
            raise ValueError("predictor.attention.per_stage must be a list")
        if stage_idx >= len(per_stage):
            tw = temporal_windows[stage_idx]
            return {
                "type": "factorized",
                "t_window": tw,
                "spatial_window": 8,
            }
        entry = per_stage[stage_idx]
        return dict(entry) if isinstance(entry, dict) else {"type": "full"}

    legacy_keys = ["coarse", "mid", "fine"]
    key = legacy_keys[min(stage_idx, len(legacy_keys) - 1)]
    return dict(attention_cfg.get(key, {"type": "full"}))
