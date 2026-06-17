"""Predictor config helpers (per-stage list broadcast) + the frozen
``PredictorConfig`` typed contract (PLAN_V2 §5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.Open_MAGVIT2.modules.predictor.enums import (
    AttentionType,
    CommitMode,
    ParentMode,
    ShiftCEWeights,
)


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


@dataclass(frozen=True)
class StageConfig:
    """Everything needed to build one ``PredictorStage`` (validated once)."""

    dim: int
    n_heads: int
    n_layers: int
    codebook_size: int
    codebook_dim: int
    h: int
    w: int
    has_parent: bool
    parent_dim: Optional[int]
    attention_type: str
    t_window: int
    spatial_window: Optional[int]
    temporal_window: int
    output_mode: str = "composite"
    levels: Optional[Tuple[int, ...]] = None
    pos_encoding: str = "absolute"
    rope_axes: Optional[Tuple[int, int, int]] = None
    rope_base: float = 10000.0
    norm_type: str = "layernorm"
    mlp_type: str = "mlp"
    qk_norm: bool = False

    def __post_init__(self) -> None:
        AttentionType(self.attention_type)  # raises ValueError on unknown type
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"stage dim={self.dim} must be divisible by n_heads={self.n_heads}"
            )
        if self.pos_encoding not in ("absolute", "rope"):
            raise ValueError(f"unknown pos_encoding {self.pos_encoding!r}")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"unknown norm_type {self.norm_type!r}")
        if self.mlp_type not in ("mlp", "swiglu"):
            raise ValueError(f"unknown mlp_type {self.mlp_type!r}")
        if self.output_mode == "factorized_fsq":
            if not self.levels:
                raise ValueError("output_mode='factorized_fsq' requires FSQ 'levels'")
            prod = 1
            for L in self.levels:
                prod *= int(L)
            if prod != self.codebook_size:
                raise ValueError(
                    f"prod(levels)={prod} != codebook_size={self.codebook_size}"
                )
        elif self.output_mode != "composite":
            raise ValueError(f"unknown output_mode {self.output_mode!r}")


@dataclass(frozen=True)
class PredictorConfig:
    """Frozen, validated predictor configuration — the single source of truth
    threaded to the stages, orchestrator, mask cache and batch prep (PLAN_V2 §5.4).

    Built once at model init; ``__post_init__`` consolidates the validation that
    was previously scattered across ``VideoHierPredictorModel.__init__``."""

    stages: Tuple[StageConfig, ...]
    parent_mode: str
    commit_mode: str
    shift_ce_weights: str
    lambda_pred_mse: float
    cross_spatial_window: int
    use_reversible_backprop: bool
    max_t: int
    max_shifts: int

    def __post_init__(self) -> None:
        if ParentMode(self.parent_mode) is ParentMode.NONE:
            raise ValueError(
                f"parent_conditioning must be dual_stream or fused_hard, "
                f"got {self.parent_mode!r}"
            )
        CommitMode(self.commit_mode)
        ShiftCEWeights(self.shift_ce_weights)

    @property
    def temporal_windows(self) -> List[int]:
        return [s.temporal_window for s in self.stages]

    @classmethod
    def from_predictor_cfg(
        cls,
        schedule,
        predictor: Dict[str, Any],
        *,
        t_total: int,
        parent_mode: str,
        commit_mode: str,
        shift_ce_weights: str,
        lambda_pred_mse: float,
        cross_spatial_window: int,
        use_reversible_backprop: bool,
        codebook_dims: Sequence[int],
        output_mode: str = "composite",
        levels_per_stage: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> "PredictorConfig":
        """Resolve the raw ``predictor`` YAML dict into a validated config.

        Mirrors the per-stage resolution that lived inline in
        ``VideoHierPredictorModel.__init__`` (reuses ``stage_list`` /
        ``resolve_stage_attention``) so the produced modules are identical."""
        S = schedule.S
        dims = [int(d) for d in stage_list(predictor.get("dim", 32), S, "dim")]
        layers = [int(n) for n in stage_list(predictor.get("n_layers", 4), S, "n_layers")]
        heads = [int(n) for n in stage_list(predictor.get("n_heads", 8), S, "n_heads")]
        default_windows = [-1, 3, 1] if S == 3 else [-1] * S
        windows = [
            int(w)
            for w in stage_list(
                predictor.get("temporal_windows", default_windows), S, "temporal_windows"
            )
        ]
        attention_cfg = predictor.get("attention", {})
        pos_encoding = str(predictor.get("pos_encoding", "absolute"))
        rope_cfg = predictor.get("rope", {}) or {}
        rope_base = float(rope_cfg.get("base", 10000.0))
        axes_per_stage = rope_cfg.get("axes_per_stage")
        if axes_per_stage is not None and len(axes_per_stage) != S:
            raise ValueError(
                f"predictor.rope.axes_per_stage length {len(axes_per_stage)} != stages {S}"
            )
        norm_type = str(predictor.get("norm_type", "layernorm"))
        mlp_type = str(predictor.get("mlp_type", "mlp"))
        qk_norm = bool(predictor.get("qk_norm", False))
        max_shifts = schedule.ratio ** (S - 1)
        # codebook_dim defaults to the VAE dim_dict (so SQ copies its codebook
        # in-place); an explicit predictor.codebook_dim widens the learned token
        # embedding, useful for FSQ whose value-dim is tiny (e.g. 4) and where no
        # codebook is copied. Do NOT override for SQ (must match the copied codebook).
        cb_dim_cfg = predictor.get("codebook_dim")
        cb_dims = (
            [int(d) for d in stage_list(cb_dim_cfg, S, "codebook_dim")]
            if cb_dim_cfg is not None
            else [int(d) for d in codebook_dims]
        )

        stage_cfgs: List[StageConfig] = []
        for s, spec in enumerate(schedule.stages):
            has_parent = s > 0
            stage_attn = resolve_stage_attention(attention_cfg, s, windows)
            stage_levels = None
            if output_mode == "factorized_fsq" and levels_per_stage is not None:
                lv = levels_per_stage[s]
                stage_levels = tuple(int(x) for x in lv) if lv is not None else None
            stage_rope_axes = None
            if axes_per_stage is not None and axes_per_stage[s] is not None:
                stage_rope_axes = tuple(int(x) for x in axes_per_stage[s])
            stage_cfgs.append(
                StageConfig(
                    dim=dims[s],
                    n_heads=heads[s],
                    n_layers=layers[s],
                    codebook_size=spec.codebook_size,
                    codebook_dim=cb_dims[s],
                    h=spec.H,
                    w=spec.W,
                    has_parent=has_parent,
                    parent_dim=dims[s - 1] if has_parent else None,
                    attention_type=stage_attn.get("type", "full"),
                    t_window=stage_attn.get("t_window", windows[s]),
                    spatial_window=stage_attn.get("spatial_window"),
                    temporal_window=windows[s],
                    output_mode=output_mode,
                    levels=stage_levels,
                    pos_encoding=pos_encoding,
                    rope_axes=stage_rope_axes,
                    rope_base=rope_base,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    qk_norm=qk_norm,
                )
            )
        return cls(
            stages=tuple(stage_cfgs),
            parent_mode=parent_mode,
            commit_mode=commit_mode,
            shift_ce_weights=shift_ce_weights,
            lambda_pred_mse=lambda_pred_mse,
            cross_spatial_window=cross_spatial_window,
            use_reversible_backprop=use_reversible_backprop,
            max_t=t_total,
            max_shifts=max_shifts,
        )
