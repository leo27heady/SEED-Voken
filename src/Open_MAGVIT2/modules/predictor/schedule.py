"""Pyramid schedule for hierarchical video predictor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def causal_output_length(t_in: int, kernel_t: int = 3, stride_t: int = 2) -> int:
    return (t_in - 1) // stride_t + 1


def causal_receptive_groups(t_in: int, kernel_t: int = 3, stride_t: int = 2) -> List[List[int]]:
    pad = kernel_t - 1
    t_out = causal_output_length(t_in, kernel_t, stride_t)
    groups: List[List[int]] = []
    for p in range(t_out):
        start = p * stride_t - pad
        groups.append(list(range(max(0, start), min(t_in - 1, start + kernel_t - 1) + 1)))
    return groups


@dataclass(frozen=True)
class DownsampleLink:
    kernel_t: int = 3
    stride_t: int = 2
    spatial_stride: int = 2


@dataclass(frozen=True)
class StageSpec:
    spatial_key: str
    H: int
    W: int
    codebook_size: int
    dim_dict: int
    channels: int = 0

    @property
    def n_spatial(self) -> int:
        return self.H * self.W


@dataclass
class ShiftSupervision:
    stage_idx: int
    shift_idx: int
    query_positions: torch.Tensor
    target_positions: torch.Tensor
    context_positions: torch.Tensor
    parent_stage: Optional[int] = None
    parent_shift: Optional[int] = None


@dataclass
class ShiftMasks:
    self_attn: torch.Tensor
    cross_attn: Optional[torch.Tensor] = None


class PyramidSchedule:
    def __init__(
        self,
        stages: Sequence[StageSpec],
        links: Sequence[DownsampleLink],
        *,
        t_context: int,
        t_total: int,
        ratio: Optional[int] = None,
    ) -> None:
        self.stages = tuple(stages)
        self.links = tuple(links)
        self.t_context = int(t_context)
        self.t_total = int(t_total)
        self.ratio = int(ratio if ratio is not None else (links[0].stride_t if links else 2))
        self.S = len(stages)
        if self.S == 0:
            raise ValueError("At least one stage required")
        if len(links) != max(0, self.S - 1):
            raise ValueError(f"Expected {self.S - 1} links, got {len(links)}")

        self._native_t = self._compute_native_lengths(t_total)
        self._rf_groups = self._compute_rf_groups(t_total)
        self._rf_to_frames = self._compose_rf_to_frames()
        self._context_end = self._compute_context_end_indices()

    @property
    def horizon_frames(self) -> int:
        return self.t_total - self.t_context

    def native_t(self, stage_idx: int) -> int:
        return self._native_t[stage_idx]

    def shifts_per_stage(self, stage_idx: int) -> int:
        return self.ratio ** stage_idx

    def required_total_frames(self, t_context: int) -> int:
        finest_shifts = self.ratio ** (self.S - 1)
        return t_context + finest_shifts

    def envelope_order(self) -> List[Tuple[int, int]]:
        order: List[Tuple[int, int]] = []

        def visit(s: int, k: int) -> None:
            order.append((s, k))
            if s < self.S - 1:
                for j in range(self.ratio):
                    visit(s + 1, k * self.ratio + j)

        visit(0, 0)
        return order

    def parent_for(self, stage_idx: int, shift_idx: int) -> Optional[Tuple[int, int]]:
        if stage_idx == 0:
            return None
        return (stage_idx - 1, shift_idx // self.ratio)

    def _compute_native_lengths(self, t_frames: int) -> Tuple[int, ...]:
        lengths = [t_frames]
        for link in self.links:
            lengths.append(causal_output_length(lengths[-1], link.kernel_t, link.stride_t))
        return tuple(reversed(lengths))

    def _compute_rf_groups(self, t_frames: int) -> List[List[set]]:
        stage_grids: List[List[set]] = [[{i} for i in range(t_frames)]]
        grid = t_frames
        for link in self.links:
            parent_groups = causal_receptive_groups(grid, link.kernel_t, link.stride_t)
            child = stage_grids[-1]
            parent: List[set] = []
            for grp in parent_groups:
                acc: set = set()
                for ci in grp:
                    acc.update(child[ci])
                parent.append(acc)
            stage_grids.append(parent)
            grid = len(parent)
        return list(reversed(stage_grids))

    def _compose_rf_to_frames(self) -> List[List[set]]:
        return self._rf_groups

    def _compute_context_end_indices(self) -> List[int]:
        ctx_frames = set(range(self.t_context))
        ends: List[int] = []
        for stage_frames in self._rf_to_frames:
            end = -1
            for ti, frames in enumerate(stage_frames):
                if frames <= ctx_frames:
                    end = ti
            ends.append(end)
        return ends

    def target_token_index(self, stage_idx: int, shift_idx: int) -> int:
        return self._context_end[stage_idx] + shift_idx + 1

    def token_active_for_shift(self, stage_idx: int, token_idx: int, shift_idx: int) -> bool:
        """Token participates in cross-attn for this shift event."""
        ctx_end = self._context_end[stage_idx]
        if token_idx <= ctx_end:
            return True
        tgt = self.target_token_index(stage_idx, shift_idx)
        return token_idx <= tgt

    def rf_frames_for_shift(self, stage_idx: int, token_idx: int, shift_idx: int) -> set:
        """Finest-frame RF for an active token at this shift."""
        if not self.token_active_for_shift(stage_idx, token_idx, shift_idx):
            return set()
        return set(self._rf_to_frames[stage_idx][token_idx])

    def build_shift_supervision(self, stage_idx: int, shift_idx: int) -> ShiftSupervision:
        ctx_end = self._context_end[stage_idx]
        tgt_t = self.target_token_index(stage_idx, shift_idx)
        if tgt_t >= self.native_t(stage_idx):
            raise IndexError(
                f"Target token {tgt_t} out of range for stage {stage_idx} "
                f"(T={self.native_t(stage_idx)})"
            )
        n_sp = self.stages[stage_idx].n_spatial
        # Query: last context frame predicts the next native token (shift +1)
        query_t = min(ctx_end, tgt_t - 1)
        query_positions = torch.arange(query_t * n_sp, (query_t + 1) * n_sp, dtype=torch.long)
        target_positions = torch.arange(tgt_t * n_sp, (tgt_t + 1) * n_sp, dtype=torch.long)
        context_positions = torch.arange(0, (ctx_end + 1) * n_sp, dtype=torch.long)
        parent = self.parent_for(stage_idx, shift_idx)
        p_stage, p_shift = (parent if parent else (None, None))
        return ShiftSupervision(
            stage_idx=stage_idx,
            shift_idx=shift_idx,
            query_positions=query_positions,
            target_positions=target_positions,
            context_positions=context_positions,
            parent_stage=p_stage,
            parent_shift=p_shift,
        )

    @classmethod
    def from_encoder_audit(
        cls,
        ddconfig: dict,
        hierarchy_cfg: dict,
        quantizer_cfg: dict,
        *,
        t_context: int,
        t_total: int,
    ) -> "PyramidSchedule":
        """Build schedule from VAE hierarchy config and quantizer sizes."""
        import re

        from src.Open_MAGVIT2.modules.vqvae.hierarchical.layer_string import parse_blocks_sq
        from src.Open_MAGVIT2.modules.vqvae.hierarchical.tap_keys import normalize_resolution_key

        sizes = quantizer_cfg["size_dict"]
        dims = quantizer_cfg["dim_dict"]
        if isinstance(sizes, int):
            sizes = [sizes]
        if isinstance(dims, int):
            dims = [dims]

        stages: List[StageSpec] = []
        seen: set[str] = set()
        for spec in parse_blocks_sq(hierarchy_cfg["blocks_sq"]):
            key = normalize_resolution_key(spec.resolution_key)
            if key in seen:
                continue
            seen.add(key)
            m = re.match(r"h(\d+)_w(\d+)", key)
            if not m:
                raise ValueError(f"Cannot parse spatial key {key!r}")
            h, w = int(m.group(1)), int(m.group(2))
            idx = len(stages)
            stages.append(StageSpec(key, h, w, sizes[idx], dims[idx]))

        return cls.from_stage_specs(tuple(stages), t_context=t_context, t_total=t_total)

    @classmethod
    def from_v2_64s(cls, t_context: int = 9, t_total: int = 13) -> "PyramidSchedule":
        stages = (
            StageSpec("h8_w8", 8, 8, 1536, 32),
            StageSpec("h16_w16", 16, 16, 768, 32),
            StageSpec("h32_w32", 32, 32, 384, 32),
        )
        links = (DownsampleLink(), DownsampleLink())
        return cls(stages, links, t_context=t_context, t_total=t_total)

    @classmethod
    def from_stage_specs(
        cls,
        stages: Sequence[StageSpec],
        *,
        t_context: int,
        t_total: int,
        kernel_t: int = 3,
        stride_t: int = 2,
        spatial_stride: int = 2,
    ) -> "PyramidSchedule":
        links = tuple(
            DownsampleLink(kernel_t=kernel_t, stride_t=stride_t, spatial_stride=spatial_stride)
            for _ in range(len(stages) - 1)
        )
        return cls(stages, links, t_context=t_context, t_total=t_total, ratio=stride_t)
