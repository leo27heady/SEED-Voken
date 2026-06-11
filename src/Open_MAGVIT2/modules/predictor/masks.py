"""Attention masks for hierarchical video predictor."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, ShiftMasks


def sanitize_cross_attn_mask(cross_m: torch.Tensor | None) -> torch.Tensor | None:
    """Ensure each child row has at least one parent connection (avoids softmax NaN)."""
    if cross_m is None:
        return None
    if cross_m.any(dim=1).all():
        return cross_m
    safe = cross_m.clone()
    empty = ~safe.any(dim=1)
    safe[empty, 0] = True
    return safe


class PyramidMaskBuilder:
    def __init__(self, schedule: PyramidSchedule) -> None:
        self.schedule = schedule

    def build_self_attn_mask(
        self,
        stage_idx: int,
        temporal_window: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tokens = self.schedule.native_t(stage_idx)
        n_sp = self.schedule.stages[stage_idx].n_spatial
        if temporal_window == -1:
            temporal_window = t_tokens
        frame_i = torch.arange(t_tokens, device=device)
        diff = frame_i.unsqueeze(1) - frame_i.unsqueeze(0)
        frame_mask = (diff >= 0) & (diff < temporal_window)
        token_mask = frame_mask.repeat_interleave(n_sp, dim=0).repeat_interleave(n_sp, dim=1)
        return token_mask

    def build_cross_attn_mask(
        self,
        child_stage: int,
        parent_stage: int,
        *,
        child_shift: int = 0,
        parent_shift: int = 0,
        device: torch.device,
    ) -> torch.Tensor:
        if parent_stage != child_stage - 1:
            raise ValueError("Only adjacent stage cross-attn supported")
        t_c = self.schedule.native_t(child_stage)
        t_p = self.schedule.native_t(parent_stage)
        n_c = self.schedule.stages[child_stage].n_spatial
        n_p = self.schedule.stages[parent_stage].n_spatial
        sp_ratio = self.schedule.stages[child_stage].H // self.schedule.stages[parent_stage].H

        child_rf = self.schedule._rf_to_frames[child_stage]
        parent_rf = self.schedule._rf_to_frames[parent_stage]

        mask = torch.zeros(t_c * n_c, t_p * n_p, dtype=torch.bool, device=device)
        for tc in range(t_c):
            child_frames = self.schedule.rf_frames_for_shift(
                child_stage, tc, child_shift
            )
            if not child_frames:
                continue
            for tp in range(t_p):
                parent_frames = self.schedule.rf_frames_for_shift(
                    parent_stage, tp, parent_shift
                )
                if not parent_frames or child_frames.isdisjoint(parent_frames):
                    continue
                spatial = self._spatial_block(sp_ratio, n_c, n_p, device)
                mask[tc * n_c : (tc + 1) * n_c, tp * n_p : (tp + 1) * n_p] = spatial
        return sanitize_cross_attn_mask(mask)

    @staticmethod
    def _spatial_block(sp_ratio: int, n_c: int, n_p: int, device: torch.device) -> torch.Tensor:
        side_c = int(n_c ** 0.5)
        side_p = int(n_p ** 0.5)
        if side_p == 1:
            return torch.ones(n_c, n_p, dtype=torch.bool, device=device)
        r = torch.arange(side_c, device=device)
        c = torch.arange(side_c, device=device)
        grid_r, grid_c = torch.meshgrid(r, c, indexing="ij")
        parent_r = grid_r // sp_ratio
        parent_c = grid_c // sp_ratio
        parent_idx = (parent_r * side_p + parent_c).reshape(n_c)
        block = torch.zeros(n_c, n_p, dtype=torch.bool, device=device)
        block[torch.arange(n_c, device=device), parent_idx] = True
        return block

    def build_all_masks(
        self,
        temporal_windows: Sequence[int],
        device: torch.device,
    ) -> dict[tuple[int, int], ShiftMasks]:
        out: dict[tuple[int, int], ShiftMasks] = {}
        for s in range(self.schedule.S):
            for k in range(self.schedule.shifts_per_stage(s)):
                self_m = self.build_self_attn_mask(s, temporal_windows[s], device)
                cross_m = None
                if s > 0:
                    parent = self.schedule.parent_for(s, k)
                    assert parent is not None
                    ps, pk = parent
                    cross_m = self.build_cross_attn_mask(
                        s, ps, child_shift=k, parent_shift=pk, device=device
                    )
                out[(s, k)] = ShiftMasks(self_attn=self_m, cross_attn=cross_m)
        return out


class PredictorMaskCache(nn.Module):
    """Pre-baked envelope masks; moved with the model, no per-step rebuild."""

    def __init__(
        self,
        schedule: PyramidSchedule,
        temporal_windows: Sequence[int],
    ) -> None:
        super().__init__()
        self._keys: list[tuple[int, int]] = []
        self._has_cross: dict[tuple[int, int], bool] = {}
        builder = PyramidMaskBuilder(schedule)
        all_masks = builder.build_all_masks(temporal_windows, torch.device("cpu"))
        for (s, k), masks in sorted(all_masks.items()):
            self._keys.append((s, k))
            self.register_buffer(f"self_{s}_{k}", masks.self_attn)
            if masks.cross_attn is not None:
                self.register_buffer(f"cross_{s}_{k}", masks.cross_attn)
                self._has_cross[(s, k)] = True

    def get_all(self, device: torch.device) -> dict[tuple[int, int], ShiftMasks]:
        out: dict[tuple[int, int], ShiftMasks] = {}
        for s, k in self._keys:
            self_m = getattr(self, f"self_{s}_{k}").to(device)
            cross_m = None
            if self._has_cross.get((s, k), False):
                cross_m = getattr(self, f"cross_{s}_{k}").to(device)
            out[(s, k)] = ShiftMasks(self_attn=self_m, cross_attn=cross_m)
        return out
