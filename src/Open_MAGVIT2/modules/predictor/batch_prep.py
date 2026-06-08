"""Batch preparation: encode video and build schedule supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, ShiftMasks, ShiftSupervision


@dataclass
class PreparedBatch:
    gt_indices: Dict[int, torch.Tensor]
    context_embed: Dict[int, torch.Tensor]
    supervision: Dict[int, Dict[int, ShiftSupervision]]
    target_indices: Dict[int, Dict[int, torch.Tensor]]
    masks: Dict[int, Dict[int, ShiftMasks]]
    native_t: Dict[int, int]
    stream_carry: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    activations: Optional[dict] = None
    encoder_bottleneck: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None


class BatchPrep:
    def __init__(
        self,
        schedule: PyramidSchedule,
        temporal_windows: List[int],
    ) -> None:
        self.schedule = schedule
        self.temporal_windows = temporal_windows
        self.mask_builder = PyramidMaskBuilder(schedule)

    @torch.no_grad()
    def build_full_context_embed(
        self,
        gt_indices: Dict[int, torch.Tensor],
        stages: torch.nn.ModuleList,
    ) -> Dict[int, torch.Tensor]:
        """Oracle eval: embed full native sequence per stage (context + horizon GT)."""
        return {s: stages[s].embed_indices(gt_indices[s]) for s in gt_indices}

    @torch.no_grad()
    def encode_and_schedule(self, vae, video: torch.Tensor, stages: torch.nn.ModuleList) -> PreparedBatch:
        """video: (B, C, T, H, W)"""
        tokens = vae.encode_tokens(video, flg_quant_det=True)
        device = video.device
        gt_indices: Dict[int, torch.Tensor] = {}
        context_embed: Dict[int, torch.Tensor] = {}
        supervision: Dict[int, Dict[int, ShiftSupervision]] = {}
        target_indices: Dict[int, Dict[int, torch.Tensor]] = {}
        native_t: Dict[int, int] = {}

        for s, level in enumerate(tokens["levels"]):
            idx = level["indices"]
            gt_indices[s] = idx
            native_t[s] = idx.shape[1]
            ctx_end = self.schedule._context_end[s]
            ctx_idx = idx[:, : ctx_end + 1]
            embed = stages[s].embed_indices(ctx_idx)
            context_embed[s] = embed
            supervision[s] = {}
            target_indices[s] = {}
            for k in range(self.schedule.shifts_per_stage(s)):
                sup = self.schedule.build_shift_supervision(s, k)
                tgt_t = self.schedule.target_token_index(s, k)
                n_sp = self.schedule.stages[s].n_spatial
                flat_tgt = idx[:, tgt_t].reshape(idx.shape[0], n_sp)
                target_indices[s][k] = flat_tgt
                supervision[s][k] = sup

        masks = self.mask_builder.build_all_masks(self.temporal_windows, device)
        return PreparedBatch(
            gt_indices=gt_indices,
            context_embed=context_embed,
            supervision=supervision,
            target_indices=target_indices,
            masks=masks,
            native_t=native_t,
            activations=tokens["activations"],
            encoder_bottleneck=tokens["encoder_bottleneck"],
            video=video,
        )
