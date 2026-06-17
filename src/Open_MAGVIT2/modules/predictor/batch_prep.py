"""Batch preparation: encode video and build schedule supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.Open_MAGVIT2.modules.predictor.masks import PredictorMaskCache, PyramidMaskBuilder
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, ShiftMasks, ShiftSupervision


@dataclass
class PreparedBatch:
    gt_indices: Dict[int, torch.Tensor]
    context_embed: Dict[int, torch.Tensor]
    supervision: Dict[int, Dict[int, ShiftSupervision]]
    target_indices: Dict[int, Dict[int, torch.Tensor]]
    masks: Dict[tuple[int, int], ShiftMasks]
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
        mask_cache: PredictorMaskCache | None = None,
        cross_spatial_window: int = 0,
        dense_supervision: bool = False,
    ) -> None:
        self.schedule = schedule
        self.temporal_windows = temporal_windows
        self.mask_builder = PyramidMaskBuilder(
            schedule, cross_spatial_window=cross_spatial_window
        )
        self.mask_cache = mask_cache
        # Dense causal supervision (PLAN_V2 §8.1a): supervise every t->t+1 transition
        # in the context, not just the canonical target frame.
        self.dense_supervision = dense_supervision

    def build_full_context_embed(
        self,
        gt_indices: Dict[int, torch.Tensor],
        stages: torch.nn.ModuleList,
    ) -> Dict[int, torch.Tensor]:
        """Oracle eval: embed full native sequence per stage (context + horizon GT)."""
        return {s: stages[s].embed_indices(gt_indices[s]) for s in gt_indices}

    def encode_and_schedule(self, vae, video: torch.Tensor, stages: torch.nn.ModuleList) -> PreparedBatch:
        """video: (B, C, T, H, W)

        Only the frozen VAE tokenization runs under no_grad. Embedding must stay
        in the grad context: `stage.codebook_proj` is trainable, and a blanket
        @torch.no_grad here silently froze it at random init (review §2.1).
        """
        with torch.no_grad():
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
            idx_flat = idx.reshape(idx.shape[0], -1)
            for k in range(self.schedule.shifts_per_stage(s)):
                sup = self.schedule.build_shift_supervision(
                    s, k, dense=self.dense_supervision
                )
                # gather targets by position (covers both single-frame and dense);
                # for non-dense these are exactly the canonical tgt_t frame's tokens.
                target_indices[s][k] = idx_flat[:, sup.target_positions]
                supervision[s][k] = sup

        if self.mask_cache is not None:
            masks = self.mask_cache.get_all(device)
        else:
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
