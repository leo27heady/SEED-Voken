"""Envelope orchestrator for multi-shift hierarchical prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Open_MAGVIT2.modules.predictor.batch_prep import PreparedBatch
from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder, sanitize_cross_attn_mask
from src.Open_MAGVIT2.modules.predictor.parent_condition import ParentCondition
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, ShiftMasks
from src.Open_MAGVIT2.modules.predictor.stage import ShiftOutput


@dataclass
class PredictorOutput:
    loss_ce: torch.Tensor
    loss_mse: Optional[torch.Tensor]
    logits: Dict[Tuple[int, int], torch.Tensor]
    pred_indices: Dict[int, torch.Tensor]
    ar_context_len: Optional[int] = None
    ce_breakdown: Optional[Dict[Tuple[int, int], torch.Tensor]] = None


class EnvelopeOrchestrator(nn.Module):
    def __init__(
        self,
        stages: nn.ModuleList,
        schedule: PyramidSchedule,
        *,
        parent_mode: str = "dual_stream",
        lambda_pred_mse: float = 0.5,
        shift_ce_weights: str = "uniform",
        temporal_windows: Optional[List[int]] = None,
        commit_mode: str = "argmax",
    ) -> None:
        super().__init__()
        self.stages = stages
        self.schedule = schedule
        self.parent_mode = parent_mode
        self.lambda_pred_mse = lambda_pred_mse
        self.shift_ce_weights = shift_ce_weights
        self.temporal_windows = temporal_windows or [-1, 3, 1]
        self.commit_mode = commit_mode
        self.mask_builder = PyramidMaskBuilder(schedule)

    def _parent_from_output(self, out: ShiftOutput, stage_idx: int) -> ParentCondition:
        fused = None
        if self.parent_mode == "fused_hard":
            pred = out.logits.argmax(dim=-1)
            fused = self.stages[stage_idx].embed_token_ids(pred)
        return ParentCondition.from_shift_output(
            out.o1, out.o2, mode=self.parent_mode, fused=fused
        )

    def _ce_weight(self, s: int, k: int) -> float:
        if self.shift_ce_weights == "inverse_shift":
            return 1.0 / (k + 1)
        return 1.0

    def _commit_tokens(self, q_logits: torch.Tensor) -> torch.Tensor:
        if self.commit_mode == "sample":
            probs = F.softmax(q_logits, dim=-1)
            b, n, k = probs.shape
            flat = probs.reshape(b * n, k)
            return torch.multinomial(flat, 1).reshape(b, n)
        return q_logits.argmax(dim=-1)

    def forward_train(self, batch: PreparedBatch) -> PredictorOutput:
        return self._forward_envelope(batch, mode="train")

    def forward_parallel(
        self, batch: PreparedBatch, *, parallel_mode: str = "full"
    ) -> PredictorOutput:
        """Oracle eval: full native-T GT embeds (full) or context-only embeds (context)."""
        if parallel_mode == "full":
            full_ctx = {
                s: self.stages[s].embed_indices(batch.gt_indices[s])
                for s in batch.gt_indices
            }
            aug = replace(batch, context_embed=full_ctx)
        elif parallel_mode == "context":
            aug = batch
        else:
            raise ValueError(f"Unknown parallel_mode: {parallel_mode!r}")
        return self._forward_envelope(aug, mode="parallel")

    def forward_autoregressive(self, batch: PreparedBatch) -> PredictorOutput:
        """Predicted embeds; finest stage re-inits from rolling buffer on k>0."""
        rolling_ctx = {s: batch.context_embed[s].clone() for s in batch.context_embed}
        return self._forward_envelope(
            batch, mode="autoregressive", rolling_ctx=rolling_ctx
        )

    def _forward_envelope(
        self,
        batch: PreparedBatch,
        *,
        mode: str = "train",
        rolling_ctx: Optional[Dict[int, torch.Tensor]] = None,
    ) -> PredictorOutput:
        parent_by_stage: Dict[int, ParentCondition] = {}
        stream_carry: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        logits_out: Dict[Tuple[int, int], torch.Tensor] = {}
        ce_breakdown: Dict[Tuple[int, int], torch.Tensor] = {}
        loss_ce = None
        finest_s = self.schedule.S - 1

        for s, k in self.schedule.envelope_order():
            parent = None
            if s > 0:
                ps, pk = self.schedule.parent_for(s, k)
                assert ps is not None and pk is not None
                parent = parent_by_stage.get(ps)

            n_sp = self.schedule.stages[s].n_spatial
            ar_reinit = (
                mode == "autoregressive"
                and rolling_ctx is not None
                and s == finest_s
                and k > 0
            )
            if k == 0 or ar_reinit:
                if mode == "parallel":
                    ctx = batch.context_embed[s]
                elif rolling_ctx is not None and (k == 0 or ar_reinit):
                    ctx = rolling_ctx[s]
                else:
                    ctx = batch.context_embed[s]
                prev = None
            else:
                ctx = None
                prev = stream_carry.get((s, k - 1))
                assert prev is not None

            n_tokens = ctx.shape[1] if ctx is not None else prev[0].shape[1]
            t_len = n_tokens // n_sp

            masks = batch.masks[(s, k)]
            if ar_reinit:
                device = ctx.device
                full_self = self.mask_builder.build_self_attn_mask(
                    s, self.temporal_windows[s], device
                )
                self_m = full_self[:n_tokens, :n_tokens]
            else:
                self_m = masks.self_attn[:n_tokens, :n_tokens]
            cross_m = None
            if masks.cross_attn is not None:
                parent_n = (
                    parent.o1.shape[1]
                    if parent and parent.o1 is not None
                    else masks.cross_attn.shape[1]
                )
                cross_m = masks.cross_attn[:n_tokens, :parent_n]
                cross_m = sanitize_cross_attn_mask(cross_m)
            shift_masks = ShiftMasks(self_attn=self_m, cross_attn=cross_m)

            out = self.stages[s].execute_shift(
                k,
                context_embed=ctx,
                stream_state=prev,
                parent=parent,
                masks=shift_masks,
                t_len=t_len,
            )
            if not ar_reinit:
                stream_carry[(s, k)] = (out.o1, out.o2)
            parent_by_stage[s] = self._parent_from_output(out, s)
            logits_out[(s, k)] = out.logits

            sup = batch.supervision[s][k]
            q_logits = out.logits[:, sup.query_positions]
            tgt = batch.target_indices[s][k]
            ce = F.cross_entropy(
                q_logits.reshape(-1, q_logits.shape[-1]).float(),
                tgt.reshape(-1),
            )
            ce_breakdown[(s, k)] = ce.detach()
            w = self._ce_weight(s, k)
            loss_ce = ce * w if loss_ce is None else loss_ce + ce * w

            if mode == "autoregressive" and rolling_ctx is not None and s == finest_s:
                pred_tok = self._commit_tokens(q_logits)
                embed_frame = self.stages[s].embed_token_ids(pred_tok)
                rolling_ctx[s] = torch.cat([rolling_ctx[s], embed_frame], dim=1)

        pred_indices = self._collect_pred_indices(batch, logits_out)

        ar_len = rolling_ctx[finest_s].shape[1] if rolling_ctx is not None else None
        return PredictorOutput(
            loss_ce=loss_ce if loss_ce is not None else torch.tensor(0.0),
            loss_mse=None,
            logits=logits_out,
            pred_indices=pred_indices,
            ar_context_len=ar_len,
            ce_breakdown=ce_breakdown,
        )

    def _collect_pred_indices(
        self,
        batch: PreparedBatch,
        logits: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        pred: Dict[int, torch.Tensor] = {}
        for s in range(self.schedule.S):
            idx = batch.gt_indices[s].clone()
            spec = self.schedule.stages[s]
            h, w = spec.H, spec.W
            for k in range(self.schedule.shifts_per_stage(s)):
                sup = batch.supervision[s][k]
                q_logits = logits[(s, k)][:, sup.query_positions]
                pred_tok = self._commit_tokens(q_logits)
                tgt_t = self.schedule.target_token_index(s, k)
                b = idx.shape[0]
                idx[:, tgt_t] = pred_tok.reshape(b, h, w)
            pred[s] = idx
        return pred

