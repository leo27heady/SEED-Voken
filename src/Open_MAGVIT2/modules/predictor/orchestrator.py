"""Envelope orchestrator for multi-shift hierarchical prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.predictor.batch_prep import PreparedBatch
from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder, sanitize_cross_attn_mask
from src.Open_MAGVIT2.modules.predictor.parent_condition import ParentCondition
from src.Open_MAGVIT2.modules.predictor.rollout import Autoregressive, RolloutPolicy, TeacherForced
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
    # CE on the canonical target frame only (last n_sp query positions). Equals
    # loss_ce in single-frame mode; under dense supervision it is the gate-
    # comparable metric (PLAN_V2 §8.1a, _lasttok). Always populated.
    loss_ce_lasttok: Optional[torch.Tensor] = None


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

    def forward_train(self, batch: PreparedBatch) -> PredictorOutput:
        return self._run_envelope(batch, TeacherForced(batch))

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
        return self._run_envelope(aug, TeacherForced(aug))

    def forward_autoregressive(self, batch: PreparedBatch) -> PredictorOutput:
        """Predicted embeds; finest stage re-inits from rolling buffer on k>0."""
        policy = Autoregressive(
            batch, self.schedule.S - 1, sample=self.commit_mode == "sample"
        )
        return self._run_envelope(batch, policy)

    def _run_envelope(
        self, batch: PreparedBatch, policy: RolloutPolicy
    ) -> PredictorOutput:
        """One shared envelope loop; `policy` supplies the mode-varying decisions
        (context source, AR re-entry, token commit). PLAN_V2 §5.1."""
        parent_by_stage: Dict[int, ParentCondition] = {}
        stream_carry: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        logits_out: Dict[Tuple[int, int], torch.Tensor] = {}
        ce_breakdown: Dict[Tuple[int, int], torch.Tensor] = {}
        loss_ce = None
        loss_ce_lasttok = None
        finest_s = self.schedule.S - 1

        for shift in self.schedule.envelope_shifts():
            s, k = shift.stage, shift.k
            reinit = policy.is_reinit(shift)

            parent = None
            if not shift.is_root:
                ps, _pk = shift.parent
                parent = parent_by_stage.get(ps)

            n_sp = self.schedule.stages[s].n_spatial
            if shift.is_first or reinit:
                ctx = policy.initial_context(shift)
                prev = None
            else:
                ctx = None
                prev = stream_carry.get((s, k - 1))
                assert prev is not None

            n_tokens = ctx.shape[1] if ctx is not None else prev[0].shape[1]
            t_len = n_tokens // n_sp

            masks = batch.masks[(s, k)]
            if reinit:
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
            if not reinit:
                stream_carry[(s, k)] = (out.o1, out.o2)
            parent_by_stage[s] = self._parent_from_output(out, s)
            logits_out[(s, k)] = out.logits

            sup = batch.supervision[s][k]
            q_logits = out.logits[:, sup.query_positions]
            tgt = batch.target_indices[s][k]
            ce = self.stages[s].shift_cross_entropy(q_logits, tgt)
            ce_breakdown[(s, k)] = ce.detach()
            w = self._ce_weight(s, k)
            loss_ce = ce * w if loss_ce is None else loss_ce + ce * w

            # Canonical (last) frame slice: under dense supervision it's the tail
            # n_sp transition (p=ctx_end -> tgt_t); identical to `ce` otherwise.
            # Used for commit, AR re-entry, and the gate-comparable `_lasttok` CE.
            q_logits_last = q_logits[:, -n_sp:]
            tgt_last = tgt[:, -n_sp:]
            ce_last = self.stages[s].shift_cross_entropy(q_logits_last, tgt_last)
            loss_ce_lasttok = (
                ce_last * w if loss_ce_lasttok is None else loss_ce_lasttok + ce_last * w
            )

            policy.on_output(shift, self.stages[s], q_logits_last)

        pred_indices = self._collect_pred_indices(batch, logits_out)

        ar_len = policy.final_ar_len(finest_s)
        return PredictorOutput(
            loss_ce=loss_ce if loss_ce is not None else torch.tensor(0.0),
            loss_mse=None,
            logits=logits_out,
            pred_indices=pred_indices,
            ar_context_len=ar_len,
            ce_breakdown=ce_breakdown,
            loss_ce_lasttok=(
                loss_ce_lasttok if loss_ce_lasttok is not None else torch.tensor(0.0)
            ),
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
            n_sp = spec.n_spatial
            for k in range(self.schedule.shifts_per_stage(s)):
                sup = batch.supervision[s][k]
                # canonical target frame = last n_sp query positions (tail under
                # dense supervision; the whole frame otherwise).
                q_logits = logits[(s, k)][:, sup.query_positions][:, -n_sp:]
                pred_tok = self.stages[s].commit_index(
                    q_logits, sample=self.commit_mode == "sample"
                )
                tgt_t = self.schedule.target_token_index(s, k)
                b = idx.shape[0]
                idx[:, tgt_t] = pred_tok.reshape(b, h, w)
            pred[s] = idx
        return pred

