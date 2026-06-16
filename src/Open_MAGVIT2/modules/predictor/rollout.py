"""Rollout policies — the strategy objects that fold the three envelope modes
(train / parallel-oracle / autoregressive) into one shared loop (PLAN_V2 §5.1, B4).

There is exactly ONE hierarchy. The orchestrator owns the *generic* per-shift
mechanics (parent lookup, mask slicing, stream-carry, CE); the policy answers the
handful of questions that actually differ between modes:

  * `is_reinit(shift)`     — is this the finest-stage AR re-entry (k>0)? It drives
                             the full-mask rebuild and skips stream-carry storage.
  * `initial_context(shift)` — the embeds a *fresh* shift (k==0 or reinit) starts
                             from: teacher context (train/parallel) vs the AR
                             rolling buffer.
  * `on_output(shift, stage, q_logits)` — commit the predicted token into the AR
                             rolling buffer (no-op when teacher-forced). This is
                             the AR seam that KV-cache (§7.2) and scheduled
                             sampling (§8.1, the future `ScheduledSampling`
                             subclass) will hook.
  * `final_ar_len(finest_s)` — rolling-buffer length, or None.

`ScheduledSampling` (P3.4b) becomes a subclass of `Autoregressive` overriding
`on_output` to mix GT vs predicted tokens — NOT a second policy hierarchy.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from src.Open_MAGVIT2.modules.predictor.shift import Shift


class RolloutPolicy:
    """Teacher-forced / stream-carry default (covers `train` and `parallel`).

    `batch.context_embed` already holds the right teacher embeds for both: the
    GT context (train / parallel_mode="context") or the full native-T GT embeds
    (parallel_mode="full", augmented by the orchestrator before construction)."""

    def __init__(self, batch) -> None:
        self.batch = batch

    def is_reinit(self, shift: Shift) -> bool:
        return False

    def initial_context(self, shift: Shift) -> torch.Tensor:
        return self.batch.context_embed[shift.stage]

    def on_output(self, shift: Shift, stage, q_logits: torch.Tensor) -> None:
        return None

    def final_ar_len(self, finest_s: int) -> Optional[int]:
        return None


class TeacherForced(RolloutPolicy):
    """Explicit name for the teacher-forced default (train + parallel)."""


class Autoregressive(RolloutPolicy):
    """Predicted re-entry: the finest stage re-inits from a rolling buffer that
    grows by one committed frame per finest shift."""

    def __init__(self, batch, finest_s: int, sample: bool = False) -> None:
        super().__init__(batch)
        self.finest_s = finest_s
        self._sample = sample
        self.rolling_ctx: Dict[int, torch.Tensor] = {
            s: batch.context_embed[s].clone() for s in batch.context_embed
        }

    def is_reinit(self, shift: Shift) -> bool:
        return shift.stage == self.finest_s and shift.k > 0

    def initial_context(self, shift: Shift) -> torch.Tensor:
        return self.rolling_ctx[shift.stage]

    def on_output(self, shift: Shift, stage, q_logits: torch.Tensor) -> None:
        # commit the predicted finest frame and append to the rolling buffer
        # (fires for EVERY finest shift, k==0 included — matches legacy). The
        # stage owns commit semantics (composite argmax vs per-channel compose).
        if shift.stage != self.finest_s:
            return
        pred_tok = stage.commit_index(q_logits, sample=self._sample)
        embed_frame = stage.embed_token_ids(pred_tok)
        self.rolling_ctx[shift.stage] = torch.cat(
            [self.rolling_ctx[shift.stage], embed_frame], dim=1
        )

    def final_ar_len(self, finest_s: int) -> Optional[int]:
        return self.rolling_ctx[finest_s].shape[1]
