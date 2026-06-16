"""The ``Shift`` value object — one (stage, k) event in an envelope (PLAN_V2 §5.2).

Internally the orchestrator/rollout code reasons about a `Shift` (it carries the
parent pointer and the AR-reinit flag alongside the indices). At the orchestrator
*boundary* — `PredictorOutput.logits` / `ce_breakdown` and every W&B metric key —
we convert back to the plain `Tuple[int, int]` via `as_tuple()` so downstream
code (metrics.py, tests, dashboards) is unchanged. This is the "contract bridge"
that keeps the golden CE snapshot bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Shift:
    stage: int
    k: int
    parent_stage: Optional[int] = None
    parent_k: Optional[int] = None
    is_reinit: bool = False

    @property
    def is_root(self) -> bool:
        """Coarsest stage — no parent conditioning."""
        return self.stage == 0

    @property
    def is_first(self) -> bool:
        """First shift of its stage — initialises from context, not stream-carry."""
        return self.k == 0

    @property
    def prev_k(self) -> int:
        return self.k - 1

    @property
    def parent(self) -> Optional[Tuple[int, int]]:
        if self.parent_stage is None or self.parent_k is None:
            return None
        return (self.parent_stage, self.parent_k)

    def as_tuple(self) -> Tuple[int, int]:
        return (self.stage, self.k)

    def with_reinit(self, flag: bool) -> "Shift":
        """A copy with `is_reinit` set (the AR-reentry flag is policy-decided)."""
        if flag == self.is_reinit:
            return self
        return Shift(self.stage, self.k, self.parent_stage, self.parent_k, flag)
