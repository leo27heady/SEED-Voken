"""Parent conditioning for cross-attention between predictor stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


ParentMode = Literal["dual_stream", "fused_hard", "none"]


@dataclass
class ParentCondition:
    mode: ParentMode
    o1: Optional[torch.Tensor] = None
    o2: Optional[torch.Tensor] = None
    fused: Optional[torch.Tensor] = None

    @classmethod
    def none(cls) -> "ParentCondition":
        return cls(mode="none")

    @classmethod
    def from_shift_output(
        cls,
        o1: torch.Tensor,
        o2: torch.Tensor,
        *,
        mode: ParentMode = "dual_stream",
        fused: Optional[torch.Tensor] = None,
    ) -> "ParentCondition":
        return cls(mode=mode, o1=o1, o2=o2, fused=fused)
