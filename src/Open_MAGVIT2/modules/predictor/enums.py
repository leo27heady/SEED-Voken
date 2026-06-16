"""String-valued enums for predictor mode strings (PLAN_V2 §5.3).

Each is a ``str, Enum`` mixin, so members compare equal to their raw string
(``ParentMode.DUAL_STREAM == "dual_stream"`` is ``True``) and YAML configs that
carry the plain strings keep parsing unchanged. The enums exist to give
``PredictorConfig.__post_init__`` a single, typed validation point —
``ParentMode(value)`` raises ``ValueError`` on an unknown string, replacing the
scattered ``if mode not in (...)`` checks.
"""

from __future__ import annotations

from enum import Enum


class ParentMode(str, Enum):
    DUAL_STREAM = "dual_stream"
    FUSED_HARD = "fused_hard"
    NONE = "none"


class CommitMode(str, Enum):
    ARGMAX = "argmax"
    SAMPLE = "sample"


class ShiftCEWeights(str, Enum):
    UNIFORM = "uniform"
    INVERSE_SHIFT = "inverse_shift"


class AttentionType(str, Enum):
    FULL = "full"
    FACTORIZED = "factorized"


class ParallelMode(str, Enum):
    FULL = "full"
    CONTEXT = "context"
