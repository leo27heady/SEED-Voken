"""Unit tests for the Shift value object + mode enums (PLAN_V2 §5.2/§5.3)."""

import pytest

from src.Open_MAGVIT2.modules.predictor.enums import (
    AttentionType,
    CommitMode,
    ParallelMode,
    ParentMode,
    ShiftCEWeights,
)
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.predictor.shift import Shift


def test_enum_str_equality_and_yaml_compat():
    # str-mixin: members compare equal to raw strings (YAML stays valid).
    assert ParentMode.DUAL_STREAM == "dual_stream"
    assert CommitMode.ARGMAX == "argmax"
    assert ShiftCEWeights.INVERSE_SHIFT == "inverse_shift"
    assert AttentionType.FACTORIZED == "factorized"
    assert ParallelMode.CONTEXT == "context"
    # constructable-from-string (validation entry point)
    assert ParentMode("fused_hard") is ParentMode.FUSED_HARD


def test_enum_rejects_unknown_value():
    for enum in (ParentMode, CommitMode, ShiftCEWeights, AttentionType, ParallelMode):
        with pytest.raises(ValueError):
            enum("not_a_real_mode")


def test_shift_properties():
    root = Shift(stage=0, k=0)
    assert root.is_root and root.is_first
    assert root.parent is None
    assert root.as_tuple() == (0, 0)

    child = Shift(stage=2, k=3, parent_stage=1, parent_k=1)
    assert not child.is_root and not child.is_first
    assert child.prev_k == 2
    assert child.parent == (1, 1)
    assert child.as_tuple() == (2, 3)

    assert child.with_reinit(True).is_reinit
    assert child.with_reinit(False) is child  # no-op returns same instance
    assert child.with_reinit(True).as_tuple() == (2, 3)


def test_envelope_shifts_matches_envelope_order():
    for sched in (PyramidSchedule.from_v2_64s(), PyramidSchedule.from_v4_64s()):
        order = sched.envelope_order()
        shifts = sched.envelope_shifts()
        assert [sh.as_tuple() for sh in shifts] == order
        for sh in shifts:
            assert sh.parent == sched.parent_for(sh.stage, sh.k)
            assert sh.is_reinit is False
