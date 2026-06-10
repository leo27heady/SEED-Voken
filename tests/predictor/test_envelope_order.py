"""Envelope order and parent ancestry tests."""

from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, StageSpec, DownsampleLink


def test_env_01_exact_order():
    sched = PyramidSchedule.from_v2_64s()
    assert sched.envelope_order() == [
        (0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3),
    ]


def test_env_02_s4_envelope_length():
    stages = tuple(
        StageSpec(f"h{s}_w{s}", 2 ** (3 - s), 2 ** (3 - s), 256, 32)
        for s in range(4)
    )
    links = (DownsampleLink(),) * 3
    sched = PyramidSchedule(stages, links, t_context=9, t_total=25)
    assert len(sched.envelope_order()) == 15


def test_env_03_parent_ancestry():
    sched = PyramidSchedule.from_v2_64s()
    assert sched.parent_for(1, 0) == (0, 0)
    assert sched.parent_for(1, 1) == (0, 0)
    assert sched.parent_for(2, 2) == (1, 1)
    assert sched.parent_for(2, 3) == (1, 1)


def test_env_05_finest_shift_count():
    sched = PyramidSchedule.from_v2_64s()
    assert sched.shifts_per_stage(2) == 4


def test_env_04_v4_envelope_and_ancestry():
    sched = PyramidSchedule.from_v4_64s()
    assert len(sched.envelope_order()) == 15
    assert sched.parent_for(1, 0) == (0, 0)
    assert sched.parent_for(2, 2) == (1, 1)
    assert sched.parent_for(3, 7) == (2, 3)
    assert sched.shifts_per_stage(3) == 8
