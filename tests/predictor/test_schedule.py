"""Tests for PyramidSchedule."""

import pytest
import torch

from src.Open_MAGVIT2.modules.predictor.schedule import (
    DownsampleLink,
    PyramidSchedule,
    StageSpec,
    causal_receptive_groups,
)


def test_env_01_envelope_order_s3():
    sched = PyramidSchedule.from_v2_64s()
    order = sched.envelope_order()
    assert order == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3)]


def test_env_04_coverage():
    sched = PyramidSchedule.from_v2_64s()
    order = sched.envelope_order()
    expected = set()
    for s in range(sched.S):
        for k in range(sched.shifts_per_stage(s)):
            expected.add((s, k))
    assert set(order) == expected


def test_rf_01_bot_mid_groups_t13():
    groups = causal_receptive_groups(13, 3, 2)
    assert len(groups) == 7
    assert groups[0] == [0]
    assert groups[4] == [6, 7, 8]


def test_rf_07_required_total_frames():
    sched = PyramidSchedule.from_v2_64s(t_context=9, t_total=13)
    assert sched.required_total_frames(9) == 13


def test_rf_03_native_t():
    sched = PyramidSchedule.from_v2_64s()
    assert sched.native_t(0) == 4
    assert sched.native_t(1) == 7
    assert sched.native_t(2) == 13


def test_rf_02_mid_top_groups():
    groups = causal_receptive_groups(7, 3, 2)
    assert len(groups) == 4
    assert groups[0] == [0]
    assert groups[-1] == [4, 5, 6]


def test_rf_04_compose_corner_cases():
    sched = PyramidSchedule.from_v2_64s()
    rf = sched._rf_to_frames
    assert 6 in rf[1][4]
    assert 8 in rf[2][8]


def test_rf_06_ratio_3_synthetic():
    stages = (
        StageSpec("h4_w4", 4, 4, 128, 16),
        StageSpec("h8_w8", 8, 8, 128, 16),
    )
    links = (DownsampleLink(stride_t=3),)
    sched = PyramidSchedule(stages, links, t_context=6, t_total=15, ratio=3)
    assert sched.shifts_per_stage(1) == 3
    assert len(sched.envelope_order()) == 4


def test_rf_08_invalid_t():
    stages = (StageSpec("h8_w8", 8, 8, 128, 16),)
    sched = PyramidSchedule(stages, (), t_context=9, t_total=13)
    with pytest.raises(IndexError):
        sched.build_shift_supervision(0, 5)


def test_rf_05_rf_to_frames():
    sched = PyramidSchedule.from_v2_64s()
    assert 0 in sched._rf_to_frames[2][0]
    assert max(sched._rf_to_frames[0][3]) >= 7


def test_rf_09_one_by_one_top():
    stages = (
        StageSpec("h1_w1", 1, 1, 64, 16),
        StageSpec("h2_w2", 2, 2, 64, 16),
    )
    sched = PyramidSchedule.from_stage_specs(stages, t_context=5, t_total=9)
    from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
    builder = PyramidMaskBuilder(sched)
    m = builder.build_self_attn_mask(0, -1, torch.device("cpu"))
    assert m.shape == (sched.native_t(0), sched.native_t(0))


def test_rf_10_s4_envelope():
    stages = tuple(
        StageSpec(f"h{s}_w{s}", 2 ** (4 - s), 2 ** (4 - s), 128, 16) for s in [8, 4, 2, 1]
    )
    stages = (
        StageSpec("h8_w8", 8, 8, 128, 16),
        StageSpec("h16_w16", 16, 16, 128, 16),
        StageSpec("h32_w32", 32, 32, 128, 16),
        StageSpec("h64_w64", 64, 64, 128, 16),
    )
    links = (DownsampleLink(),) * 3
    sched = PyramidSchedule(stages, links, t_context=9, t_total=25)
    assert len(sched.envelope_order()) == 15


def test_from_encoder_audit_v2():
    dd = dict(
        double_z=False, z_channels=32, resolution=64, in_channels=3, out_ch=3,
        ch=64, ch_mult=[1, 2, 2, 4], num_res_blocks=2,
    )
    hierarchy = dict(
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
    )
    quant = dict(size_dict=[1536, 768, 384], dim_dict=[32, 32, 32])
    sched = PyramidSchedule.from_encoder_audit(
        dd, hierarchy, quant, t_context=9, t_total=13
    )
    assert sched.native_t(2) == 13
