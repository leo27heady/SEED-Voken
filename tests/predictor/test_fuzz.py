"""Fuzz tests for schedule and masks."""

import random

import torch

from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
from src.Open_MAGVIT2.modules.predictor.schedule import DownsampleLink, PyramidSchedule, StageSpec


def test_fuz_01_random_valid_schedules():
    random.seed(0)
    for _ in range(8):
        s_count = random.randint(2, 4)
        ratio = random.choice([2, 3])
        t_ctx = random.randint(4, 12)
        horizon = ratio ** (s_count - 1)
        t_total = t_ctx + horizon
        side = 2 ** (s_count + 1)
        stages = tuple(
            StageSpec(f"h{side // 2 ** i}_w{side // 2 ** i}", side // 2 ** i, side // 2 ** i, 128, 16)
            for i in range(s_count)
        )
        links = tuple(DownsampleLink(stride_t=ratio) for _ in range(s_count - 1))
        sched = PyramidSchedule(stages, links, t_context=t_ctx, t_total=t_total, ratio=ratio)
        order = sched.envelope_order()
        expected = {
            (s, k)
            for s in range(s_count)
            for k in range(sched.shifts_per_stage(s))
        }
        assert set(order) == expected


def test_fuz_02_cross_mask_row_sums():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    for s in range(1, sched.S):
        m = builder.build_cross_attn_mask(s, s - 1, device=torch.device("cpu"))
        assert m.sum(dim=1).min() >= 1
