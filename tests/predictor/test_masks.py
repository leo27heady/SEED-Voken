"""Tests for PyramidMaskBuilder."""

import hashlib
import json

import torch

from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, StageSpec


def test_msk_01_self_causal():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    m = builder.build_self_attn_mask(2, 1, torch.device("cpu"))
    n = sched.native_t(2) * sched.stages[2].n_spatial
    assert m.shape == (n, n)
    assert m[0, 0]
    if n > 1:
        assert not m[0, sched.stages[2].n_spatial]


def test_msk_02_bot_window_block_diagonal():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    m = builder.build_self_attn_mask(2, 1, torch.device("cpu"))
    n_sp = sched.stages[2].n_spatial
    t_len = sched.native_t(2)
    for ti in range(t_len):
        for tj in range(t_len):
            block = m[ti * n_sp : (ti + 1) * n_sp, tj * n_sp : (tj + 1) * n_sp]
            if ti == tj:
                assert block.any()
            else:
                assert not block.any()


def test_msk_03_cross_spatial_ratio():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    m = builder.build_cross_attn_mask(
        2, 1, child_shift=0, parent_shift=0, device=torch.device("cpu")
    )
    assert m.shape[0] == sched.native_t(2) * sched.stages[2].n_spatial
    active = m[sched._context_end[2] * sched.stages[2].n_spatial :]
    assert active.sum(dim=1).min() >= 1


def test_msk_04_mid_temporal_overlap():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    m = builder.build_cross_attn_mask(1, 0, child_shift=0, parent_shift=0, device=torch.device("cpu"))
    n_sp_c = sched.stages[1].n_spatial
    n_sp_p = sched.stages[0].n_spatial
    tc = 2
    child_frames = sched.rf_frames_for_shift(1, tc, 0)
    assert child_frames.intersection({1, 2}) or child_frames.intersection({2, 3})
    attended_parents = set()
    row = m[tc * n_sp_c : (tc + 1) * n_sp_c].any(dim=0)
    for tp in range(sched.native_t(0)):
        if row[tp * n_sp_p : (tp + 1) * n_sp_p].any():
            attended_parents.add(tp)
    assert len(attended_parents) >= 1


def test_msk_06_shift_aware_cross_differs():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    m_early = builder.build_cross_attn_mask(
        2, 1, child_shift=1, parent_shift=0, device=torch.device("cpu")
    )
    m_late = builder.build_cross_attn_mask(
        2, 1, child_shift=3, parent_shift=1, device=torch.device("cpu")
    )
    assert m_early.sum() != m_late.sum()


def test_msk_05_top_one_by_one():
    stages = (
        StageSpec("h1_w1", 1, 1, 64, 16),
        StageSpec("h2_w2", 2, 2, 64, 16),
    )
    sched = PyramidSchedule.from_stage_specs(stages, t_context=5, t_total=9)
    builder = PyramidMaskBuilder(sched)
    m = builder.build_cross_attn_mask(1, 0, child_shift=0, parent_shift=0, device=torch.device("cpu"))
    n_c = sched.stages[1].n_spatial
    n_p = sched.stages[0].n_spatial
    for tc in range(sched.native_t(1)):
        child_frames = sched.rf_frames_for_shift(1, tc, 0)
        if not child_frames:
            continue
        for tp in range(sched.native_t(0)):
            parent_frames = sched.rf_frames_for_shift(0, tp, 0)
            if child_frames.isdisjoint(parent_frames):
                continue
            block = m[tc * n_c : (tc + 1) * n_c, tp * n_p : (tp + 1) * n_p]
            assert block.all(), f"1x1 parent should fully connect at tc={tc} tp={tp}"


def test_msk_07_differs_from_same_frame_only():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    full = builder.build_cross_attn_mask(2, 1, child_shift=0, parent_shift=0, device=torch.device("cpu"))
    n_sp_c = sched.stages[2].n_spatial
    n_sp_p = sched.stages[1].n_spatial
    same_frame = torch.zeros_like(full)
    for tc in range(sched.native_t(2)):
        for tp in range(sched.native_t(1)):
            if tc == tp:
                same_frame[
                    tc * n_sp_c : (tc + 1) * n_sp_c, tp * n_sp_p : (tp + 1) * n_sp_p
                ] = True
    assert not torch.equal(full, same_frame)


def test_msk_08_golden_cross_mask_hash():
    sched = PyramidSchedule.from_v2_64s()
    builder = PyramidMaskBuilder(sched)
    masks = {}
    for s in range(1, sched.S):
        for k in range(sched.shifts_per_stage(s)):
            pk = sched.parent_for(s, k)
            assert pk is not None
            ps, pshift = pk
            m = builder.build_cross_attn_mask(
                s, ps, child_shift=k, parent_shift=pshift, device=torch.device("cpu")
            )
            masks[f"{s}_{k}"] = m.cpu().tolist()
    payload = json.dumps(masks, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    golden = "4ed97172d6d1a44eb7be237f39152a2c3eed4ed256f28873abeaff9b35afce11"
    assert digest == golden
