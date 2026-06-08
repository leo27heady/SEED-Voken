"""Shift supervision target tests."""

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule


def _vae():
    ddconfig = dict(
        double_z=False, z_channels=32, resolution=64, in_channels=3, out_ch=3,
        ch=64, ch_mult=[1, 2, 2, 4], num_res_blocks=2,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="native", tap_key_format="spatial",
        sequence_length=13, latent_key="h8_w8",
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h8_w8=32, h16_w16=128, h32_w32=128),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[1536, 768, 384], dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4,
    )


def test_tgt_01_top_shift_0():
    sched = PyramidSchedule.from_v2_64s()
    assert sched.target_token_index(0, 0) == 3


def test_tgt_02_mid_shifts():
    sched = PyramidSchedule.from_v2_64s()
    assert sched.target_token_index(1, 0) == 5
    assert sched.target_token_index(1, 1) == 6


def test_tgt_03_bot_shifts():
    sched = PyramidSchedule.from_v2_64s()
    for k in range(4):
        assert sched.target_token_index(2, k) == 9 + k


def test_tgt_04_no_target_leakage():
    sched = PyramidSchedule.from_v2_64s()
    ctx_frames = set(range(sched.t_context))
    for s in range(sched.S):
        for k in range(sched.shifts_per_stage(s)):
            tgt_t = sched.target_token_index(s, k)
            frames = sched._rf_to_frames[s][tgt_t]
            assert not frames.issubset(ctx_frames)


def test_tgt_05_context_positions_shift_0():
    sched = PyramidSchedule.from_v2_64s()
    sup = sched.build_shift_supervision(2, 0)
    assert sup.context_positions.max().item() == (sched._context_end[2] + 1) * 32 * 32 - 1


def test_tgt_06_reencode_consistency():
    torch.manual_seed(0)
    vae = _vae()
    vae.eval()
    clip = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        a = vae.encode_tokens(clip, flg_quant_det=True)
        b = vae.encode_tokens(clip.clone(), flg_quant_det=True)
    for la, lb in zip(a["levels"], b["levels"]):
        assert torch.equal(la["indices"], lb["indices"])
