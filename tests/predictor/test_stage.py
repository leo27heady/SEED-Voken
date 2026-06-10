"""PredictorStage unit tests."""

import torch

from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule, ShiftMasks
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage


def _stage(s=2, attention_type="full"):
    sched = PyramidSchedule.from_v2_64s()
    spec = sched.stages[s]
    return PredictorStage(
        dim=32, n_heads=4, n_layers=2, codebook_size=spec.codebook_size,
        h=spec.H, w=spec.W, max_t=13, max_shifts=4, has_parent=s > 0,
        attention_type=attention_type, t_window=1,
    ), sched


def _bot_masks(sched, t_len=9):
    builder = PyramidMaskBuilder(sched)
    full = builder.build_self_attn_mask(2, 1, torch.device("cpu"))
    n = t_len * sched.stages[2].n_spatial
    return ShiftMasks(self_attn=full[:n, :n])


def test_stg_03_mid_two_shifts_ce():
    schedule = PyramidSchedule.from_v2_64s()
    stage = PredictorStage(
        dim=32, n_heads=4, n_layers=1, codebook_size=768,
        h=16, w=16, max_t=13, max_shifts=2, has_parent=True,
    )
    ctx = torch.randn(1, 5 * 256, 32)
    builder = PyramidMaskBuilder(schedule)
    n = ctx.shape[1]
    full = builder.build_self_attn_mask(1, 3, ctx.device)
    masks = ShiftMasks(self_attn=full[:n, :n])
    out0 = stage.execute_shift(0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=5)
    out1 = stage.execute_shift(
        1, context_embed=None, stream_state=(out0.o1, out0.o2), parent=None, masks=masks, t_len=5,
    )
    assert out0.logits.shape == out1.logits.shape


def test_stg_05_codebook_match_vae():
    from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
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
    vae = VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4,
    )
    stage = PredictorStage(
        dim=32, n_heads=4, n_layers=1, codebook_size=384,
        h=32, w=32, max_t=13, max_shifts=4, has_parent=True,
    )
    stage.codebook_embed.weight.data.copy_(vae.hier_quant.blocks[2].quantizer.codebook.data)
    assert torch.allclose(stage.codebook_embed.weight, vae.hier_quant.blocks[2].quantizer.codebook)


def test_stg_01_shift_0_init():
    stage, sched = _stage(0)
    ctx = torch.randn(1, 4 * 64, 32)
    builder = PyramidMaskBuilder(sched)
    masks = ShiftMasks(self_attn=builder.build_self_attn_mask(0, -1, ctx.device))
    out = stage.execute_shift(0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=4)
    assert out.o1.shape == out.o2.shape == (1, 4 * 64, 32)


def test_stg_02_shift_1_uses_streams():
    stage, sched = _stage(2)
    ctx = torch.randn(1, 9 * 1024, 32)
    masks = _bot_masks(sched, t_len=9)
    out0 = stage.execute_shift(0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=9)
    out1 = stage.execute_shift(
        1, context_embed=None, stream_state=(out0.o1, out0.o2), parent=None, masks=masks, t_len=9,
    )
    assert out1.logits.shape[0] == 1


def test_stg_04_logits_shape():
    stage, sched = _stage(2)
    ctx = torch.randn(1, 9 * 1024, 32)
    masks = _bot_masks(sched, t_len=9)
    out = stage.execute_shift(0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=9)
    assert out.logits.shape[-1] == sched.stages[2].codebook_size


def test_stg_06_shift_embed_differs():
    stage, sched = _stage(2)
    ctx = torch.randn(1, 9 * 1024, 32)
    masks = _bot_masks(sched, t_len=9)
    out0 = stage.execute_shift(0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=9)
    out1 = stage.execute_shift(
        1, context_embed=None, stream_state=(out0.o1, out0.o2), parent=None, masks=masks, t_len=9,
    )
    assert not torch.allclose(out0.o1, out1.o1)
    assert not torch.allclose(out0.o2, out1.o2)


def test_stg_factorized_forward():
    stage, sched = _stage(2, attention_type="factorized")
    ctx = torch.randn(1, 9 * 1024, 32)
    masks = _bot_masks(sched, t_len=9)
    out = stage.execute_shift(0, context_embed=ctx, stream_state=None, parent=None, masks=masks, t_len=9)
    assert torch.isfinite(out.logits).all()


def test_stg_04_parent_dim_cross_attn_smoke():
    """Child stage dim != parent_dim; cross-attn K/V projection must work."""
    from src.Open_MAGVIT2.modules.predictor.parent_condition import ParentCondition

    sched = PyramidSchedule.from_v4_64s()
    stage = PredictorStage(
        dim=256,
        n_heads=8,
        n_layers=1,
        codebook_size=1024,
        h=8,
        w=8,
        max_t=17,
        max_shifts=8,
        has_parent=True,
        parent_dim=384,
    )
    t_len = 5
    n_sp = 64
    ctx = torch.randn(1, t_len * n_sp, 256)
    parent_o1 = torch.randn(1, 3 * 16, 384)
    parent_o2 = torch.randn(1, 3 * 16, 384)
    parent = ParentCondition.from_shift_output(parent_o1, parent_o2)
    builder = PyramidMaskBuilder(sched)
    self_m = builder.build_self_attn_mask(1, 3, ctx.device)
    n = ctx.shape[1]
    cross_m = builder.build_cross_attn_mask(1, 0, device=ctx.device)
    masks = ShiftMasks(
        self_attn=self_m[:n, :n],
        cross_attn=cross_m[:n, : parent_o1.shape[1]],
    )
    out = stage.execute_shift(
        0, context_embed=ctx, stream_state=None, parent=parent, masks=masks, t_len=t_len,
    )
    assert out.logits.shape == (1, n, 1024)
    assert torch.isfinite(out.logits).all()
