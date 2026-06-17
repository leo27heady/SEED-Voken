"""Shared memory-safe test stacks (Path A follow-up).

The original per-file stacks built FULL self-attention PredictorStages on the
v2 64px geometry (fine stage = 13x32x32 = 13312 tokens -> a (heads, N, N)
attention matrix costs ~2.8 GB fp32 per MHA call, several alive at once under
autograd). Across the suite that pushed peak RSS past 70 GB and into paging.

Rules encoded here:
- encoder/VAE at smoke size (ch=32, num_res_blocks=1) — tests verify plumbing,
  not capacity;
- full attention only on the coarse stage (4x8x8 = 256 tokens);
- mid/fine stages factorized, as in every production config.

Budget: the whole suite must stay under ~12 GB RSS (enforced by the watchdog
fixture in tests/conftest.py).
"""

import torch

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage

V2_STAGE_ATTN = [
    dict(attention_type="full"),
    dict(attention_type="factorized", t_window=3),
    dict(attention_type="factorized", t_window=1, spatial_window=4),
]


def smoke_vae_64() -> VideoHierVQModel:
    """Smoke-sized 64px native 3-level VAE matching the v2 tap layout."""
    ddconfig = dict(
        double_z=False, z_channels=32, resolution=64, in_channels=3, out_ch=3,
        ch=32, ch_mult=[1, 2, 2, 4], num_res_blocks=1, num_groups=8,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="native", tap_key_format="spatial",
        sequence_length=13, latent_key="h8_w8",
        blocks_sq="h8_w8_x1,h16_w16_x1,h32_w32_x1",
        tap_channels=dict(h8_w8=32, h16_w16=64, h32_w32=64),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[1536, 768, 384], dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer,
        learning_rate=1e-4, use_ema=False,
    )


def build_v2_stages(schedule: PyramidSchedule, *, n_layers: int = 2) -> torch.nn.ModuleList:
    return torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=n_layers, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=13, max_shifts=4, has_parent=s > 0,
            **V2_STAGE_ATTN[s],
        )
        for s, spec in enumerate(schedule.stages)
    ])


def v2_stack(*, n_layers: int = 2, lambda_pred_mse: float = 0.0):
    """(vae, prep, orch, stages, schedule) on the v2 64px geometry, smoke-sized.

    Use ONLY for no-grad forwards. Anything that builds an autograd graph
    (backward, or forward_train without torch.no_grad) must use lite_stack():
    a v2-geometry train graph holds ~10+ GB of factorized-attention activations
    across the 7-shift envelope.
    """
    vae = smoke_vae_64()
    schedule = PyramidSchedule.from_v2_64s()
    stages = build_v2_stages(schedule, n_layers=n_layers)
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=lambda_pred_mse)
    return vae, prep, orch, stages, schedule


def smoke_vae_32() -> VideoHierVQModel:
    """Smoke 32px lite VAE (h4/h8/h16 taps, T 3/5/9)."""
    ddconfig = dict(
        double_z=False, z_channels=16, resolution=32, in_channels=3, out_ch=3,
        ch=32, ch_mult=[1, 2, 2, 4], num_res_blocks=1, num_groups=8,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="native", tap_key_format="spatial",
        sequence_length=9, latent_key="h4_w4",
        blocks_sq="h4_w4_x1,h8_w8_x1,h16_w16_x1",
        tap_channels=dict(h4_w4=16, h8_w8=64, h16_w16=64),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[512, 256, 128], dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer,
        learning_rate=1e-4, use_ema=False,
    )


def smoke_vae_32_bigstep() -> VideoHierVQModel:
    """Smoke 32px big-step pyramid VAE: top 1x1@T3 / mid 4x4@T5 / bot 16x16@T9.

    x4 spatial steps (blocks_sq u4) + temporal_downsample to keep T at 9/5/3.
    Exercises the 1x1 GLOBAL top stage end-to-end through encode_tokens /
    decode_from_indices.
    """
    ddconfig = dict(
        double_z=False, z_channels=16, resolution=32, in_channels=3, out_ch=3,
        ch=32, ch_mult=[1, 2, 2, 2, 4, 4],
        temporal_downsample=[False, False, True, False, True],
        num_res_blocks=1, num_groups=8,
    )
    dec_ddconfig = dict(
        double_z=False, z_channels=16, resolution=32, in_channels=3, out_ch=3,
        ch=32, ch_mult=[1, 2], num_res_blocks=1, num_groups=8,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="pyramid", tap_key_format="spatial",
        sequence_length=9, latent_key="h1_w1",
        blocks_sq="h1_w1_x1,h4_w4_u4,h16_w16_u4", temporal_up=[2, 2],
        decoder_source="finest_state", temporal_align_mode="causal",
        tap_channels=dict(h1_w1=16, h4_w4=64, h16_w16=64),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[512, 256, 128], dim_dict=[16, 32, 32],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig, dec_ddconfig=dec_ddconfig, hierarchy=hierarchy,
        quantizer=quantizer, learning_rate=1e-4, use_ema=False,
    )


def bigstep_stack(*, n_layers: int = 2, lambda_pred_mse: float = 0.0,
                  pos_encoding: str = "absolute"):
    """32px big-step stack — top 1x1 (3 tokens) / mid 4x4 / fine 16x16; grad-safe.

    Same fine grid (9x16x16 = 2304 tokens) as lite_stack, so the train graph stays
    well under budget. The coarse stage is a single global token per frame (the
    n_spatial==1 path)."""
    from src.Open_MAGVIT2.modules.predictor.schedule import StageSpec

    vae = smoke_vae_32_bigstep()
    specs = (
        StageSpec("h1_w1", 1, 1, 512, 16),
        StageSpec("h4_w4", 4, 4, 256, 32),
        StageSpec("h16_w16", 16, 16, 128, 32),
    )
    schedule = PyramidSchedule.from_stage_specs(specs, t_context=5, t_total=9)
    stages = torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=n_layers, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=9, max_shifts=4, has_parent=s > 0,
            pos_encoding=pos_encoding,
            **V2_STAGE_ATTN[s],
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=lambda_pred_mse)
    return vae, prep, orch, stages, schedule


def lite_stack(*, n_layers: int = 2, lambda_pred_mse: float = 0.0,
               pos_encoding: str = "absolute"):
    """32px lite-geometry stack — safe for gradient/backward tests.

    Fine stage is 9x16x16 = 2304 tokens; the whole 7-shift train graph stays
    well under 1 GB.
    """
    from src.Open_MAGVIT2.modules.predictor.schedule import StageSpec

    vae = smoke_vae_32()
    specs = (
        StageSpec("h4_w4", 4, 4, 512, 32),
        StageSpec("h8_w8", 8, 8, 256, 32),
        StageSpec("h16_w16", 16, 16, 128, 32),
    )
    schedule = PyramidSchedule.from_stage_specs(specs, t_context=5, t_total=9)
    stages = torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=n_layers, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=9, max_shifts=4, has_parent=s > 0,
            pos_encoding=pos_encoding,
            **V2_STAGE_ATTN[s],
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=lambda_pred_mse)
    return vae, prep, orch, stages, schedule
