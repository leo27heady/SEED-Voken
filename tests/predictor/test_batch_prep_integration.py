"""BatchPrep + VAE integration tests."""

import torch

from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from tests.predictor.stack_factory import build_v2_stages, smoke_vae_64


def _build():
    vae = smoke_vae_64()
    schedule = PyramidSchedule.from_v2_64s()
    stages = build_v2_stages(schedule, n_layers=1)
    return vae, schedule, stages


def test_bat_01_encode_keys():
    vae, schedule, stages = _build()
    prep = BatchPrep(schedule, [-1, 3, 1])
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
    assert batch.native_t[0] == 4
    assert batch.native_t[1] == 7
    assert batch.native_t[2] == 13


def test_bat_02_index_shapes():
    vae, schedule, stages = _build()
    prep = BatchPrep(schedule, [-1, 3, 1])
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
    assert batch.gt_indices[2].shape == (1, 13, 32, 32)


def test_bat_03_context_horizon_split():
    vae, schedule, stages = _build()
    prep = BatchPrep(schedule, [-1, 3, 1])
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
    ctx_tokens = batch.context_embed[2].shape[1]
    assert ctx_tokens == (schedule._context_end[2] + 1) * 32 * 32


def test_bat_04_dummy_activations_decode():
    vae, schedule, stages = _build()
    prep = BatchPrep(schedule, [-1, 3, 1])
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        level_indices = [batch.gt_indices[s] for s in range(3)]
        zero_acts = {k: torch.zeros_like(v) for k, v in batch.activations.items()}
        recon = vae.decode_from_indices(
            level_indices, zero_acts, encoder_bottleneck=batch.encoder_bottleneck,
        )
    assert recon.shape == video.shape


def test_bat_05_predictor_init_mock_ckpt(tmp_path):
    import yaml

    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel
    from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel

    cfg_src = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml"
    with open(cfg_src, "r", encoding="utf-8") as f:
        vae_cfg = yaml.safe_load(f)
    vae = VideoHierVQModel(**vae_cfg["model"]["init_args"])
    ckpt = tmp_path / "mock.ckpt"
    torch.save({"state_dict": vae.state_dict()}, ckpt)
    model = VideoHierPredictorModel(
        vae_config=cfg_src,
        vae_ckpt=str(ckpt),
        t_context=9,
        t_total=13,
    )
    assert model.schedule.native_t(2) == 13
