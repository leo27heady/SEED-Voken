"""Per-stage predictor config tests (PRD-01)."""

import pytest

from src.Open_MAGVIT2.modules.predictor.config import resolve_stage_attention, stage_list


def test_prd_01_stage_list_broadcast():
    assert stage_list(32, 3, "dim") == [32, 32, 32]
    assert stage_list([384, 256, 128], 3, "dim") == [384, 256, 128]


def test_prd_01_stage_list_length_mismatch():
    with pytest.raises(ValueError, match="predictor.dim length"):
        stage_list([384, 256], 3, "dim")


def test_prd_01_legacy_attention_keys():
    cfg = {
        "coarse": {"type": "full"},
        "mid": {"type": "factorized", "t_window": 3},
        "fine": {"type": "factorized", "t_window": 1, "spatial_window": 8},
    }
    windows = [-1, 3, 1]
    assert resolve_stage_attention(cfg, 0, windows)["type"] == "full"
    assert resolve_stage_attention(cfg, 1, windows)["t_window"] == 3
    assert resolve_stage_attention(cfg, 2, windows)["spatial_window"] == 8
    assert resolve_stage_attention(cfg, 3, windows)["type"] == "factorized"


def test_prd_01_per_stage_attention():
    cfg = {
        "per_stage": [
            {"type": "full"},
            {"type": "full"},
            {"type": "factorized", "t_window": 3, "spatial_window": 8},
            {"type": "factorized", "t_window": 1, "spatial_window": 8},
        ],
    }
    windows = [-1, 3, 3, 1]
    assert resolve_stage_attention(cfg, 3, windows)["t_window"] == 1
    assert resolve_stage_attention(cfg, 3, windows)["spatial_window"] == 8


def test_prd_01_predictor_init_v4_smoke():
    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

    model = VideoHierPredictorModel(
        vae_config="configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v4_smoke.yaml",
        t_context=9,
        t_total=17,
        predictor={
            "dim": [384, 256, 192, 128],
            "n_layers": [2, 2, 1, 1],
            "n_heads": [8, 8, 8, 4],
            "temporal_windows": [-1, 3, 3, 1],
            "attention": {
                "per_stage": [
                    {"type": "full"},
                    {"type": "full"},
                    {"type": "factorized", "t_window": 3, "spatial_window": 8},
                    {"type": "factorized", "t_window": 1, "spatial_window": 8},
                ],
            },
        },
        loss={"lambda_pred_mse": 0.0},
    )
    assert model.schedule.S == 4
    assert model.predictor_stages[0].dim == 384
    assert model.predictor_stages[1].dim == 256
    assert model.predictor_stages[1].rev_layers is not None
    assert model.predictor_stages[2].dim == 192
    assert model.predictor_stages[3].attention_type == "factorized"


def test_prd_03_fsq_tokenizer_predictor_build():
    """Predictor must build on an FSQ tokenizer config (levels only, no
    size_dict/dim_dict). Regression for the FSQ->predictor wiring gap."""
    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

    model = VideoHierPredictorModel(
        vae_config="configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr_smoke.yaml",
        t_context=5,
        t_total=9,
        predictor={
            "dim": [64, 32, 32],
            "n_layers": [1, 1, 1],
            "n_heads": [4, 4, 4],
            "temporal_windows": [-1, 3, 1],
            "codebook_dim": [16, 16, 16],  # widen vs the FSQ value-dim (3)
            "attention": {
                "per_stage": [
                    {"type": "full"},
                    {"type": "factorized", "t_window": 3},
                    {"type": "factorized", "t_window": 1, "spatial_window": 4},
                ],
            },
        },
        loss={"lambda_pred_mse": 0.0},
    )
    assert model.schedule.S == 3
    # K = prod(levels); smoke levels are [4,4,4] per stage -> 64 each
    assert [model.predictor_stages[s].codebook_size for s in range(3)] == [64, 64, 64]
    assert model.predictor_stages[0].codebook_dim == 16  # override applied
    # FSQ has no learned codebook -> _copy_codebooks skips -> embed stays trainable
    assert model.predictor_stages[0].codebook_embed.weight.requires_grad


def test_prd_02_codebook_proj_decoupled_dim():
    import torch

    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

    model = VideoHierPredictorModel(
        vae_config="configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v4_smoke.yaml",
        t_context=9,
        t_total=17,
        predictor={
            "dim": [384, 256, 192, 128],
            "n_layers": [1, 1, 1, 1],
            "n_heads": [8, 8, 8, 4],
            "temporal_windows": [-1, 3, 3, 1],
        },
        loss={"lambda_pred_mse": 0.0},
    )
    stage = model.predictor_stages[0]
    assert stage.codebook_dim == 32
    assert stage.dim == 384
    assert stage.codebook_embed.weight.shape == (256, 32)
    idx = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    embed = stage.embed_indices(idx)
    assert embed.shape == (1, 8, 384)
    assert torch.allclose(
        stage.codebook_embed.weight,
        model.vae.hier_quant.blocks[0].quantizer.codebook.data,
    )
