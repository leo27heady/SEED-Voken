"""Unit tests for the frozen PredictorConfig contract (PLAN_V2 §5.4)."""

import pytest

from src.Open_MAGVIT2.modules.predictor.config import PredictorConfig, StageConfig
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule

_DEFAULT_PRED = {
    "dim": [64, 32, 32],
    "n_heads": [8, 4, 4],
    "n_layers": [2, 2, 1],
    "temporal_windows": [-1, 3, 1],
    "attention": {
        "per_stage": [
            {"type": "full"},
            {"type": "factorized", "t_window": 3},
            {"type": "factorized", "t_window": 1, "spatial_window": 4},
        ]
    },
}


def _build(predictor=None, **over):
    sched = PyramidSchedule.from_v2_64s()
    kw = dict(
        t_total=13,
        parent_mode="dual_stream",
        commit_mode="argmax",
        shift_ce_weights="uniform",
        lambda_pred_mse=0.0,
        cross_spatial_window=0,
        use_reversible_backprop=False,
        codebook_dims=[32, 32, 32],
    )
    kw.update(over)
    return PredictorConfig.from_predictor_cfg(sched, predictor or _DEFAULT_PRED, **kw)


def test_config_resolves_per_stage_fields():
    cfg = _build()
    assert len(cfg.stages) == 3
    assert cfg.stages[0].attention_type == "full"
    assert cfg.stages[0].has_parent is False and cfg.stages[0].parent_dim is None
    assert cfg.stages[1].has_parent is True and cfg.stages[1].parent_dim == 64
    assert cfg.stages[1].attention_type == "factorized" and cfg.stages[1].t_window == 3
    assert cfg.stages[2].spatial_window == 4
    assert cfg.stages[2].codebook_size == 384 and cfg.stages[2].codebook_dim == 32
    assert cfg.temporal_windows == [-1, 3, 1]
    assert cfg.max_shifts == 4  # ratio ** (S - 1) == 2 ** 2
    assert cfg.max_t == 13


def test_config_rejects_bad_modes():
    with pytest.raises(ValueError):
        _build(parent_mode="none")
    with pytest.raises(ValueError):
        _build(commit_mode="bogus")
    with pytest.raises(ValueError):
        _build(shift_ce_weights="bogus")


def test_config_rejects_indivisible_dim():
    with pytest.raises(ValueError):
        _build(predictor={"dim": [65, 32, 32], "n_heads": [8, 4, 4]})


def test_stageconfig_rejects_bad_attention():
    with pytest.raises(ValueError):
        StageConfig(
            dim=32, n_heads=4, n_layers=1, codebook_size=10, codebook_dim=32,
            h=4, w=4, has_parent=False, parent_dim=None,
            attention_type="bogus", t_window=1, spatial_window=None, temporal_window=-1,
        )
