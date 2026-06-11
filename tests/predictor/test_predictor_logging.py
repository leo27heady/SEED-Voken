"""Predictor logging metrics, log_images, and horizon callback smoke tests."""

from unittest.mock import MagicMock

import pytest
import torch

from src.Open_MAGVIT2.callbacks.predictor_horizon_logger import PredictorHorizonLogger
from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.metrics import (
    aggregate_ce_by_stage,
    build_train_log_dict,
    build_val_log_dict,
    ce_baseline,
    token_accuracy_from_output,
    total_loss,
)
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage
from src.Open_MAGVIT2.utils.video_viz import predictor_rows_to_grid


def _build_stack():
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
    schedule = PyramidSchedule.from_v2_64s()
    stages = torch.nn.ModuleList([
        PredictorStage(
            dim=32, n_heads=4, n_layers=2, codebook_size=spec.codebook_size,
            h=spec.H, w=spec.W, max_t=13, max_shifts=4, has_parent=s > 0,
        )
        for s, spec in enumerate(schedule.stages)
    ])
    prep = BatchPrep(schedule, [-1, 3, 1])
    orch = EnvelopeOrchestrator(stages, schedule, lambda_pred_mse=0.5)
    return vae, prep, orch, stages, schedule


class _MockModel:
    def __init__(self, schedule):
        self.schedule = schedule
        self.t_context = 9
        self.t_total = 13
        self.lambda_ce = 1.0
        self.lambda_pred_mse = 0.5
        self.trainer = None

    def optimizers(self):
        opt = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4)
        return opt


def test_metrics_total_loss_and_baseline():
    schedule = PyramidSchedule.from_v2_64s()
    ce = torch.tensor(5.0)
    mse = torch.tensor(0.1)
    loss = total_loss(ce, mse, lambda_ce=1.0, lambda_pred_mse=0.5)
    assert loss.item() == pytest.approx(5.05)
    baseline = ce_baseline(schedule)
    assert baseline > 0


def test_aggregate_ce_by_stage():
    ce_breakdown = {(0, 0): torch.tensor(1.0), (1, 0): torch.tensor(2.0), (1, 1): torch.tensor(4.0)}
    agg = aggregate_ce_by_stage(ce_breakdown)
    assert agg[0].item() == pytest.approx(1.0)
    assert agg[1].item() == pytest.approx(3.0)


def test_token_accuracy_from_output():
    torch.manual_seed(0)
    vae, prep, orch, stages, schedule = _build_stack()
    vae.eval()
    video = torch.randn(2, 3, 13, 64, 64)
    batch = prep.encode_and_schedule(vae, video, stages)
    out = orch.forward_train(batch)
    acc = token_accuracy_from_output(out, batch, schedule)
    assert len(acc) == schedule.S
    for s, a in acc.items():
        assert 0.0 <= a.item() <= 1.0


def test_build_train_log_dict_keys():
    torch.manual_seed(1)
    vae, prep, orch, stages, schedule = _build_stack()
    vae.eval()
    video = torch.randn(2, 3, 13, 64, 64)
    batch = prep.encode_and_schedule(vae, video, stages)
    batch.video = video
    out = orch.forward_train(batch)
    mock = _MockModel(schedule)
    loss = total_loss(out.loss_ce, out.loss_mse, lambda_ce=1.0, lambda_pred_mse=0.5)
    log = build_train_log_dict(mock, out, batch, loss_total=loss)
    assert "train/loss_ce" in log
    assert "train/loss_total" not in log
    assert "train/ce_over_baseline" in log
    assert any(k.startswith("train/token_acc_stage_") for k in log)
    assert any(k.startswith("train/ce_stage_") for k in log)


def test_build_val_log_dict_keys():
    torch.manual_seed(2)
    vae, prep, orch, stages, schedule = _build_stack()
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    batch = prep.encode_and_schedule(vae, video, stages)
    batch.video = video
    out_ar = orch.forward_train(batch)
    out_par = orch.forward_train(batch)
    mock = _MockModel(schedule)
    loss_ar = total_loss(out_ar.loss_ce, None, lambda_ce=1.0, lambda_pred_mse=0.5)
    log = build_val_log_dict(
        mock,
        out_ar,
        batch,
        loss_total_ar=loss_ar,
        out_par=out_par,
        parallel_skipped=False,
    )
    assert "val/loss_total_ar" not in log
    assert "val/loss_total_parallel" in log
    assert "val/ce_gap_parallel_minus_ar" in log
    assert any(k.startswith("val/token_acc_ar_stage_") for k in log)

    log_skip = build_val_log_dict(
        mock, out_ar, batch, loss_total_ar=loss_ar, out_par=None, parallel_skipped=True,
    )
    assert log_skip["val/parallel_skipped"].item() == 1.0


def test_log_images_smoke(tmp_path):
    import yaml
    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel
    from src.Open_MAGVIT2.modules.predictor.orchestrator import PredictorOutput

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
    model.eval()
    video = torch.rand(1, 3, 13, 64, 64) * 2 - 1
    batch = {"video": video}

    def _fake_decode(prep, pred_indices):
        b = prep.video.shape[0]
        return torch.zeros(b, 3, 13, 64, 64)

    def _fake_out(prep):
        return PredictorOutput(
            loss_ce=torch.tensor(0.0),
            loss_mse=None,
            logits={},
            pred_indices={s: prep.gt_indices[s] for s in prep.gt_indices},
            ar_context_len=None,
            ce_breakdown={},
        )

    model._decode_video_from_indices = _fake_decode
    model.orchestrator.forward_train = lambda prep: _fake_out(prep)
    model.orchestrator.forward_parallel = lambda prep, **kw: _fake_out(prep)
    model.orchestrator.forward_autoregressive = lambda prep: _fake_out(prep)

    with torch.no_grad():
        log = model.log_images(batch)
    expected = {"inputs", "vae_recon", "pred_train", "pred_parallel", "pred_ar"}
    assert set(log.keys()) == expected
    for k, v in log.items():
        assert v.shape == (1, 3, 13, 64, 64), k


def test_predictor_rows_to_grid():
    log = {
        "inputs": torch.randn(1, 3, 13, 64, 64),
        "vae_recon": torch.randn(1, 3, 13, 64, 64),
        "pred_train": torch.randn(1, 3, 13, 64, 64),
        "pred_parallel": torch.randn(1, 3, 13, 64, 64),
        "pred_ar": torch.randn(1, 3, 13, 64, 64),
    }
    png = predictor_rows_to_grid(log, t_context=9, t_total=13, cell_size=32)
    assert png.size[0] > 0 and png.size[1] > 0


def test_predictor_horizon_logger_smoke(tmp_path):
    video = torch.rand(1, 3, 13, 64, 64) * 2 - 1
    batch = {"video": video}

    model = MagicMock()
    model.eval = MagicMock()
    model.t_context = 9
    model.t_total = 13
    model.log_images.return_value = {
        "inputs": video,
        "vae_recon": video,
        "pred_train": video,
        "pred_parallel": video,
        "pred_ar": video,
    }

    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.default_root_dir = str(tmp_path / "run")
    trainer.logger = None
    trainer.datamodule = MagicMock()
    trainer.datamodule.val_dataloader.return_value = iter([batch])

    cb = PredictorHorizonLogger(max_samples=1, log_wandb=False)
    cb.on_validation_epoch_end(trainer, model)

    out_dir = tmp_path / "run" / "predictions"
    assert out_dir.is_dir()
    assert list(out_dir.glob("epoch0000_sample0.png"))
