"""Lightning module: hierarchical video predictor on frozen SQ-VAE-2 v2."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import lightning as L
import torch
import torch.nn as nn
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage


class VideoHierPredictorModel(L.LightningModule):
    def __init__(
        self,
        vae_config: str,
        vae_ckpt: Optional[str] = None,
        freeze_encoder: bool = True,
        t_context: int = 9,
        t_total: int = 13,
        parent_conditioning: str = "dual_stream",
        shift_reentry: Optional[Dict[str, Any]] = None,
        predictor: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        inference: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        predictor = predictor or {}
        loss = loss or {}
        inference = inference or {}
        shift_reentry = shift_reentry or {"mode": "streams_only"}

        with open(vae_config, "r", encoding="utf-8") as f:
            vae_cfg = yaml.safe_load(f)
        model_cfg = vae_cfg["model"]["init_args"]
        self.vae = VideoHierVQModel(**model_cfg)
        if vae_ckpt:
            ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            self.vae.load_state_dict(state, strict=False)

        if freeze_encoder:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)

        self.t_context = t_context
        self.t_total = t_total
        self.parent_conditioning = parent_conditioning
        self.lambda_pred_mse = float(loss.get("lambda_pred_mse", 0.5))
        self.lambda_ce = float(loss.get("lambda_ce", 1.0))
        self.inference_mode = inference.get("mode", "autoregressive")
        self.commit_mode = inference.get("commit", "argmax")
        self.shift_reentry_mode = shift_reentry.get("mode", "streams_only")
        if self.shift_reentry_mode != "streams_only":
            raise ValueError(f"Unsupported shift_reentry mode: {self.shift_reentry_mode!r}")
        self.lr = learning_rate

        q_cfg = model_cfg["quantizer"]
        sizes = q_cfg["size_dict"]
        dims = q_cfg["dim_dict"]
        if isinstance(sizes, int):
            sizes = [sizes]
        if isinstance(dims, int):
            dims = [dims]

        stages_cfg = PyramidSchedule.from_encoder_audit(
            model_cfg["ddconfig"],
            model_cfg["hierarchy"],
            q_cfg,
            t_context=t_context,
            t_total=t_total,
        )
        self.schedule = stages_cfg

        n_layers = int(predictor.get("n_layers", 4))
        n_heads = int(predictor.get("n_heads", 8))
        dim = int(predictor.get("dim", 32))
        temporal_windows = predictor.get("temporal_windows", [-1, 3, 1])
        attention_cfg = predictor.get("attention", {})
        max_shifts = self.schedule.ratio ** (self.schedule.S - 1)

        pred_stages = nn.ModuleList()
        for s, spec in enumerate(self.schedule.stages):
            has_parent = s > 0
            stage_attn = attention_cfg.get(
                ["coarse", "mid", "fine"][min(s, 2)], {"type": "full"}
            )
            attn_type = stage_attn.get("type", "full")
            t_window = stage_attn.get("t_window", temporal_windows[s])
            spatial_window = stage_attn.get("spatial_window")
            pred_stages.append(
                PredictorStage(
                    dim=dim,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    codebook_size=spec.codebook_size,
                    h=spec.H,
                    w=spec.W,
                    max_t=t_total,
                    max_shifts=max_shifts,
                    has_parent=has_parent,
                    parent_dim=dim,
                    parent_mode=parent_conditioning,
                    attention_type=attn_type,
                    t_window=t_window,
                    spatial_window=spatial_window,
                )
            )
        self.predictor_stages = pred_stages
        self.orchestrator = EnvelopeOrchestrator(
            pred_stages,
            self.schedule,
            parent_mode=parent_conditioning,
            lambda_pred_mse=self.lambda_pred_mse,
            shift_ce_weights=loss.get("shift_ce_weights", "uniform"),
            temporal_windows=temporal_windows,
            commit_mode=self.commit_mode,
        )
        self.batch_prep = BatchPrep(self.schedule, temporal_windows)
        self._copy_codebooks()

    def _copy_codebooks(self) -> None:
        for s, block in enumerate(self.vae.hier_quant.blocks):
            q = block.quantizer
            if hasattr(q, "codebook"):
                self.predictor_stages[s].codebook_embed.weight.data.copy_(q.codebook.data)
        for stage in self.predictor_stages:
            stage.codebook_embed.weight.requires_grad_(False)

    def _decode_horizon_mse(
        self,
        batch,
        pred_indices: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        level_indices = [pred_indices[s] for s in range(self.schedule.S)]
        acts = batch.activations
        zero_acts = {k: torch.zeros_like(v) for k, v in acts.items()}
        recon = self.vae.decode_from_indices(
            level_indices,
            zero_acts,
            encoder_bottleneck=torch.zeros_like(batch.encoder_bottleneck),
        )
        target = batch.video
        horizon = recon[:, :, self.t_context : self.t_total]
        gt = target[:, :, self.t_context : self.t_total]
        return torch.mean((horizon - gt) ** 2)

    def forward_batch(self, video: torch.Tensor, *, inference_mode: Optional[str] = None):
        batch = self.batch_prep.encode_and_schedule(self.vae, video, self.predictor_stages)
        mode = inference_mode or "train"
        if mode == "parallel":
            out = self.orchestrator.forward_parallel(batch)
        elif mode == "autoregressive":
            out = self.orchestrator.forward_autoregressive(batch)
        else:
            out = self.orchestrator.forward_train(batch)
        if self.lambda_pred_mse > 0:
            out.loss_mse = self._decode_horizon_mse(batch, out.pred_indices)
        total = self.lambda_ce * out.loss_ce
        if out.loss_mse is not None:
            total = total + self.lambda_pred_mse * out.loss_mse
        return total, out

    def training_step(self, batch, batch_idx):
        video = batch["video"]
        loss, out = self.forward_batch(video, inference_mode="train")
        self.log("train/loss", loss)
        self.log("train/ce", out.loss_ce)
        if out.loss_mse is not None:
            self.log("train/pred_mse", out.loss_mse)
        return loss

    def validation_step(self, batch, batch_idx):
        video = batch["video"]
        _, out_par = self.forward_batch(video, inference_mode="parallel")
        loss_ar, out_ar = self.forward_batch(video, inference_mode="autoregressive")
        self.log("val/loss_parallel", out_par.loss_ce, prog_bar=True)
        self.log("val/loss_ar", out_ar.loss_ce)
        self.log("val/loss", loss_ar, prog_bar=True)
        return loss_ar

    def configure_optimizers(self):
        params = [p for p in self.orchestrator.parameters() if p.requires_grad]
        params += [p for p in self.predictor_stages.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)
