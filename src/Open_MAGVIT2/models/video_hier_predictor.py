"""Lightning module: hierarchical video predictor on frozen SQ-VAE-2 v2."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import lightning as L
import torch
import torch.nn as nn
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.predictor.batch_prep import BatchPrep
from src.Open_MAGVIT2.modules.predictor.masks import PredictorMaskCache
from src.Open_MAGVIT2.modules.predictor.config import resolve_stage_attention, stage_list
from src.Open_MAGVIT2.modules.predictor.metrics import (
    build_train_log_dict,
    build_val_log_dict,
)
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
        wpe: float = 0.01,
        wp: int = 2,
        wp0: float = 0.0,
        sche_type: str = "cos",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        predictor = predictor or {}
        loss = loss or {}
        inference = inference or {}
        shift_reentry = shift_reentry or {"mode": "streams_only"}

        if parent_conditioning not in ("dual_stream", "fused_hard"):
            raise ValueError(
                f"parent_conditioning must be dual_stream or fused_hard, got {parent_conditioning!r}"
            )



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
        # pred-MSE is logging only: it decodes argmax indices through the frozen
        # VAE, so it can never carry gradient (review §2.2). `lambda_pred_mse`
        # is accepted as a legacy on/off switch.
        legacy_lambda = float(loss.get("lambda_pred_mse", 0.5))
        self.log_pred_mse = bool(loss.get("log_pred_mse", legacy_lambda > 0))
        self.pred_mse_every_n_steps = int(loss.get("pred_mse_every_n_steps", 50))
        self.lambda_pred_mse = legacy_lambda
        self.lambda_ce = float(loss.get("lambda_ce", 1.0))
        self.pred_mse_activations = loss.get("pred_mse_activations", "zero")
        self.inference_mode = inference.get("mode", "autoregressive")
        self.commit_mode = inference.get("commit", "argmax")
        self.parallel_mode = inference.get("parallel_mode", "full")
        self.shift_reentry_mode = shift_reentry.get("mode", "streams_only")
        if self.shift_reentry_mode != "streams_only":
            raise ValueError(f"Unsupported shift_reentry mode: {self.shift_reentry_mode!r}")

        self.lr = learning_rate
        self.wpe = wpe
        self.wp = wp
        self.wp0 = wp0
        self.sche_type = sche_type

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

        S = self.schedule.S
        stage_dims = [int(d) for d in stage_list(predictor.get("dim", 32), S, "dim")]
        stage_layers = [int(n) for n in stage_list(predictor.get("n_layers", 4), S, "n_layers")]
        stage_heads = [int(n) for n in stage_list(predictor.get("n_heads", 8), S, "n_heads")]
        default_windows = [-1, 3, 1] if S == 3 else [-1] * S
        temporal_windows = [
            int(w) for w in stage_list(
                predictor.get("temporal_windows", default_windows), S, "temporal_windows"
            )
        ]
        attention_cfg = predictor.get("attention", {})
        max_shifts = self.schedule.ratio ** (self.schedule.S - 1)
        use_rev_bp = bool(predictor.get("use_reversible_backprop", False))

        for s in range(S):
            if stage_dims[s] % stage_heads[s] != 0:
                raise ValueError(
                    f"predictor.dim[{s}]={stage_dims[s]} must be divisible by "
                    f"predictor.n_heads[{s}]={stage_heads[s]}"
                )

        pred_stages = nn.ModuleList()
        for s, spec in enumerate(self.schedule.stages):
            has_parent = s > 0
            stage_attn = resolve_stage_attention(attention_cfg, s, temporal_windows)
            attn_type = stage_attn.get("type", "full")
            t_window = stage_attn.get("t_window", temporal_windows[s])
            spatial_window = stage_attn.get("spatial_window")
            parent_dim = stage_dims[s - 1] if has_parent else None
            vae_codebook_dim = int(dims[s])

            pred_stages.append(
                PredictorStage(
                    dim=stage_dims[s],
                    n_heads=stage_heads[s],
                    n_layers=stage_layers[s],
                    codebook_size=spec.codebook_size,
                    codebook_dim=vae_codebook_dim,
                    h=spec.H,
                    w=spec.W,
                    max_t=t_total,
                    max_shifts=max_shifts,
                    has_parent=has_parent,
                    parent_dim=parent_dim,
                    parent_mode=parent_conditioning,
                    attention_type=attn_type,
                    t_window=t_window,
                    spatial_window=spatial_window,
                    use_reversible_backprop=use_rev_bp,
                )
            )
        self.predictor_stages = pred_stages
        cross_window = int(predictor.get("cross_spatial_window", 0))
        self.orchestrator = EnvelopeOrchestrator(
            pred_stages,
            self.schedule,
            parent_mode=parent_conditioning,
            lambda_pred_mse=self.lambda_pred_mse,
            shift_ce_weights=loss.get("shift_ce_weights", "uniform"),
            temporal_windows=temporal_windows,
            commit_mode=self.commit_mode,
        )
        self.mask_cache = PredictorMaskCache(
            self.schedule, temporal_windows, cross_spatial_window=cross_window
        )
        self.batch_prep = BatchPrep(
            self.schedule,
            temporal_windows,
            mask_cache=self.mask_cache,
            cross_spatial_window=cross_window,
        )
        self._copy_codebooks()

    def _copy_codebooks(self) -> None:
        for s, block in enumerate(self.vae.hier_quant.blocks):
            q = block.quantizer
            if not hasattr(q, "codebook"):
                continue
            stage = self.predictor_stages[s]
            vae_cb = q.codebook.data
            if vae_cb.shape != stage.codebook_embed.weight.shape:
                raise ValueError(
                    f"Stage {s} codebook shape mismatch: VAE {tuple(vae_cb.shape)} vs "
                    f"predictor embed {tuple(stage.codebook_embed.weight.shape)}"
                )
            stage.codebook_embed.weight.data.copy_(vae_cb)
            stage.codebook_embed.weight.requires_grad_(False)

    def _decode_activations(self, batch):
        if self.pred_mse_activations == "gt":
            return batch.activations, batch.encoder_bottleneck
        acts = {k: torch.zeros_like(v) for k, v in batch.activations.items()}
        return acts, torch.zeros_like(batch.encoder_bottleneck)

    def _decode_video_from_indices(
        self,
        batch,
        pred_indices: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        level_indices = [pred_indices[s] for s in range(self.schedule.S)]
        acts, bottleneck = self._decode_activations(batch)
        return self.vae.decode_from_indices(
            level_indices,
            acts,
            encoder_bottleneck=bottleneck,
        ).clamp(-1, 1)

    @torch.no_grad()
    def _decode_horizon_mse(
        self,
        batch,
        pred_indices: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Logging-only pixel MSE: indices are discrete, so no gradient exists."""
        recon = self._decode_video_from_indices(batch, pred_indices)
        target = batch.video
        horizon = recon[:, :, self.t_context : self.t_total]
        gt = target[:, :, self.t_context : self.t_total]
        return torch.mean((horizon - gt) ** 2)

    def forward_batch(
        self,
        video: torch.Tensor,
        *,
        inference_mode: Optional[str] = None,
        compute_pred_mse: Optional[bool] = None,
    ):
        # Predictor + large-vocab CE are unstable in fp16 (batch 32, K up to 4096).
        device_type = "cuda" if video.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            batch = self.batch_prep.encode_and_schedule(self.vae, video, self.predictor_stages)
            mode = inference_mode or "train"

            if mode == "parallel":
                out = self.orchestrator.forward_parallel(batch, parallel_mode=self.parallel_mode)
            elif mode == "autoregressive":
                out = self.orchestrator.forward_autoregressive(batch)
            else:
                out = self.orchestrator.forward_train(batch)

            if compute_pred_mse is None:
                compute_pred_mse = self.log_pred_mse
            if compute_pred_mse and self.log_pred_mse:
                out.loss_mse = self._decode_horizon_mse(batch, out.pred_indices)

            # CE is the only differentiable objective (pred-MSE is logging only).
            total = self.lambda_ce * out.loss_ce

        return total, out, batch

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        """Decode paths for horizon visualization (callback contract)."""
        video = batch["video"].to(self.device)
        prep = self.batch_prep.encode_and_schedule(self.vae, video, self.predictor_stages)

        gt_indices = {s: prep.gt_indices[s] for s in range(self.schedule.S)}
        vae_recon = self._decode_video_from_indices(prep, gt_indices)

        out_train = self.orchestrator.forward_train(prep)
        pred_train = self._decode_video_from_indices(prep, out_train.pred_indices)

        out_parallel = self.orchestrator.forward_parallel(
            prep, parallel_mode=self.parallel_mode
        )
        pred_parallel = self._decode_video_from_indices(prep, out_parallel.pred_indices)

        out_ar = self.orchestrator.forward_autoregressive(prep)
        pred_ar = self._decode_video_from_indices(prep, out_ar.pred_indices)

        return {
            "inputs": video,
            "vae_recon": vae_recon,
            "pred_train": pred_train,
            "pred_parallel": pred_parallel,
            "pred_ar": pred_ar,
        }

    def training_step(self, batch, batch_idx):
        video = batch["video"]
        if self.trainer is not None and self.trainer.optimizers:
            self._lr_annealing()

        compute_mse = self.log_pred_mse and (
            self.global_step % self.pred_mse_every_n_steps == 0
        )
        loss, out, prep = self.forward_batch(
            video, inference_mode="train", compute_pred_mse=compute_mse
        )
        log_dict = build_train_log_dict(self, out, prep, loss_total=loss)
        self.log_dict(log_dict, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_total", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video = batch["video"]
        out_par = None
        parallel_skipped = False

        try:
            _, out_par, _ = self.forward_batch(video, inference_mode="parallel")
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "not enough memory" in msg:
                parallel_skipped = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

        loss_ar, out_ar, prep = self.forward_batch(video, inference_mode="autoregressive")
        log_dict = build_val_log_dict(
            self,
            out_ar,
            prep,
            loss_total_ar=loss_ar,
            out_par=out_par,
            parallel_skipped=parallel_skipped,
        )
        self.log_dict(log_dict, logger=True, on_step=False, on_epoch=True)
        self.log(
            "val/loss_total_ar",
            loss_ar,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss_ar

    def _lr_annealing(self) -> None:
        g_it = self.trainer.global_step
        iters_train = len(self.trainer.train_dataloader)
        max_it = self.trainer.max_epochs * iters_train
        wp_it = self.wp * iters_train

        if g_it < wp_it:
            cur_lr = self.wp0 + (1 - self.wp0) * g_it / max(wp_it, 1)
        else:
            pasd = (g_it - wp_it) / max(max_it - 1 - wp_it, 1)
            if self.sche_type == "cos":
                cur_lr = self.wpe + (1 - self.wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
            else:
                cur_lr = 1.0
        cur_lr *= self.lr
        opt = self.optimizers()
        if not isinstance(opt, (list, tuple)):
            opt = [opt]
        for o in opt:
            for pg in o.param_groups:
                pg["lr"] = cur_lr

    def configure_optimizers(self):
        params = [p for p in self.predictor_stages.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)
