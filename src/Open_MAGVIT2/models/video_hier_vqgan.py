import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import lightning as L
import torch
import torch.nn.functional as F

from main import instantiate_from_config
from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Decoder, Encoder
from src.Open_MAGVIT2.modules.ema import LitEma
from src.Open_MAGVIT2.modules.scheduler.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.factory import build_top_down, set_all_sq_temperature
from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import QuantizerResult
from src.Open_MAGVIT2.modules.losses.hier_video_loss import HierVideoReconLoss
from src.Open_MAGVIT2.modules.numerical_debug import (
    check_tensor,
    get_numerical_report,
    numerical_debug_enabled,
    reset_numerical_report,
    set_numerical_debug,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.hier_elbo_loss import compute_hier_elbo_loss
from src.Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import validate_hierarchy_taps


def _as_float_list(value, name: str) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _build_kl_weights_tensor(
    kl_weights: Optional[List[float]], num_layers: int
) -> Optional[torch.Tensor]:
    if kl_weights is None:
        return None
    if len(kl_weights) == 1:
        kl_weights = kl_weights * num_layers
    if len(kl_weights) != num_layers:
        raise ValueError(
            f"loss.kl_weights length {len(kl_weights)} != num_layers {num_layers}"
        )
    return torch.tensor(kl_weights, dtype=torch.float32)


def _layer_codebook_usage_logs(
    layer_results: List[QuantizerResult], prefix: str
) -> Dict[str, torch.Tensor]:
    log_dict: Dict[str, torch.Tensor] = {}
    for i, result in enumerate(layer_results):
        stats = result.log_stats
        if "active_codes" not in stats:
            continue
        ac = stats["active_codes"]
        if not torch.is_tensor(ac):
            ac = torch.tensor(float(ac))
        log_dict[f"{prefix}/active_codes_layer_{i + 1}"] = ac.detach().float()
        uf = stats.get("usage_fraction")
        if uf is not None:
            if not torch.is_tensor(uf):
                uf = torch.tensor(float(uf))
            log_dict[f"{prefix}/code_usage_frac_layer_{i + 1}"] = uf.detach().float()
        ppl = result.perplexity
        if torch.is_tensor(ppl) and "size_dict" in stats:
            k = stats["size_dict"]
            if not torch.is_tensor(k):
                k = torch.tensor(float(k), device=ppl.device)
            log_dict[f"{prefix}/perplexity_frac_layer_{i + 1}"] = (ppl / k).detach()
    return log_dict


class VideoHierVQModel(L.LightningModule):
    def __init__(
        self,
        ddconfig,
        hierarchy: Dict[str, Any],
        quantizer: Dict[str, Any],
        learning_rate: float = 1e-4,
        training_objective: str = "hq_elbo",
        use_gan: bool = False,
        lossconfig: Optional[Dict[str, Any]] = None,
        image_key: str = "video",
        use_ema: bool = True,
        ckpt_path: Optional[str] = None,
        ignore_keys: Optional[List[str]] = None,
        warmup_epochs: float = 1.0,
        scheduler_type: str = "None",
        min_learning_rate: float = 0.0,
        sche_type: Optional[str] = None,
        wp: int = 0,
        wp0: float = 0.0,
        wpe: float = 0.01,
        max_iter: Optional[int] = None,
        wp_iter: Optional[int] = None,
        resume_lr: Optional[float] = None,
        top_down_width: int = 64,
        progressive_coding: bool = False,
        progressive_noise_weight: float = 0.0,
        loss_cfg: Optional[Dict[str, Any]] = None,
        numerical_debug: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["ddconfig", "hierarchy", "quantizer", "lossconfig", "loss_cfg"]
        )

        self.image_key = image_key
        self.training_objective = training_objective
        self.use_gan = use_gan
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.use_ema = use_ema
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        self.sche_type = sche_type
        self.max_it = max_iter
        self.wp_iter = wp_iter
        self.resume_lr = resume_lr
        self.progressive_coding = progressive_coding
        self.progressive_noise_weight = progressive_noise_weight
        self.numerical_debug = numerical_debug
        self._last_numerical_report = None
        self.automatic_optimization = False

        loss_cfg = dict(loss_cfg or {})
        self.loss_cfg = loss_cfg
        num_layers_hint = len(
            _as_float_list(quantizer.get("size_dict"), "size_dict") or [2]
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        z_channels = ddconfig["z_channels"]

        hierarchy = dict(hierarchy)
        self.hierarchy_mode = hierarchy.get("mode", "sqvae2").lower()
        self.hier_quant = build_top_down(
            hierarchy_cfg=hierarchy,
            quantizer_cfg=dict(quantizer),
            z_channels=z_channels,
            width=top_down_width,
        )
        tap_seq_len = hierarchy.get("sequence_length")
        if (
            tap_seq_len is not None
            and self.hierarchy_mode == "sqvae2"
            and hasattr(self.hier_quant, "resolution_keys")
        ):
            validate_hierarchy_taps(
                ddconfig,
                int(tap_seq_len),
                list(self.hier_quant.resolution_keys),
                hierarchy.get("tap_channels"),
                tap_key_format=hierarchy.get("tap_key_format", "spatial"),
            )
        if hasattr(self.hier_quant, "num_layers"):
            num_layers_hint = self.hier_quant.num_layers
        self.kl_weights = _build_kl_weights_tensor(
            _as_float_list(loss_cfg.get("kl_weights"), "kl_weights"),
            num_layers_hint,
        )

        self.temp_cfg = dict(quantizer.get("temperature", {}))
        self.temp_init = self.temp_cfg.get("init", 1.0)
        self.temp_decay = self.temp_cfg.get("decay", 1e-5)
        self.temp_min = self.temp_cfg.get("min", 0.0)

        perceptual_cfg = loss_cfg.get("perceptual", {})
        gan_cfg = loss_cfg.get("gan", {})
        self.perceptual_weight = float(
            perceptual_cfg.get("weight", loss_cfg.get("perceptual_weight", 0.0))
        )
        self.use_perceptual = bool(
            perceptual_cfg.get("enabled", self.perceptual_weight > 0)
        )
        self.use_gan_loss = bool(
            gan_cfg.get("enabled", use_gan or loss_cfg.get("gan_enabled", False))
        )
        self.recon_loss = None
        if self.use_perceptual or self.use_gan_loss:
            self.recon_loss = HierVideoReconLoss(
                perceptual_weight=self.perceptual_weight if self.use_perceptual else 0.0,
                disc_start=int(gan_cfg.get("disc_start", loss_cfg.get("disc_start", 0))),
                disc_weight=float(gan_cfg.get("weight", loss_cfg.get("disc_weight", 0.0))),
                disc_factor=float(gan_cfg.get("disc_factor", 1.0)),
                disc_loss=str(gan_cfg.get("disc_loss", "hinge")),
                codebook_weight=float(loss_cfg.get("codebook_weight", 0.0)),
            )

        self.loss = None
        if use_gan and lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)

        if self.use_ema:
            self.model_ema = LitEma(self)

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            self.load_state_dict(sd, strict=False)

        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def get_input(self, batch, k):
        return batch[k]

    def _update_temperature(self):
        step = self.global_step
        tau = max(self.temp_min, self.temp_init * math.exp(-self.temp_decay * step))
        set_all_sq_temperature(self.hier_quant, tau)
        return tau

    def on_train_start(self) -> None:
        set_numerical_debug(self.numerical_debug, fail_fast=True)

    def encode(self, x, flg_train=True, flg_quant_det=False):
        if self.numerical_debug:
            reset_numerical_report()
        if self.hierarchy_mode == "sqvae2":
            h, activations = self.encoder(x, return_intermediates=True)
            if self.numerical_debug:
                check_tensor("encoder/bottleneck", h)
                for key, act in activations.items():
                    check_tensor(f"encoder/tap_{key}", act)
            z_q, layer_results = self.hier_quant(
                activations,
                encoder_bottleneck=h,
                flg_train=flg_train,
                flg_quant_det=flg_quant_det,
            )
        else:
            h = self.encoder(x)
            activations = None
            z_q, layer_results = self.hier_quant(
                h, flg_train=flg_train, flg_quant_det=flg_quant_det
            )
        return z_q, layer_results, h, activations

    def _pack_token_levels(self, layer_results) -> List[Dict[str, Any]]:
        levels = []
        for i, result in enumerate(layer_results):
            meta = self.hier_quant.level_metadata(i)
            grid = result.grid_shape or tuple(result.indices.shape[1:])
            levels.append(
                {
                    "key": result.resolution_key or meta["resolution_key"],
                    "indices": result.indices,
                    "codebook_size": meta["codebook_size"],
                    "shape": grid,
                }
            )
        return levels

    def encode_tokens(self, x, flg_train=False, flg_quant_det=True):
        """Per-level discrete tokens plus fused z_q for the MAGVIT decoder."""
        if self.hierarchy_mode != "sqvae2":
            raise NotImplementedError(f"encode_tokens requires sqvae2, got {self.hierarchy_mode}")
        h, activations = self.encoder(x, return_intermediates=True)
        z_q, layer_results = self.hier_quant(
            activations,
            encoder_bottleneck=h,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det,
        )
        return {
            "levels": self._pack_token_levels(layer_results),
            "z_q": z_q,
            "activations": activations,
            "encoder_bottleneck": h,
        }

    def decode_from_indices(
        self,
        level_indices: List[torch.Tensor],
        activations: Optional[Dict[str, torch.Tensor]] = None,
        encoder_bottleneck: Optional[torch.Tensor] = None,
    ):
        if self.hierarchy_mode != "sqvae2":
            raise NotImplementedError(f"decode_from_indices requires sqvae2, got {self.hierarchy_mode}")
        if activations is None or encoder_bottleneck is None:
            raise ValueError("sqvae2 decode_from_indices requires activations and encoder_bottleneck")
        z_q = self.hier_quant.decode_from_indices(
            level_indices,
            activations,
            encoder_bottleneck=encoder_bottleneck,
        )
        return self.decode(z_q)

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x, flg_train=True, flg_quant_det=False):
        z_q, layer_results, _, _ = self.encode(x, flg_train, flg_quant_det)
        x_rec = self.decode(z_q)
        if self.numerical_debug:
            check_tensor("decoder/x_rec", x_rec)
            self._last_numerical_report = get_numerical_report()
        return x_rec, layer_results

    def decode_progressive(self, x, flg_quant_det=True, flg_train=False):
        if self.hierarchy_mode == "sqvae2":
            h, activations = self.encoder(x, return_intermediates=True)
            _, partial_z = self.hier_quant.forward_progressive(
                activations,
                encoder_bottleneck=h,
                flg_quant_det=flg_quant_det,
                flg_train=flg_train,
            )
        else:
            h = self.encoder(x)
            _, partial_z = self.hier_quant.forward_progressive(
                h, flg_quant_det=flg_quant_det, flg_train=flg_train
            )
        return [self.decode(z).clamp(-1, 1) for z in partial_z]

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        self._update_temperature()
        x_rec, layer_results = self(x, flg_train=True, flg_quant_det=False)

        progressive_recs = None
        if self.progressive_coding:
            progressive_recs = self.decode_progressive(x, flg_quant_det=False, flg_train=True)

        if self.sche_type is not None and self.resume_lr is None:
            g_it = self.trainer.global_step
            if self.max_it is None:
                iters_train = len(self.trainer.train_dataloader)
                max_it = self.trainer.max_epochs * iters_train
                wp_it = self.wp * iters_train
            else:
                max_it = self.max_it
                wp_it = self.wp_iter
            self.lr_annealing(self.learning_rate, g_it, wp_it, max_it, wp0=self.wp0, wpe=self.wpe)

        if (
            self.recon_loss is not None
            and getattr(self.recon_loss, "use_discriminator", False)
        ):
            opt_gen, opt_disc = self.optimizers()
            kl_w = self.kl_weights
            if kl_w is not None:
                kl_w = kl_w.to(x.device)
            base_loss, log_dict = compute_hier_elbo_loss(
                x,
                x_rec,
                layer_results,
                progressive_recs=progressive_recs,
                progressive_noise_weight=self.progressive_noise_weight,
                kl_weights=kl_w,
            )
            codebook_loss = log_dict["loss/kl_total"]
            recon_extra, recon_log = self.recon_loss(
                x,
                x_rec,
                codebook_loss,
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            loss = base_loss + recon_extra
            log_dict.update(recon_log)
            log_dict.update(_layer_codebook_usage_logs(layer_results, "train"))
            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()

            discloss, disc_log = self.recon_loss(
                x,
                x_rec,
                codebook_loss,
                optimizer_idx=1,
                global_step=self.global_step,
                split="train",
            )
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()
            log_dict.update(disc_log)
        else:
            opt = self.optimizers()
            kl_w = self.kl_weights
            if kl_w is not None:
                kl_w = kl_w.to(x.device)
            loss, log_dict = compute_hier_elbo_loss(
                x,
                x_rec,
                layer_results,
                progressive_recs=progressive_recs,
                progressive_noise_weight=self.progressive_noise_weight,
                kl_weights=kl_w,
            )
            if self.use_perceptual and self.recon_loss is not None:
                codebook_loss = log_dict["loss/kl_total"]
                p_extra, p_log = self.recon_loss(
                    x,
                    x_rec,
                    codebook_loss,
                    optimizer_idx=0,
                    global_step=self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train",
                )
                loss = loss + p_extra
                log_dict.update(p_log)
            log_dict.update(_layer_codebook_usage_logs(layer_results, "train"))
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        if self.numerical_debug:
            report = get_numerical_report()
            if report is not None:
                self._last_numerical_report = report
                if self.global_step % 100 == 0:
                    log_dict.update(report.as_log_dict())

        log_dict["train/temperature"] = torch.tensor(
            max(self.temp_min, self.temp_init * math.exp(-self.temp_decay * self.global_step))
        )
        self.log_dict(
            {k: v for k, v in log_dict.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        x_rec, layer_results = self(x, flg_train=False, flg_quant_det=True)
        loss, log_dict = compute_hier_elbo_loss(x, x_rec, layer_results, kl_weights=self.kl_weights)
        log_dict.update(_layer_codebook_usage_logs(layer_results, f"val{suffix}"))
        log_dict = {f"val{suffix}/{k.split('/', 1)[-1]}": v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def lr_annealing(self, peak_lr, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
        wp_it = round(wp_it)
        if cur_it < wp_it:
            cur_lr = wp0 + (1 - wp0) * cur_it / max(wp_it, 1)
        else:
            pasd = (cur_it - wp_it) / max(max_it - 1 - wp_it, 1)
            if self.sche_type == "cos":
                cur_lr = wpe + (1 - wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
            else:
                cur_lr = 1.0
        cur_lr *= peak_lr
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        for opt in opts:
            for param_group in opt.param_groups:
                param_group["lr"] = cur_lr * param_group.get("lr_sc", 1)

    def configure_optimizers(self):
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.hier_quant.parameters())
        )
        opt_gen = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.5, 0.9))
        if (
            self.recon_loss is not None
            and getattr(self.recon_loss, "use_discriminator", False)
        ):
            opt_disc = torch.optim.Adam(
                self.recon_loss.discriminator.parameters(),
                lr=self.learning_rate,
                betas=(0.5, 0.9),
            )
            return [opt_gen, opt_disc]
        return opt_gen

    def get_last_layer(self):
        return self.decoder.conv_out.conv_1.weight

    def log_images(self, batch, **kwargs):
        x = self.get_input(batch, self.image_key).to(self.device)
        x_rec, _ = self(x, flg_train=False, flg_quant_det=True)
        log = {"inputs": x, "reconstructions": x_rec.clamp(-1, 1)}

        # Only generate progressive rows when progressive coding is enabled; this
        # keeps non-progressive runs focused on GT vs reconstruction grids and
        # avoids decoding untrained partial latents.
        if self.progressive_coding:
            progressive = self.decode_progressive(x, flg_quant_det=True)
            for i, x_prog in enumerate(progressive):
                label = f"progressive_L{i + 1}" if i == 0 else f"progressive_L1-L{i + 1}"
                log[label] = x_prog

        return log
