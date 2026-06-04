"""Optional perceptual + GAN losses on top of HQ-ELBO hierarchical training."""

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Open_MAGVIT2.modules.discriminator.model import NLayerDiscriminator3D, weights_init
from src.Open_MAGVIT2.modules.losses.lpips import LPIPS
from src.Open_MAGVIT2.modules.losses.video_vqperceptual import (
    adopt_weight,
    flatten_t_dim,
    hinge_d_loss,
    non_saturate_discriminator_loss,
    non_saturate_gen_loss,
    vanilla_d_loss,
)


class HierVideoReconLoss(nn.Module):
    """Perceptual and/or adversarial loss on reconstructions (video B,C,T,H,W).

    Designed to combine with ``compute_hier_elbo_loss`` (HQ-ELBO + layer KL).
    """

    def __init__(
        self,
        perceptual_weight: float = 0.0,
        disc_start: int = 0,
        disc_weight: float = 1.0,
        disc_factor: float = 1.0,
        disc_loss: str = "hinge",
        disc_num_layers: int = 3,
        disc_ndf: int = 64,
        gen_loss_weight: Optional[float] = None,
        codebook_weight: float = 1.0,
    ):
        super().__init__()
        self.perceptual_weight = float(perceptual_weight)
        self.discriminator_iter_start = int(disc_start)
        self.discriminator_weight = float(disc_weight)
        self.disc_factor = float(disc_factor)
        self.gen_loss_weight = gen_loss_weight
        self.codebook_weight = float(codebook_weight)

        self.perceptual_loss = LPIPS().eval() if self.perceptual_weight > 0 else None

        self.use_discriminator = disc_weight > 0 and disc_factor > 0
        if self.use_discriminator:
            self.discriminator = NLayerDiscriminator3D(
                input_nc=3, n_layers=disc_num_layers, ndf=disc_ndf
            ).apply(weights_init)
            if disc_loss == "hinge":
                self.disc_loss_fn = hinge_d_loss
            elif disc_loss == "vanilla":
                self.disc_loss_fn = vanilla_d_loss
            elif disc_loss == "non_saturate":
                self.disc_loss_fn = non_saturate_discriminator_loss
            else:
                raise ValueError(f"Unknown disc_loss: {disc_loss}")
        else:
            self.discriminator = None
            self.disc_loss_fn = None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(d_weight, 0.0, 1e4).detach() * self.discriminator_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        codebook_loss: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: Optional[torch.Tensor] = None,
        split: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec_loss = torch.abs(
            flatten_t_dim(inputs).contiguous() - flatten_t_dim(reconstructions).contiguous()
        )
        nll_loss = rec_loss.clone()
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                flatten_t_dim(inputs).contiguous(),
                flatten_t_dim(reconstructions).contiguous(),
            )
            nll_loss = nll_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0.0, device=inputs.device)

        nll_loss = torch.mean(nll_loss)
        scaled_codebook = self.codebook_weight * codebook_loss

        if optimizer_idx == 0:
            g_loss = torch.tensor(0.0, device=inputs.device)
            d_weight = torch.tensor(0.0, device=inputs.device)
            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            if self.discriminator is not None and disc_factor > 0:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = non_saturate_gen_loss(logits_fake)
                if self.gen_loss_weight is None and last_layer is not None:
                    try:
                        d_weight = self.calculate_adaptive_weight(
                            nll_loss, g_loss, last_layer
                        )
                    except RuntimeError:
                        d_weight = torch.tensor(0.0, device=inputs.device)
                else:
                    d_weight = torch.tensor(
                        self.gen_loss_weight or 0.0, device=inputs.device
                    )
                g_loss = d_weight * disc_factor * g_loss

            loss = nll_loss + g_loss + scaled_codebook
            log = {
                f"{split}/recon_nll": nll_loss.detach(),
                f"{split}/perceptual": p_loss.detach().mean()
                if torch.is_tensor(p_loss)
                else p_loss,
                f"{split}/g_loss": g_loss.detach(),
                f"{split}/d_weight": d_weight.detach()
                if torch.is_tensor(d_weight)
                else d_weight,
                f"{split}/disc_factor": torch.tensor(disc_factor, device=inputs.device),
            }
            return loss, log

        if optimizer_idx == 1:
            if self.discriminator is None:
                return torch.tensor(0.0, device=inputs.device), {}
            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)
            log = {f"{split}/d_loss": d_loss.detach()}
            return d_loss, log

        raise ValueError(f"optimizer_idx must be 0 or 1, got {optimizer_idx}")


def dummy_loss_break(device: torch.device) -> SimpleNamespace:
    """Placeholder for VQLPIPSWithDiscriminator API compatibility."""
    z = torch.tensor(0.0, device=device)
    return SimpleNamespace(
        commitment=z,
        per_sample_entropy=z,
        codebook_entropy=z,
    )
