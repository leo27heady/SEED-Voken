"""Optional perceptual loss on top of HQ-ELBO hierarchical training."""

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.losses.lpips import LPIPS


def flatten_t_dim(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


class HierVideoReconLoss(nn.Module):
    """Perceptual loss on reconstructions (video B,C,T,H,W)."""

    def __init__(self, perceptual_weight: float = 0.0, codebook_weight: float = 1.0, **_):
        super().__init__()
        self.perceptual_weight = float(perceptual_weight)
        self.codebook_weight = float(codebook_weight)
        self.perceptual_loss = LPIPS().eval() if self.perceptual_weight > 0 else None
        self.use_discriminator = False
        self.discriminator = None

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
        if optimizer_idx != 0:
            return torch.tensor(0.0, device=inputs.device), {}
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
        loss = nll_loss + scaled_codebook
        log = {
            f"{split}/recon_nll": nll_loss.detach(),
            f"{split}/perceptual": p_loss.detach().mean()
            if torch.is_tensor(p_loss)
            else p_loss,
        }
        return loss, log


def dummy_loss_break(device: torch.device) -> SimpleNamespace:
    z = torch.tensor(0.0, device=device)
    return SimpleNamespace(
        commitment=z,
        per_sample_entropy=z,
        codebook_entropy=z,
    )
