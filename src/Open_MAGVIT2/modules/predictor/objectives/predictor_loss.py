"""Predictor loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    codebook_size: int,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, codebook_size),
        targets.reshape(-1),
    )
