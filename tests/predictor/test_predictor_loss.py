"""Unit tests for predictor loss helpers."""

import torch

from src.Open_MAGVIT2.modules.predictor.objectives.predictor_loss import shift_ce_loss


def test_shift_ce_loss():
    logits = torch.randn(2, 16, 64)
    targets = torch.randint(0, 64, (2, 16))
    loss = shift_ce_loss(logits, targets, 64)
    assert torch.isfinite(loss)
    assert loss.item() > 0
