"""Tests for NanGuardCallback."""

import pytest
import torch

from src.Open_MAGVIT2.callbacks.nan_guard import _is_finite, _resolve_train_loss


class _FakeTrainer:
    def __init__(self, metrics=None):
        self.callback_metrics = metrics or {}


def test_nan_guard_is_finite():
    assert _is_finite(torch.tensor(1.0))
    assert not _is_finite(torch.tensor(float("nan")))
    assert not _is_finite(float("nan"))


def test_nan_guard_resolve_loss_from_output():
    t = _FakeTrainer()
    loss = torch.tensor(1.5)
    assert _resolve_train_loss(t, loss) is loss
    assert _resolve_train_loss(t, {"loss/total": loss}) is loss


def test_nan_guard_skips_unknown_output_dict():
    t = _FakeTrainer()
    assert _resolve_train_loss(t, {"foo": 1}) is None
