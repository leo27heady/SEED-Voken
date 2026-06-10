"""Stop training when loss becomes non-finite."""

from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


def _is_finite(value: Any) -> bool:
    if value is None:
        return True
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (int, float)):
        return value == value and abs(value) != float("inf")
    return True


def _resolve_train_loss(trainer: L.Trainer, outputs: Any) -> Optional[Any]:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for key in ("loss", "loss/total", "loss/total_step"):
            if key in outputs:
                return outputs[key]
        return None
    metrics = trainer.callback_metrics
    for key in ("loss/total_step", "loss/total", "train/loss"):
        if key in metrics:
            return metrics[key]
    return None


class NanGuardCallback(Callback):
    """Raise if the training loss is NaN or Inf (avoids silent weight corruption)."""

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        loss = _resolve_train_loss(trainer, outputs)
        if loss is None:
            return
        if not _is_finite(loss):
            from src.Open_MAGVIT2.modules.numerical_debug import get_numerical_report

            msg = f"Non-finite training loss at global step {trainer.global_step}"
            report = get_numerical_report()
            if report is not None and report.checks:
                msg = f"{msg}\n{report.summary()}"
            stored = getattr(pl_module, "_last_numerical_report", None)
            if stored is not None and stored is not report:
                msg = f"{msg}\n{stored.summary()}"
            raise RuntimeError(msg)
