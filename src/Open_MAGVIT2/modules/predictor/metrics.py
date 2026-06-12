"""Scalar metrics for hierarchical video predictor training."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from src.Open_MAGVIT2.modules.predictor.orchestrator import PredictorOutput
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule


def total_loss(
    ce: torch.Tensor,
    pred_mse: Optional[torch.Tensor],
    *,
    lambda_ce: float,
    lambda_pred_mse: float,
) -> torch.Tensor:
    """Legacy diagnostic combination. The optimized loss is CE-only since
    Path A Phase 2 — pred-MSE decodes argmax indices and carries no gradient."""
    loss = lambda_ce * ce
    if pred_mse is not None and lambda_pred_mse > 0:
        loss = loss + lambda_pred_mse * pred_mse
    return loss


def ce_baseline(schedule: PyramidSchedule) -> float:
    """Sum of log(K) over all envelope shifts (uniform CE at init)."""
    total = 0.0
    for s, _k in schedule.envelope_order():
        total += math.log(schedule.stages[s].codebook_size)
    return total


def aggregate_ce_by_stage(
    ce_breakdown: Optional[Dict[Tuple[int, int], torch.Tensor]],
) -> Dict[int, torch.Tensor]:
    if not ce_breakdown:
        return {}
    sums: Dict[int, list] = {}
    for (s, _k), ce in ce_breakdown.items():
        sums.setdefault(s, []).append(ce)
    return {s: torch.stack(v).mean() for s, v in sums.items()}


def token_accuracy_from_output(
    out: PredictorOutput,
    batch,
    schedule: PyramidSchedule,
) -> Dict[int, torch.Tensor]:
    """Per-stage mean top-1 accuracy over shifts at query positions."""
    correct: Dict[int, list] = {}
    total: Dict[int, list] = {}
    for s, k in schedule.envelope_order():
        logits = out.logits.get((s, k))
        if logits is None:
            continue
        sup = batch.supervision[s][k]
        tgt = batch.target_indices[s][k]
        q_logits = logits[:, sup.query_positions]
        pred = q_logits.argmax(dim=-1).reshape(-1)
        tgt_flat = tgt.reshape(-1)
        n = tgt_flat.numel()
        correct.setdefault(s, []).append((pred == tgt_flat).float().sum())
        total.setdefault(s, []).append(torch.tensor(float(n), device=tgt.device))
    return {
        s: torch.stack(correct[s]).sum() / torch.stack(total[s]).sum().clamp(min=1)
        for s in correct
    }


def build_train_log_dict(
    model,
    out: PredictorOutput,
    batch,
    *,
    loss_total: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    _ = loss_total  # primary loss logged separately (prog_bar) in training_step
    log: Dict[str, torch.Tensor] = {
        "train/loss_ce": out.loss_ce.detach(),
    }
    if out.loss_mse is not None:
        # already a per-pixel mean over the decoded horizon
        log["train/loss_pred_mse"] = out.loss_mse.detach()

    baseline = ce_baseline(model.schedule)
    log["train/ce_over_baseline"] = (out.loss_ce.detach() / baseline)

    for s, ce in aggregate_ce_by_stage(out.ce_breakdown).items():
        log[f"train/ce_stage_{s}"] = ce
    if out.ce_breakdown:
        for (s, k), ce in out.ce_breakdown.items():
            log[f"train/ce_s{s}_k{k}"] = ce

    for s, acc in token_accuracy_from_output(out, batch, model.schedule).items():
        log[f"train/token_acc_stage_{s}"] = acc.detach()

    if model.trainer is not None and model.trainer.optimizers:
        opt = model.optimizers()
        if not isinstance(opt, (list, tuple)):
            opt = [opt]
        log["train/lr"] = torch.tensor(opt[0].param_groups[0]["lr"])
    return log


def build_val_log_dict(
    model,
    out_ar: PredictorOutput,
    batch,
    *,
    loss_total_ar: torch.Tensor,
    out_par: Optional[PredictorOutput] = None,
    parallel_skipped: bool = False,
) -> Dict[str, torch.Tensor]:
    _ = loss_total_ar  # primary val loss logged separately (prog_bar) in validation_step
    log: Dict[str, torch.Tensor] = {
        "val/loss_ce_ar": out_ar.loss_ce.detach(),
    }
    if out_ar.loss_mse is not None:
        # already a per-pixel mean over the decoded horizon
        log["val/pred_mse_ar"] = out_ar.loss_mse.detach()

    for s, ce in aggregate_ce_by_stage(out_ar.ce_breakdown).items():
        log[f"val/ce_ar_stage_{s}"] = ce
    if out_ar.ce_breakdown:
        for (s, k), ce in out_ar.ce_breakdown.items():
            log[f"val/ce_s{s}_k{k}"] = ce

    for s, acc in token_accuracy_from_output(out_ar, batch, model.schedule).items():
        log[f"val/token_acc_ar_stage_{s}"] = acc.detach()

    if parallel_skipped:
        log["val/parallel_skipped"] = torch.tensor(1.0)
    elif out_par is not None:
        # "tf" = teacher-forced/stream-carry mode. With parallel_mode="context"
        # this is exactly the training computation evaluated on val data (it is
        # NOT an independent oracle) — review §2.1. Legacy "parallel" keys are
        # kept for one release for W&B continuity.
        loss_total_par = (model.lambda_ce * out_par.loss_ce).detach()
        log["val/loss_total_parallel"] = loss_total_par
        log["val/loss_ce_parallel"] = out_par.loss_ce.detach()
        log["val/loss_ce_tf"] = out_par.loss_ce.detach()
        gap = (out_par.loss_ce - out_ar.loss_ce).detach()
        log["val/ce_gap_parallel_minus_ar"] = gap
        log["val/ce_gap_tf_minus_ar"] = gap
        if out_par.loss_mse is not None:
            log["val/pred_mse_parallel"] = out_par.loss_mse.detach()
            log["val/pred_mse_tf"] = out_par.loss_mse.detach()
        for s, ce in aggregate_ce_by_stage(out_par.ce_breakdown).items():
            log[f"val/ce_parallel_stage_{s}"] = ce
            log[f"val/ce_tf_stage_{s}"] = ce
        if out_par.ce_breakdown:
            for (s, k), ce in out_par.ce_breakdown.items():
                log[f"val_parallel/ce_s{s}_k{k}"] = ce
                log[f"val_tf/ce_s{s}_k{k}"] = ce
        for s, acc in token_accuracy_from_output(out_par, batch, model.schedule).items():
            log[f"val/token_acc_parallel_stage_{s}"] = acc.detach()
            log[f"val/token_acc_tf_stage_{s}"] = acc.detach()

    return log
