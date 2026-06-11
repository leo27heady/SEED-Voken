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
        log["train/loss_pred_mse"] = out.loss_mse.detach()
        n_pix = batch.video[:, :, model.t_context : model.t_total].numel()
        log["train/pred_mse_per_pixel"] = (out.loss_mse / max(n_pix, 1)).detach()

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
        log["val/pred_mse_ar"] = out_ar.loss_mse.detach()
        n_pix = batch.video[:, :, model.t_context : model.t_total].numel()
        log["val/pred_mse_ar_per_pixel"] = (out_ar.loss_mse / max(n_pix, 1)).detach()

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
        loss_total_par = total_loss(
            out_par.loss_ce,
            out_par.loss_mse,
            lambda_ce=model.lambda_ce,
            lambda_pred_mse=model.lambda_pred_mse,
        )
        log["val/loss_total_parallel"] = loss_total_par.detach()
        log["val/loss_ce_parallel"] = out_par.loss_ce.detach()
        log["val/ce_gap_parallel_minus_ar"] = (out_par.loss_ce - out_ar.loss_ce).detach()
        if out_par.loss_mse is not None:
            log["val/pred_mse_parallel"] = out_par.loss_mse.detach()
            n_pix = batch.video[:, :, model.t_context : model.t_total].numel()
            log["val/pred_mse_parallel_per_pixel"] = (
                out_par.loss_mse / max(n_pix, 1)
            ).detach()
        for s, ce in aggregate_ce_by_stage(out_par.ce_breakdown).items():
            log[f"val/ce_parallel_stage_{s}"] = ce
        if out_par.ce_breakdown:
            for (s, k), ce in out_par.ce_breakdown.items():
                log[f"val_parallel/ce_s{s}_k{k}"] = ce
        for s, acc in token_accuracy_from_output(out_par, batch, model.schedule).items():
            log[f"val/token_acc_parallel_stage_{s}"] = acc.detach()

    return log
