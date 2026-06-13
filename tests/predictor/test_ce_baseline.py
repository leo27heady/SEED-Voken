"""PRED-CE-01: untrained predictor CE near sum of log(K) per shift."""

import math

import torch

from tests.predictor.stack_factory import v2_stack


def _stack():
    return v2_stack(n_layers=2, lambda_pred_mse=0.0)


def test_pred_ce_01_baseline_log_k():
    torch.manual_seed(0)
    vae, prep, orch, stages, schedule = _stack()
    vae.eval()
    stages.eval()
    # B=1: with zero-init heads the CE equals sum log K exactly, batch-independent
    video = torch.randn(1, 3, 13, 64, 64)
    expected = 0.0
    for s in range(schedule.S):
        k_size = schedule.stages[s].codebook_size
        for k in range(schedule.shifts_per_stage(s)):
            expected += math.log(k_size)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        out = orch.forward_train(batch)
    assert torch.isfinite(out.loss_ce)
    assert abs(float(out.loss_ce) - expected) < 0.5
