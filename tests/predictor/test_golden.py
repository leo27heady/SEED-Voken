"""Golden regression snapshots for schedule targets."""

import json

import torch

from src.Open_MAGVIT2.modules.predictor.orchestrator import EnvelopeOrchestrator
from src.Open_MAGVIT2.modules.predictor.schedule import PyramidSchedule
from tests.predictor.test_orchestrator_smoke import _stack as orch_stack


def test_gld_01_envelope_targets_snapshot():
    sched = PyramidSchedule.from_v2_64s()
    payload = {
        "envelope": sched.envelope_order(),
        "targets": {
            f"{s}_{k}": sched.target_token_index(s, k)
            for s in range(sched.S)
            for k in range(sched.shifts_per_stage(s))
        },
        "context_end": sched._context_end,
        "native_t": [sched.native_t(s) for s in range(sched.S)],
    }
    golden = {
        "envelope": [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3)],
        "targets": {
            "0_0": 3, "1_0": 5, "1_1": 6,
            "2_0": 9, "2_1": 10, "2_2": 11, "2_3": 12,
        },
        "context_end": [2, 4, 8],
        "native_t": [4, 7, 13],
    }
    assert payload == golden
    assert json.dumps(payload, sort_keys=True) == json.dumps(golden, sort_keys=True)


def test_gld_02_e2e_loss_tolerance():
    torch.manual_seed(42)
    vae, prep, orch, stages = orch_stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        a = orch.forward_train(batch)
        b = orch.forward_train(batch)
    assert torch.allclose(a.loss_ce, b.loss_ce, rtol=0, atol=1e-5)
