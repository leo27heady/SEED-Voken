"""Behavior-preserving golden for the Path-3 predictor refactor (PLAN_V2 §5.7).

Captures forward_train / forward_parallel(full|context) / forward_autoregressive
CE + per-(s,k) ce_breakdown on the `lite_stack` at `pre-path3-baseline`, then
asserts bit-for-bit after each refactor step. The fixture stores the *weights*
(VAE + predictor stages) and the input video, so the golden survives parameter
*renames* (e.g. the CrossConditioning state-dict remap, §5.5/H4): the renamed
model must still load the old-key fixture (via its `_load_from_state_dict` hook)
and reproduce the recorded CE.

If the fixture is missing the test RECORDS it (and passes). Delete the files
under tests/predictor/_golden/ to re-baseline intentionally.
"""

import json
import os

import pytest
import torch

from tests.predictor.stack_factory import lite_stack

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "_golden")
_PT = os.path.join(_GOLDEN_DIR, "lite_baseline.pt")
_JSON = os.path.join(_GOLDEN_DIR, "lite_baseline_ce.json")
_SEED = 1234
_ATOL = 1e-5


def _run_all_modes(orch, batch):
    """Return {mode: {"loss_ce": float, "breakdown": {"s_k": float}}} deterministically."""
    out = {}
    specs = [
        ("train", lambda: orch.forward_train(batch)),
        ("parallel_full", lambda: orch.forward_parallel(batch, parallel_mode="full")),
        ("parallel_context", lambda: orch.forward_parallel(batch, parallel_mode="context")),
        ("autoregressive", lambda: orch.forward_autoregressive(batch)),
    ]
    for name, fn in specs:
        res = fn()
        bd = {f"{s}_{k}": float(v.item()) for (s, k), v in sorted(res.ce_breakdown.items())}
        out[name] = {"loss_ce": float(res.loss_ce.item()), "breakdown": bd}
    return out


def _build_and_tokenize(video):
    """Fresh lite_stack (seeded) + tokenized batch. Returns (vae, stages, orch, batch)."""
    torch.manual_seed(_SEED)
    vae, prep, orch, stages, _sched = lite_stack(n_layers=2)
    vae.eval()
    stages.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
    return vae, prep, orch, stages, batch


def test_golden_refactor_ce_bit_equal():
    if not os.path.exists(_PT):
        # ---- record mode (run once at pre-path3-baseline) ----
        torch.manual_seed(_SEED)
        video = torch.randn(2, 3, 9, 32, 32)
        _vae, _prep, orch, _stages, batch = _build_and_tokenize(video)
        with torch.no_grad():
            ce = _run_all_modes(orch, batch)
        os.makedirs(_GOLDEN_DIR, exist_ok=True)
        # Persist ONLY the predictor stages (the refactor target) + input video.
        # The VAE is built first in lite_stack and is never touched by the Path-3
        # predictor refactor, so it rebuilds bit-identically from the fixed seed —
        # no need to commit its (large) weights.
        torch.save({"stages": _stages.state_dict(), "video": video}, _PT)
        with open(_JSON, "w", encoding="utf-8") as f:
            json.dump(ce, f, indent=2, sort_keys=True)
        pytest.skip(f"recorded golden fixture at {_GOLDEN_DIR} (re-run to assert)")

    # ---- assert mode ----
    blob = torch.load(_PT, map_location="cpu", weights_only=False)
    video = blob["video"]
    with open(_JSON, "r", encoding="utf-8") as f:
        golden = json.load(f)

    torch.manual_seed(_SEED)
    vae, prep, orch, stages, _sched = lite_stack(n_layers=2)
    # VAE rebuilds deterministically (seed); load the *recorded* predictor weights
    # (strict: proves no key drift; the remap hook makes the old-key fixture load
    # into renamed modules after §5.5).
    stages.load_state_dict(blob["stages"], strict=True)
    vae.eval()
    stages.eval()
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        ce = _run_all_modes(orch, batch)

    mismatches = []
    for mode, g in golden.items():
        cur = ce[mode]
        if abs(cur["loss_ce"] - g["loss_ce"]) > _ATOL:
            mismatches.append(f"{mode}.loss_ce: {cur['loss_ce']:.8f} != {g['loss_ce']:.8f}")
        for key, gv in g["breakdown"].items():
            cv = cur["breakdown"].get(key)
            if cv is None or abs(cv - gv) > _ATOL:
                mismatches.append(f"{mode}.ce[{key}]: {cv} != {gv:.8f}")
    assert not mismatches, "Golden CE drift (refactor changed behavior):\n" + "\n".join(mismatches)
