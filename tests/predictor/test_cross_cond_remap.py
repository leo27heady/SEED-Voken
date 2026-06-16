"""CrossConditioning state-dict remap: old checkpoints load + forward bit-equal.

PLAN_V2 §5.5 / H4. We can't instantiate the *old* code (it's gone), so we
simulate a legacy checkpoint by taking a fresh module's state_dict and renaming
the unified keys (`cross_f.q` -> `cross_q_f`, `cross.q` -> `cross_attn_q`, …)
back to the legacy names, then load it into a differently-initialised module via
the `_load_from_state_dict` remap hook and assert (1) strict load succeeds and
(2) the forward output is bit-identical to the reference module.
"""

import torch

from src.Open_MAGVIT2.modules.predictor.attention.factorized_layer import (
    FactorizedPredictorLayer,
)
from src.Open_MAGVIT2.modules.predictor.parallel_residual import ParallelPredictorResidual
from src.Open_MAGVIT2.modules.predictor.reversible_block import ReversibleCouplingBlock
from src.Open_MAGVIT2.modules.predictor.schedule import ShiftMasks


def _to_legacy(sd, mapping):
    """Rename unified keys back to legacy keys (old->new mapping, applied inverse)."""
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in mapping.items():
            if nk == new or nk.startswith(new + "."):
                nk = old + nk[len(new):]
                break
            token = "." + new + "."
            idx = nk.find(token)
            if idx != -1:
                nk = nk[: idx + 1] + old + nk[idx + len(new) + 1:]
                break
        out[nk] = v
    return out


def _assert_remap_roundtrip(make_module, mapping, run_forward):
    torch.manual_seed(0)
    ref = make_module()
    ref.eval()
    legacy_sd = _to_legacy(ref.state_dict(), mapping)
    # at least one key must actually have been renamed
    assert any(k not in ref.state_dict() for k in legacy_sd), "no legacy keys produced"

    torch.manual_seed(123)  # different init than ref
    loaded = make_module()
    loaded.load_state_dict(legacy_sd, strict=True)  # remap hook must make this pass
    loaded.eval()

    with torch.no_grad():
        a = run_forward(ref)
        b = run_forward(loaded)
    assert torch.allclose(a, b, rtol=0, atol=0), "forward differs after legacy load"


def test_factorized_layer_remap():
    dim, n_heads, n_spatial = 16, 4, 4
    x = torch.randn(1, 8, 2 * dim)  # 2 frames x 4 spatial, dual-stream (2*dim)
    parent = torch.randn(1, 8, dim)
    cross_mask = torch.ones(8, 8, dtype=torch.bool)

    def fwd(m):
        return m(
            x, self_attn_mask=None, cross_kv=None, cross_attn_mask=cross_mask,
            layer_idx=0, parent_mode="dual_stream",
            parent_o1=parent, parent_o2=parent, parent_fused=None,
        )

    _assert_remap_roundtrip(
        lambda: FactorizedPredictorLayer(
            dim, n_heads, n_spatial, t_window=1, has_cross_attn=True, parent_dim=dim
        ),
        FactorizedPredictorLayer._CROSS_REMAP,
        fwd,
    )


def test_parallel_residual_remap():
    dim, n_heads = 16, 4
    x = torch.randn(1, 8, dim)
    cross_kv = torch.randn(1, 8, dim)
    cross_mask = torch.ones(8, 8, dtype=torch.bool)

    def fwd(m):
        return m(x, self_attn_mask=None, cross_kv=cross_kv, cross_attn_mask=cross_mask)

    _assert_remap_roundtrip(
        lambda: ParallelPredictorResidual(dim, n_heads, has_cross_attn=True, parent_dim=dim),
        ParallelPredictorResidual._CROSS_REMAP,
        fwd,
    )


def test_reversible_block_remap_nested_fg():
    """Reversible block nests two ParallelPredictorResidual (F, G); legacy keys
    are F.cross_attn_q / G.cross_attn_q and must remap via each child's hook."""
    dim, n_heads = 16, 4
    x = torch.randn(1, 8, 2 * dim)
    parent = torch.randn(1, 8, dim)
    cross_mask = torch.ones(8, 8, dtype=torch.bool)

    def make():
        return ReversibleCouplingBlock(dim, n_heads, has_cross_attn=True, parent_dim=dim)

    def fwd(m):
        m.set_masks(None, cross_mask)
        m.set_parent(parent_mode="dual_stream", o1=parent, o2=parent, fused=None, layer_idx=0)
        return m(x)

    # legacy keys are nested under F./G., so prefix the mapping accordingly
    mapping = {}
    for old, new in ParallelPredictorResidual._CROSS_REMAP.items():
        mapping[f"F.{old}"] = f"F.{new}"
        mapping[f"G.{old}"] = f"G.{new}"
    _assert_remap_roundtrip(make, mapping, fwd)
