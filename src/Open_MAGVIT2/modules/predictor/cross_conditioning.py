"""Unified cross-attention conditioning block (PLAN_V2 §5.5 / H4).

Previously the same `q=Linear(query); k,v=Linear(parent); attn; (~mask)` pattern
was hand-written in three places (`factorized_layer.py`, `parallel_residual.py`,
and — indirectly via F/G — `reversible_block.py`). `CrossConditioning` is the one
implementation; the call sites own the residual (factorized adds `x + out`, the
parallel-residual block folds `out` into its single residual `delta`).

`forward` returns the *raw* attention output (no internal residual, no `None`
guard) so both call sites map onto it exactly. Mask polarity is owned here:
incoming `cross_mask` is bool with True=allowed; `nn.MultiheadAttention` wants
True=disallowed, so we pass `~cross_mask` — identical to every legacy call site.

Construction order (q, k, v, attn) matches the old attribute-creation order so a
fresh seeded init is byte-identical; `remap_cross_state_dict` lets old checkpoints
(`cross_q_f`, `cross_attn_q`, …) load into the renamed submodules.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as MHA


class CrossConditioning(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.attn = MHA(dim, n_heads, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        cross_kv: torch.Tensor,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Raw cross-attention output (caller applies the residual)."""
        q = self.q(query)
        k = self.k(cross_kv)
        v = self.v(cross_kv)
        ca_mask = ~cross_mask if cross_mask is not None else None
        out, _ = self.attn(q, k, v, attn_mask=ca_mask)
        return out


def remap_cross_state_dict(
    state_dict: Dict[str, torch.Tensor], prefix: str, mapping: Dict[str, str]
) -> None:
    """In-place rename of legacy cross-attn keys under `prefix` (BC load hook).

    `mapping` maps old child name -> new child name (e.g. ``cross_q_f`` ->
    ``cross_f.q``). Matches ``prefix + old + '.'`` exactly so it is immune to the
    ``cross_attn`` / ``cross_attn_q`` shared-prefix trap."""
    for old, new in mapping.items():
        old_p = f"{prefix}{old}."
        new_p = f"{prefix}{new}."
        for key in [k for k in state_dict if k.startswith(old_p)]:
            state_dict[new_p + key[len(old_p):]] = state_dict.pop(key)
