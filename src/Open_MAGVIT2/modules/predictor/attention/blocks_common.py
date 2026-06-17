"""Stability/precision kit for predictor attention blocks (PLAN_V2 §6.4).

Three flags, bundled into a frozen ``BlockKit`` so they thread through the block
hierarchy as a single argument. All default to the legacy choices so existing
configs/checkpoints are byte-identical (the golden CE fixture stays green):

  * ``norm_type``: ``"layernorm"`` (legacy) | ``"rmsnorm"``
  * ``mlp_type``:  ``"mlp"`` (legacy GELU) | ``"swiglu"``
  * ``qk_norm``:   parameter-free RMSNorm on q,k before SDPA/RoPE (default off)

``qk_norm`` is parameter-free (no new state) and applied *before* the rotary step
("norm then rotate"), inside ``attention/utils.mha_self_attention``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.w_in = nn.Linear(dim, hidden * 2)
        self.w_out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


def build_norm(dim: int, norm_type: str, eps: float = 1e-5) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"unknown norm_type {norm_type!r}")


def build_mlp(dim: int, mlp_ratio: int, mlp_type: str) -> nn.Module:
    if mlp_type == "mlp":
        return nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
    if mlp_type == "swiglu":
        return SwiGLU(dim, mlp_ratio)
    raise ValueError(f"unknown mlp_type {mlp_type!r}")


def mlp_out_linear(mlp: nn.Module) -> nn.Linear:
    """The final projection of an MLP built by ``build_mlp`` (for zero-init)."""
    if isinstance(mlp, SwiGLU):
        return mlp.w_out
    return mlp[-1]


def maybe_qk_norm(q: torch.Tensor, k: torch.Tensor, enabled: bool, eps: float = 1e-6):
    """Parameter-free RMSNorm over head_dim on q and k (no-op when disabled)."""
    if not enabled:
        return q, k

    def _n(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    return _n(q), _n(k)


@dataclass(frozen=True)
class BlockKit:
    norm_type: str = "layernorm"
    mlp_type: str = "mlp"
    qk_norm: bool = False


DEFAULT_KIT = BlockKit()
