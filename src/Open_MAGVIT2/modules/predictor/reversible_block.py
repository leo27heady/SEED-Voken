"""Reversible coupling block for predictor stages."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import DEFAULT_KIT, BlockKit
from src.Open_MAGVIT2.modules.predictor.parallel_residual import ParallelPredictorResidual
from src.Open_MAGVIT2.modules.predictor.rev_back_prop import ReversibleModule


class ReversibleCouplingBlock(ReversibleModule):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        has_cross_attn: bool = False,
        parent_dim: int | None = None,
        mlp_ratio: int = 4,
        custom_backward: bool = True,
        kit: BlockKit = DEFAULT_KIT,
    ) -> None:
        super().__init__()
        self.custom_backward = custom_backward

        self.F = ParallelPredictorResidual(
            dim=dim, n_heads=n_heads, has_cross_attn=has_cross_attn,
            parent_dim=parent_dim, mlp_ratio=mlp_ratio, kit=kit,
        )

        self.G = ParallelPredictorResidual(
            dim=dim, n_heads=n_heads, has_cross_attn=has_cross_attn,
            parent_dim=parent_dim, mlp_ratio=mlp_ratio, kit=kit,
        )

        self._self_attn_mask = None
        self._cross_attn_mask = None
        self._cross_kv = None
        self._parent_mode = "dual_stream"
        self._parent_o1 = None
        self._parent_o2 = None
        self._rope_apply = None

    def set_rope(self, rope_apply) -> None:
        """RoPE (q,k)->(q,k) applier for the full self-attention (None = off).

        Stored as state (like masks/parent) so the custom reversible backward
        recomputes F/G with the same rotation — it's a pure function of position,
        so reuse is exact."""
        self._rope_apply = rope_apply

    def set_masks(
        self,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> None:
        self._self_attn_mask = self_attn_mask
        self._cross_attn_mask = cross_attn_mask

    def set_parent(
        self,
        *,
        parent_mode: str,
        o1: torch.Tensor | None,
        o2: torch.Tensor | None,
        fused: torch.Tensor | None,
        layer_idx: int,
    ) -> None:
        self._parent_mode = parent_mode
        self._parent_o1 = o1
        self._parent_o2 = o2

        if parent_mode == "dual_stream":
            self._cross_kv = o1 if layer_idx % 2 == 0 else o2
        else:
            self._cross_kv = fused

    def _kwargs(self) -> dict:
        return {
            "self_attn_mask": self._self_attn_mask,
            "cross_kv": self._cross_kv,
            "cross_attn_mask": self._cross_attn_mask,
            "rope_apply": self._rope_apply,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i1, i2 = torch.chunk(x, 2, dim=-1)
        kwargs = self._kwargs()

        self.seed_cuda("F")
        o2 = i2 + self.F(i1, **kwargs)
        del i2

        self.seed_cuda("G")
        o1 = i1 + self.G(o2, **kwargs)
        del i1

        return torch.cat([o1, o2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        o1, o2 = torch.chunk(y, 2, dim=-1)
        i1 = o1 - self.G(o2, **self._kwargs())
        i2 = o2 - self.F(i1, **self._kwargs())
        return torch.cat([i1, i2], dim=-1)

    def backward_pass(self, y: torch.Tensor, dy: torch.Tensor):
        o1, o2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)
        kwargs = self._kwargs()
        cross_kv = kwargs.get("cross_kv")

        with torch.enable_grad():
            o2_g = o2.detach().requires_grad_(True)
            g_kwargs = dict(kwargs)
            
            if cross_kv is not None:
                g_kwargs["cross_kv"] = (
                    cross_kv if cross_kv.requires_grad else cross_kv.detach()
                )
            
            self.seed_cuda("G", safe=True)
            g_out = self.G(o2_g, **g_kwargs)
            g_out.backward(dY_1, retain_graph=True)

        with torch.no_grad():
            i1 = o1 - g_out.detach()
            del g_out
            dY_2 = dY_2 + o2_g.grad
            o2_g.grad = None

        with torch.enable_grad():
            i1_f = i1.detach().requires_grad_(True)
            f_kwargs = dict(kwargs)
            
            if cross_kv is not None:
                f_kwargs["cross_kv"] = (
                    cross_kv if cross_kv.requires_grad else cross_kv.detach()
                )

            self.seed_cuda("F", safe=True)
            f_out = self.F(i1_f, **f_kwargs)
            f_out.backward(dY_2, retain_graph=True)

        with torch.no_grad():
            i2 = o2 - f_out.detach()
            del f_out, o2
            dY_1 = dY_1 + i1_f.grad
            i1_f.grad = None
            i1 = i1.detach()

        return torch.cat([i1, i2], dim=-1), torch.cat([dY_1, dY_2], dim=-1)
