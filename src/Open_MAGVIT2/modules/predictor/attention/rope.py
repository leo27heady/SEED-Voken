"""Factorized 3D rotary position embedding (RoPE) for the predictor (PLAN_V2 §6.1).

`head_dim` is split into per-axis rotary budgets ``(D_t, D_h, D_w)`` (time gets the
largest; any remainder is left un-rotated / NoPE). Rotation is applied to q,k after
in-projection and before SDPA via the ``rope_apply`` seam in
``attention/utils.mha_self_attention``.

Because the predictor's mid/fine attention is itself FACTORIZED (separate temporal
and spatial MHA), each factorized pass rotates only its own axis budget:
  * the temporal MHA rotates ``D_t`` by frame index,
  * the spatial MHA rotates ``D_h`` by row and ``D_w`` by column.
The coarse stage uses FULL attention over the flattened (t,h,w) sequence, so its
``full_applier`` rotates all three axes by each token's 3D position. A 1x1 top stage
(``has_spatial=False``) puts the whole budget on time (``D_h=D_w=0``).

cos/sin are computed per call from the actual positions (cheap at these sizes and
naturally length-generalizing for AR rollout past ``t_total`` — RoPE scores depend
only on relative position).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def _even(x: int) -> int:
    return (x // 2) * 2


def default_axes(head_dim: int, has_spatial: bool) -> Tuple[int, int, int]:
    """Time-heavy even split of ``head_dim`` into (D_t, D_h, D_w)."""
    if not has_spatial:
        return (_even(head_dim), 0, 0)
    d_t = _even(head_dim // 2)
    rem = head_dim - d_t
    d_h = _even(rem // 2)
    d_w = _even(rem - d_h)
    return (d_t, d_h, d_w)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


RopeApply = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class Rotary3D(nn.Module):
    def __init__(
        self,
        head_dim: int,
        axes: Optional[Tuple[int, int, int]] = None,
        has_spatial: bool = True,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if axes is None:
            axes = default_axes(head_dim, has_spatial)
        d_t, d_h, d_w = (int(a) for a in axes)
        for a in (d_t, d_h, d_w):
            if a < 0 or a % 2 != 0:
                raise ValueError(f"rope axis budgets must be non-negative even, got {axes}")
        if d_t + d_h + d_w > head_dim:
            raise ValueError(f"sum(axes)={d_t + d_h + d_w} > head_dim={head_dim}")
        self.head_dim = head_dim
        self.axes = (d_t, d_h, d_w)
        self.base = float(base)
        self._offsets = {"t": 0, "h": d_t, "w": d_t + d_h}
        for name, d in (("t", d_t), ("h", d_h), ("w", d_w)):
            if d > 0:
                inv = 1.0 / (self.base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
                self.register_buffer(f"inv_{name}", inv, persistent=False)

    def _cos_sin(self, name: str, positions: torch.Tensor, dtype: torch.dtype):
        inv = getattr(self, f"inv_{name}")
        ang = torch.outer(positions.to(inv.dtype), inv)  # (seq, d/2)
        emb = torch.cat([ang, ang], dim=-1)              # (seq, d)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _rotate_slice(self, x: torch.Tensor, off: int, cos: torch.Tensor, sin: torch.Tensor):
        d = cos.shape[-1]
        seg = x[..., off : off + d]
        seg = seg * cos + _rotate_half(seg) * sin
        return torch.cat([x[..., :off], seg, x[..., off + d :]], dim=-1)

    def _rot_pair(self, q, k, name, positions):
        d = self.axes[{"t": 0, "h": 1, "w": 2}[name]]
        if d == 0:
            return q, k
        off = self._offsets[name]
        cos, sin = self._cos_sin(name, positions, q.dtype)
        return self._rotate_slice(q, off, cos, sin), self._rotate_slice(k, off, cos, sin)

    # ---- applier factories (return a (q,k)->(q,k) closure, or None) -----------
    def temporal_applier(self, t_len: int) -> Optional[RopeApply]:
        if self.axes[0] == 0:
            return None
        pos = torch.arange(t_len)

        def apply(q, k):
            return self._rot_pair(q, k, "t", pos.to(q.device))

        return apply

    def spatial_applier(self, h: int, w: int) -> Optional[RopeApply]:
        if self.axes[1] == 0 and self.axes[2] == 0:
            return None
        n = h * w
        rows = torch.arange(n) // w
        cols = torch.arange(n) % w

        def apply(q, k):
            q, k = self._rot_pair(q, k, "h", rows.to(q.device))
            q, k = self._rot_pair(q, k, "w", cols.to(q.device))
            return q, k

        return apply

    def full_applier(self, t_len: int, h: int, w: int) -> Optional[RopeApply]:
        if self.axes == (0, 0, 0):
            return None
        n = h * w
        idx = torch.arange(t_len * n)
        t_pos = idx // n
        sp = idx % n
        rows = sp // w
        cols = sp % w

        def apply(q, k):
            q, k = self._rot_pair(q, k, "t", t_pos.to(q.device))
            q, k = self._rot_pair(q, k, "h", rows.to(q.device))
            q, k = self._rot_pair(q, k, "w", cols.to(q.device))
            return q, k

        return apply
