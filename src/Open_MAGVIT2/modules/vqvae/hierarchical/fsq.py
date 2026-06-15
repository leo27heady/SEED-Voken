"""Finite Scalar Quantization (Mentzer et al. 2023) layer quantizer.

Train == eval: straight-through hard rounding in BOTH phases, so there is no
soft/hard quantization gap (unlike GaussianSQ's Gumbel-soft train / argmax eval,
which on the progressive pyramid produced a ~800x val-vs-train recon gap). No
learned codebook, no auxiliary loss, no temperature, ~uniform usage by
construction.

Drop-in ``LayerQuantizer`` sibling of ``GaussianSQQuantizer``: mirrors the
in_proj/out_proj and ``QuantizerResult``/``log_stats`` contract so
``SQVAE2TopDown``, the ELBO loss, and ``token_quality_report`` are unchanged.
``var_q_pos`` / ``z_pri`` / ``var_q_pri`` / prior are accepted and ignored.

Math (official FSQ, with the even-level offset so an even ``L`` keeps all ``L``
codes — a naive ``round((L-1)/2 * tanh)`` would collapse even levels to ``L-1``
codes):

    half_l = (L - 1) * (1 - eps) / 2
    offset = 0.5 if L even else 0.0
    shift  = atanh(offset / half_l)
    bound(h)     = tanh(h + shift) * half_l - offset        # round-bin centred
    quantized    = round_ste(bound(h))                      # integer lattice
    value        = quantized / (L // 2)                     # decoder input
    code         = quantized + (L // 2)            in {0..L-1}
    index        = sum_i code_i * basis_i,  basis = cumprod([1, L_1, ...])
"""

from math import prod
from typing import List, Optional

import torch
from torch import nn

from src.Open_MAGVIT2.modules.numerical_debug import check_tensor, numerical_debug_enabled
from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import LayerQuantizer, QuantizerResult


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient (backward grad == 1)."""
    return z + (torch.round(z) - z).detach()


class FSQLayerQuantizer(LayerQuantizer):
    def __init__(self, levels: List[int], in_channels: Optional[int] = None, eps: float = 1e-3):
        super().__init__()
        levels = [int(level) for level in levels]
        if not levels or any(level < 2 for level in levels):
            raise ValueError(f"FSQ levels must each be >= 2, got {levels}")
        if any(level == 2 for level in levels):
            raise ValueError(
                "FSQ does not support level == 2 (binary); use quantizer type 'lfq' "
                f"for binary codes. Got levels {levels}"
            )
        self.levels = levels
        self.dim_dict = len(levels)
        self.size_dict = int(prod(levels))
        self.in_channels = int(in_channels if in_channels is not None else self.dim_dict)
        # Topdown guards route z_pri/var_q_pri only when prior == "learned";
        # "zero" makes FSQ never receive them in the pyramid/native path.
        self.prior = "zero"
        self.flg_loss_continuous = False

        levels_t = torch.tensor(levels, dtype=torch.float32)
        half_l = (levels_t - 1.0) * (1.0 - eps) / 2.0
        offset = torch.where(levels_t % 2 == 0, torch.tensor(0.5), torch.tensor(0.0))
        shift = torch.atanh(offset / half_l)
        half_width = torch.div(levels_t, 2, rounding_mode="floor")  # L // 2
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)

        # non-persistent: FSQ adds no state_dict keys beyond in/out_proj params.
        self.register_buffer("_levels", levels_t, persistent=False)
        self.register_buffer("_half_l", half_l, persistent=False)
        self.register_buffer("_offset", offset, persistent=False)
        self.register_buffer("_shift", shift, persistent=False)
        self.register_buffer("_half_width", half_width, persistent=False)
        self.register_buffer("_basis", basis, persistent=False)

        if self.in_channels != self.dim_dict:
            self.in_proj = nn.Conv3d(self.in_channels, self.dim_dict, kernel_size=1)
            self.out_proj = nn.Conv3d(self.dim_dict, self.in_channels, kernel_size=1)
        else:
            self.in_proj = None
            self.out_proj = None

    # ---- projections (mirror gaussian_sq.py) ------------------------------
    def _to_codebook_space(self, z: torch.Tensor) -> torch.Tensor:
        return z if self.in_proj is None else self.in_proj(z)

    def _from_codebook_space(self, z_q: torch.Tensor) -> torch.Tensor:
        return z_q if self.out_proj is None else self.out_proj(z_q)

    def _c(self, buf: torch.Tensor) -> torch.Tensor:
        """Reshape a per-dim buffer to broadcast over (B, d, T, H, W)."""
        return buf.view(1, -1, 1, 1, 1)

    def _bound(self, h: torch.Tensor) -> torch.Tensor:
        half_l = self._c(self._half_l).to(h.dtype)
        offset = self._c(self._offset).to(h.dtype)
        shift = self._c(self._shift).to(h.dtype)
        return torch.tanh(h + shift) * half_l - offset

    def _quantize(self, h: torch.Tensor):
        """h: (B, d, T, H, W) -> (value, codes_long), both (B, d, T, H, W)."""
        quantized = round_ste(self._bound(h))
        half_width = self._c(self._half_width).to(h.dtype)
        value = quantized / half_width
        codes = (quantized + half_width).long()
        levels = self._c(self._levels).long()
        codes = codes.clamp(min=0)
        codes = torch.minimum(codes, levels - 1)  # numerical safety at tanh extremes
        return value, codes

    def _codes_to_index(self, codes: torch.Tensor) -> torch.Tensor:
        return (codes * self._c(self._basis)).sum(dim=1)  # (B, T, H, W)

    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos: Optional[torch.Tensor] = None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
        z_pri: Optional[torch.Tensor] = None,
        var_q_pri: Optional[torch.Tensor] = None,
    ) -> QuantizerResult:
        dbg = numerical_debug_enabled()
        if dbg:
            check_tensor("fsq/z_in", z, extra={"K": self.size_dict, "d": self.dim_dict})
        h = self._to_codebook_space(z)
        value, codes = self._quantize(h)
        indices = self._codes_to_index(codes)
        z_q = self._from_codebook_space(value)
        if dbg:
            check_tensor("fsq/z_q", z_q)

        # perplexity = exp(entropy of the marginal code distribution), matching SQ
        # log semantics. ~uniform by construction (G1 informational for FSQ).
        counts = torch.bincount(indices.reshape(-1), minlength=self.size_dict).float()
        p = counts / counts.sum().clamp(min=1.0)
        nz = p[p > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())

        active_codes = int((counts > 0).sum().item())
        usage_fraction = active_codes / self.size_dict
        return QuantizerResult(
            z_q=z_q,
            aux_loss=z.new_zeros(()),  # FSQ has NO auxiliary loss
            perplexity=perplexity,
            indices=indices,
            log_stats={
                # no "posterior_var" — FSQ has none (loss/Lightning guard on key presence)
                "active_codes": torch.tensor(active_codes, device=z.device, dtype=z.dtype),
                "usage_fraction": torch.tensor(usage_fraction, device=z.device, dtype=z.dtype),
                "size_dict": torch.tensor(float(self.size_dict), device=z.device, dtype=z.dtype),
            },
        )

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B,T,H,W) -> z_q (B, in_channels, T, H, W); exact inverse of forward."""
        idx = indices.long().unsqueeze(1)  # (B,1,T,H,W)
        basis = self._c(self._basis)
        levels = self._c(self._levels).long()
        codes = (idx // basis) % levels  # (B,d,T,H,W) in {0..L-1}
        half_width = self._c(self._half_width)
        value = (codes.to(half_width.dtype) - half_width) / half_width
        return self._from_codebook_space(value)
