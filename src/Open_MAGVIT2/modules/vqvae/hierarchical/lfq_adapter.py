import math

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import ConvBlock3D
from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import LayerQuantizer, QuantizerResult
from src.Open_MAGVIT2.modules.vqvae.lookup_free_quantize import LFQ


class LFQAdapter(LayerQuantizer):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        sample_minimization_weight: float = 1.0,
        batch_maximization_weight: float = 1.0,
    ):
        super().__init__()
        if not (codebook_size > 0 and (codebook_size & (codebook_size - 1)) == 0):
            raise ValueError("LFQ codebook_size must be a power of 2")
        self.dim = dim
        self.codebook_size = codebook_size
        self.lfq_dim = int(math.log2(codebook_size))

        self.in_proj = None
        self.out_proj = None
        if dim != self.lfq_dim:
            self.in_proj = ConvBlock3D(
                dim, self.lfq_dim, kernel_size=(1, 1, 1), causal=True, padding=0
            )
            self.out_proj = ConvBlock3D(
                self.lfq_dim, dim, kernel_size=(1, 1, 1), causal=True, padding=0
            )

        self.lfq = LFQ(
            dim=self.lfq_dim,
            codebook_size=codebook_size,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
        )

    def _to_lfq(self, z: torch.Tensor) -> torch.Tensor:
        if self.in_proj is not None:
            return self.in_proj(z)
        return z

    def _from_lfq(self, z: torch.Tensor) -> torch.Tensor:
        if self.out_proj is not None:
            return self.out_proj(z)
        return z

    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos=None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> QuantizerResult:
        z_lfq = self._to_lfq(z)
        if flg_train:
            (quantized, entropy_loss, indices), breakdown = self.lfq(
                z_lfq, return_loss_breakdown=True, return_loss=True
            )
            aux_loss = entropy_loss + breakdown.commitment
            log_stats = {"commitment": breakdown.commitment.detach()}
        else:
            quantized, entropy_loss, indices = self.lfq(
                z_lfq, return_loss_breakdown=False, return_loss=False
            )
            aux_loss = torch.zeros((), device=z.device, dtype=z.dtype)
            log_stats = {}

        quantized = self._from_lfq(quantized)

        if isinstance(indices, tuple):
            indices = indices[0]

        bs, _, t_len, h, w = z.shape
        indices = indices.reshape(bs, t_len, h, w)

        perplexity = torch.tensor(float(self.lfq.codebook_size), device=z.device)
        return QuantizerResult(
            z_q=quantized,
            aux_loss=aux_loss,
            perplexity=perplexity,
            indices=indices,
            log_stats=log_stats,
        )

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.lfq.decode(indices.reshape(-1))
        bs, t_len, h, w = indices.shape
        z_lfq = codes.reshape(bs, self.lfq_dim, t_len, h, w)
        return self._from_lfq(z_lfq)
