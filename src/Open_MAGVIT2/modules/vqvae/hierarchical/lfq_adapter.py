import torch
import torch.nn.functional as F

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
        self.lfq = LFQ(
            dim=dim,
            codebook_size=codebook_size,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
        )
        self.dim = dim

    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos=None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> QuantizerResult:
        if flg_train:
            (quantized, entropy_loss, indices), breakdown = self.lfq(
                z, return_loss_breakdown=True, return_loss=True
            )
            aux_loss = entropy_loss + breakdown.commitment
            log_stats = {"commitment": breakdown.commitment.detach()}
        else:
            quantized, entropy_loss, indices = self.lfq(
                z, return_loss_breakdown=False, return_loss=False
            )
            aux_loss = torch.zeros((), device=z.device, dtype=z.dtype)
            log_stats = {}

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
        return codes.reshape(bs, self.dim, t_len, h, w)
