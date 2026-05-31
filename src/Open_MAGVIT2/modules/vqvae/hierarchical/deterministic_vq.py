import torch
import torch.nn.functional as F
from torch import nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import LayerQuantizer, QuantizerResult
from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import calc_distance


class DeterministicVQQuantizer(LayerQuantizer):
    def __init__(self, size_dict: int, dim_dict: int, commitment_weight: float = 0.25):
        super().__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.commitment_weight = commitment_weight
        self.codebook = nn.Parameter(torch.randn(size_dict, dim_dict))

    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos=None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> QuantizerResult:
        bs, c, t_len, h, w = z.shape
        z_pos = z.permute(0, 2, 3, 4, 1).contiguous()
        distances = calc_distance(z_pos, self.codebook, self.dim_dict)
        indices = torch.argmin(distances, dim=-1).reshape(bs, t_len, h, w)
        encodings = F.one_hot(indices.reshape(-1), self.size_dict).type_as(self.codebook)
        z_q = torch.matmul(encodings, self.codebook).reshape(bs, t_len, h, w, c)
        z_to_decoder = z_q.permute(0, 4, 1, 2, 3).contiguous()

        if flg_train:
            commit_loss = self.commitment_weight * F.mse_loss(
                z_to_decoder.detach(), z
            ) + F.mse_loss(z_to_decoder, z.detach())
        else:
            commit_loss = torch.zeros((), device=z.device, dtype=z.dtype)

        avg_probs = torch.mean(encodings.float(), dim=0)
        flat_avg = avg_probs.reshape(-1)
        perplexity = torch.exp(
            -torch.sum(flat_avg * torch.log(flat_avg + 1e-7))
        )

        return QuantizerResult(
            z_q=z_to_decoder,
            aux_loss=commit_loss,
            perplexity=perplexity,
            indices=indices,
            log_stats={},
        )

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        bs, t_len, h, w = indices.shape
        flat = indices.reshape(-1)
        encodings = F.one_hot(flat, self.size_dict).type_as(self.codebook)
        z_q = torch.matmul(encodings, self.codebook).reshape(
            bs, t_len, h, w, self.dim_dict
        )
        return z_q.permute(0, 4, 1, 2, 3).contiguous()
