import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import LayerQuantizer, QuantizerResult


def sample_gumbel(shape, device, eps=1e-10):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size(), logits.device)
    return F.softmax((logits + g) / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_flat = z_continuous.reshape(-1, dim_dict)
    distances = (
        torch.sum(z_flat ** 2, dim=1, keepdim=True)
        + torch.sum(codebook ** 2, dim=1)
        - 2 * torch.matmul(z_flat, codebook.t())
    )
    return distances


class GaussianSQQuantizer(LayerQuantizer):
    """5D stochastic quantizer (B, C, T, H, W) ported from HQ-VAE.

    Perplexity is exp(entropy) of the code distribution averaged over batch and (T, H, W),
    in the range [1, size_dict] (uniform gives size_dict).
    """

    def __init__(
        self,
        size_dict: int,
        dim_dict: int,
        flg_loss_continuous: bool = False,
        temperature: float = 1.0,
        prior: str = "zero",
    ):
        super().__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
        self.flg_loss_continuous = flg_loss_continuous
        self.prior = prior.lower()
        self.codebook = nn.Parameter(torch.randn(size_dict, dim_dict))

    def set_temperature(self, tau: float) -> None:
        self.temperature = tau

    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos: torch.Tensor = None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> QuantizerResult:
        if var_q_pos is None:
            raise ValueError("GaussianSQQuantizer requires var_q_pos")
        bs, dim_z, t_len, h, w = z.shape
        z_pos = z.permute(0, 2, 3, 4, 1).contiguous()

        if torch.numel(var_q_pos) > 1:
            var_main = var_q_pos[-1]
        else:
            var_main = var_q_pos
        precision = 1.0 / torch.clamp(var_main, min=1e-10)
        distances = calc_distance(z_pos, self.codebook, self.dim_dict)
        logit_pos = (-0.5 * precision * distances).reshape(
            bs, t_len, h, w, self.size_dict
        )
        prob_pos = F.softmax(logit_pos, dim=-1)
        log_prob_pos = F.log_softmax(logit_pos, dim=-1)
        if self.prior == "zero":
            log_prob_pri = torch.zeros_like(log_prob_pos)
        else:
            log_prob_pri = torch.log(
                torch.full_like(prob_pos, 1.0 / self.size_dict)
            )

        if flg_train:
            indices = torch.argmax(logit_pos, dim=-1)
            encodings = gumbel_softmax_sample(logit_pos, self.temperature)
            z_q = torch.matmul(
                encodings.reshape(-1, self.size_dict), self.codebook
            ).reshape(bs, t_len, h, w, dim_z)
            avg_probs = torch.mean(prob_pos.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit_pos, dim=-1)
                encodings = F.one_hot(
                    indices.reshape(-1), num_classes=self.size_dict
                ).type_as(self.codebook)
                avg_probs = torch.mean(encodings, dim=0)
            else:
                flat_prob = prob_pos.reshape(-1, self.size_dict)
                dist = Categorical(flat_prob)
                indices = dist.sample().reshape(bs, t_len, h, w)
                encodings = F.one_hot(indices.reshape(-1), self.size_dict).type_as(
                    self.codebook
                )
                avg_probs = torch.mean(prob_pos, dim=0)
            z_q = torch.matmul(encodings, self.codebook).reshape(bs, t_len, h, w, dim_z)

        z_to_decoder = z_q.permute(0, 4, 1, 2, 3).contiguous()
        kld_discrete = (
            prob_pos * (log_prob_pos - log_prob_pri)
        ).mean(dim=(1, 2, 3, 4)).mean()

        if self.flg_loss_continuous:
            precision_sum = 1.0 / torch.clamp(var_q_pos.sum(), min=1e-10)
            kld_continuous = (
                (z - z_to_decoder).pow(2).mean(dim=(1, 2, 3, 4))
                * (0.5 * precision_sum)
            ).mean()
            aux_loss = kld_discrete + kld_continuous
        else:
            aux_loss = kld_discrete

        avg_probs_k = prob_pos.mean(dim=(0, 1, 2, 3))
        perplexity = torch.exp(
            -torch.sum(avg_probs_k * torch.log(avg_probs_k + 1e-7))
        )

        posterior_var = var_main.detach() if torch.is_tensor(var_main) else var_main
        active_codes = indices.unique().numel()
        usage_fraction = float(active_codes) / float(self.size_dict)
        return QuantizerResult(
            z_q=z_to_decoder,
            aux_loss=aux_loss,
            perplexity=perplexity,
            indices=indices,
            log_stats={
                "posterior_var": posterior_var,
                "avg_probs": avg_probs_k,
                "active_codes": torch.tensor(active_codes, device=z.device, dtype=z.dtype),
                "usage_fraction": torch.tensor(
                    usage_fraction, device=z.device, dtype=z.dtype
                ),
            },
        )

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        bs, t_len, h, w = indices.shape
        flat = indices.reshape(-1)
        encodings = F.one_hot(flat, num_classes=self.size_dict).type_as(self.codebook)
        z_q = torch.matmul(encodings, self.codebook).reshape(bs, t_len, h, w, self.dim_dict)
        return z_q.permute(0, 4, 1, 2, 3).contiguous()
