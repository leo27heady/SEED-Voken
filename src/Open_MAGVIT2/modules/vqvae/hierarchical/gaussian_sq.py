from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from src.Open_MAGVIT2.modules.numerical_debug import check_tensor, numerical_debug_enabled
from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import LayerQuantizer, QuantizerResult


def sample_gumbel(shape, device, eps=1e-10):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size(), logits.device)
    return F.softmax((logits + g) / temperature, dim=-1)


def calc_distance(z_continuous, codebook):
    """Pairwise squared L2 distance; z_continuous last dim must match codebook dim."""
    feat_dim = z_continuous.shape[-1]
    if feat_dim != codebook.shape[1]:
        raise ValueError(
            f"z feature dim {feat_dim} != codebook dim {codebook.shape[1]}"
        )
    # fp32 matmul avoids fp16 overflow on large codebooks (K x D dot products).
    z_flat = z_continuous.reshape(-1, feat_dim).float()
    cb = codebook.float()
    distances = (
        torch.sum(z_flat ** 2, dim=1, keepdim=True)
        + torch.sum(cb ** 2, dim=1)
        - 2 * torch.matmul(z_flat, cb.t())
    )
    return distances.to(dtype=z_continuous.dtype)


class GaussianSQQuantizer(LayerQuantizer):
    """5D stochastic quantizer (B, C, T, H, W) ported from HQ-VAE.

    Perplexity is exp(entropy) of the code distribution averaged over batch and (T, H, W),
    in the range [1, size_dict] (uniform gives size_dict).

    Prior modes (``prior`` string):
    - ``zero``: log p_prior = 0 (legacy HQ default).
    - ``uniform``: log p_prior = log(1/K).
    - ``learned``: requires ``z_pri`` (and optional ``var_q_pri``) at forward time.
    """

    def __init__(
        self,
        size_dict: int,
        dim_dict: int,
        flg_loss_continuous: bool = False,
        temperature: float = 1.0,
        prior: str = "zero",
        usage_reg_weight: float = 0.0,
        usage_reg_target_perplexity: float = 0.0,
        in_channels: int | None = None,
    ):
        super().__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.in_channels = int(in_channels if in_channels is not None else dim_dict)
        self.temperature = temperature
        self.flg_loss_continuous = flg_loss_continuous
        self.prior = prior.lower()
        self.usage_reg_weight = float(usage_reg_weight)
        self.usage_reg_target_perplexity = float(usage_reg_target_perplexity)
        self.codebook = nn.Parameter(torch.randn(size_dict, dim_dict))
        if self.in_channels != dim_dict:
            self.in_proj = nn.Conv3d(self.in_channels, dim_dict, kernel_size=1)
            self.out_proj = nn.Conv3d(dim_dict, self.in_channels, kernel_size=1)
        else:
            self.in_proj = None
            self.out_proj = None

    def _to_codebook_space(self, z: torch.Tensor) -> torch.Tensor:
        if self.in_proj is not None:
            return self.in_proj(z)
        return z

    def _from_codebook_space(self, z_q: torch.Tensor) -> torch.Tensor:
        if self.out_proj is not None:
            return self.out_proj(z_q)
        return z_q

    def set_temperature(self, tau: float) -> None:
        self.temperature = tau

    def _log_prob_prior(
        self,
        prob_pos: torch.Tensor,
        logit_pos: torch.Tensor,
        z_pri: Optional[torch.Tensor],
        var_q_pri: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.prior in ("zero", "none"):
            return torch.zeros_like(logit_pos)
        if self.prior == "uniform":
            return torch.log(torch.full_like(prob_pos, 1.0 / self.size_dict))
        if self.prior == "learned":
            if z_pri is None:
                raise ValueError("prior='learned' requires z_pri")
            bs, dim_z, t_len, h, w = z_pri.shape
            z_pri_perm = z_pri.permute(0, 2, 3, 4, 1).contiguous()
            if var_q_pri is None:
                var_q_pri = torch.tensor(1.0, device=z_pri.device, dtype=z_pri.dtype)
            if torch.numel(var_q_pri) > 1:
                var_main = var_q_pri.reshape(-1)[0]
            else:
                var_main = var_q_pri
            precision_pri = 1.0 / torch.clamp(var_main, min=1e-10)
            if z_pri_perm.shape[-1] != self.dim_dict:
                raise ValueError(
                    "learned prior z_pri must already be in codebook dim "
                    f"({self.dim_dict}), got {z_pri_perm.shape[-1]}"
                )
            distances_pri = calc_distance(z_pri_perm, self.codebook)
            logit_pri = (-0.5 * precision_pri * distances_pri).reshape(
                bs, t_len, h, w, self.size_dict
            )
            return F.log_softmax(logit_pri, dim=-1)
        raise ValueError(f"Unknown prior mode: {self.prior}")

    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos: torch.Tensor = None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
        z_pri: Optional[torch.Tensor] = None,
        var_q_pri: Optional[torch.Tensor] = None,
    ) -> QuantizerResult:
        if var_q_pos is None:
            raise ValueError("GaussianSQQuantizer requires var_q_pos")
        bs, _, t_len, h, w = z.shape
        dbg = numerical_debug_enabled()
        if dbg:
            check_tensor("sq/z_in", z, extra={"K": self.size_dict, "D": self.dim_dict})
        z_cb = self._to_codebook_space(z)
        if dbg:
            check_tensor("sq/z_cb", z_cb)
        z_pos = z_cb.permute(0, 2, 3, 4, 1).contiguous()

        if torch.numel(var_q_pos) > 1:
            var_main = var_q_pos[-1]
        else:
            var_main = var_q_pos
        precision = 1.0 / torch.clamp(var_main, min=1e-10)
        distances = calc_distance(z_pos, self.codebook)
        if dbg:
            check_tensor(
                "sq/distances",
                distances,
                extra={"max_dist": float(distances.max().item())},
            )
        logit_pos = (-0.5 * precision * distances).reshape(
            bs, t_len, h, w, self.size_dict
        )
        if dbg:
            check_tensor("sq/logit_pos", logit_pos)
        prob_pos = F.softmax(logit_pos.float(), dim=-1).to(dtype=z.dtype)
        log_prob_pos = F.log_softmax(logit_pos, dim=-1)
        log_prob_pri = self._log_prob_prior(prob_pos, logit_pos, z_pri, var_q_pri)

        if flg_train:
            indices = torch.argmax(logit_pos, dim=-1)
            encodings = gumbel_softmax_sample(logit_pos, self.temperature)
            z_q = torch.matmul(
                encodings.reshape(-1, self.size_dict), self.codebook
            ).reshape(bs, t_len, h, w, self.dim_dict)
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
            z_q = torch.matmul(encodings, self.codebook).reshape(
                bs, t_len, h, w, self.dim_dict
            )

        z_cb_out = z_q.permute(0, 4, 1, 2, 3).contiguous()
        z_to_decoder = self._from_codebook_space(z_cb_out)
        if dbg:
            check_tensor("sq/z_q", z_to_decoder)
        avg_probs_k = prob_pos.mean(dim=(0, 1, 2, 3))
        perplexity = torch.exp(
            -torch.sum(avg_probs_k * torch.log(avg_probs_k + 1e-7))
        )

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
        if dbg:
            check_tensor("sq/aux_loss", aux_loss.unsqueeze(0))

        if self.usage_reg_weight > 0.0 and self.usage_reg_target_perplexity > 0.0:
            usage_reg = self.usage_reg_weight * (
                (perplexity - self.usage_reg_target_perplexity) ** 2
            )
            aux_loss = aux_loss + usage_reg

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
        z_cb = z_q.permute(0, 4, 1, 2, 3).contiguous()
        return self._from_codebook_space(z_cb)
