from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import QuantizerResult


def _arelbo_distortion(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    noise_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = x.shape[0]
    dim_x = x[0].numel()
    mse = F.mse_loss(x_rec, x, reduction="sum") / bs
    mse_term = mse + noise_weight
    distortion = dim_x * torch.log(mse_term.clamp(min=1e-8)) / 2
    return distortion, mse


def compute_hier_elbo_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    layer_results: List[QuantizerResult],
    progressive_recs: Optional[List[torch.Tensor]] = None,
    progressive_noise_weight: float = 0.0,
    kl_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ARELBO distortion + sum of layer aux losses (HQ-VAE)."""
    log_dict_prog: Dict[str, torch.Tensor] = {}
    if progressive_recs is not None:
        distortion = torch.zeros((), device=x.device, dtype=x.dtype)
        mse = torch.zeros((), device=x.device, dtype=x.dtype)
        for i, x_partial in enumerate(progressive_recs):
            d_i, mse_i = _arelbo_distortion(x, x_partial, progressive_noise_weight)
            distortion = distortion + d_i
            mse = mse + mse_i
            log_dict_prog[f"loss/mse_progressive_L{i + 1}"] = mse_i.detach()
        denom = max(len(progressive_recs), 1)
        # Keep progressive loss scale comparable to non-progressive training:
        # average per-progressive-stage distortion/MSE instead of summing.
        distortion = distortion / denom
        mse = mse / denom
    else:
        distortion, mse = _arelbo_distortion(x, x_rec, progressive_noise_weight)

    kl_total = torch.zeros((), device=x.device, dtype=x.dtype)
    dim_x = x[0].numel()
    log_dict: Dict[str, torch.Tensor] = {
        "loss/distortion": distortion.detach(),
        "loss/mse": mse.detach(),
        "loss/mse_per_pixel": (mse / dim_x).detach(),
        **log_dict_prog,
    }
    num_layers = len(layer_results)
    if kl_weights is None:
        kl_weights = torch.ones(num_layers, device=x.device, dtype=x.dtype)
    else:
        if kl_weights.numel() != num_layers:
            raise ValueError(
                f"kl_weights length {kl_weights.numel()} != num_layers {num_layers}"
            )
        kl_weights = kl_weights.to(x.device, dtype=x.dtype)
    for i, result in enumerate(layer_results):
        raw_kl = result.aux_loss
        weighted_kl = kl_weights[i] * raw_kl
        kl_total = kl_total + weighted_kl
        log_dict[f"loss/kl_layer_{i + 1}_raw"] = raw_kl.detach()
        log_dict[f"loss/kl_layer_{i + 1}"] = weighted_kl.detach()
        log_dict[f"train/perplexity_layer_{i + 1}"] = result.perplexity.detach()
        if "posterior_var" in result.log_stats:
            pv = result.log_stats["posterior_var"]
            if torch.is_tensor(pv):
                log_dict[f"train/posterior_var_layer_{i + 1}"] = pv.detach()

    total = distortion + kl_total
    log_dict["loss/total"] = total.detach()
    log_dict["loss/kl_total"] = kl_total.detach()
    return total, log_dict
