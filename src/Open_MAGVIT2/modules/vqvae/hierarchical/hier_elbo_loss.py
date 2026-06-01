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
        mse = mse / max(len(progressive_recs), 1)
    else:
        distortion, mse = _arelbo_distortion(x, x_rec, progressive_noise_weight)

    kl_total = torch.zeros((), device=x.device, dtype=x.dtype)
    log_dict: Dict[str, torch.Tensor] = {
        "loss/distortion": distortion.detach(),
        "loss/mse": mse.detach(),
        **log_dict_prog,
    }
    for i, result in enumerate(layer_results):
        kl_total = kl_total + result.aux_loss
        log_dict[f"loss/kl_layer_{i + 1}"] = result.aux_loss.detach()
        log_dict[f"train/perplexity_layer_{i + 1}"] = result.perplexity.detach()
        if "posterior_var" in result.log_stats:
            pv = result.log_stats["posterior_var"]
            if torch.is_tensor(pv):
                log_dict[f"train/posterior_var_layer_{i + 1}"] = pv.detach()

    total = distortion + kl_total
    log_dict["loss/total"] = total.detach()
    log_dict["loss/kl_total"] = kl_total.detach()
    return total, log_dict
