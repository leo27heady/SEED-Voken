from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import QuantizerResult


def compute_hier_elbo_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    layer_results: List[QuantizerResult],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ARELBO distortion + sum of layer aux losses (HQ-VAE)."""
    bs = x.shape[0]
    dim_x = x[0].numel()
    mse = F.mse_loss(x_rec, x, reduction="sum") / bs
    distortion = dim_x * torch.log(mse.clamp(min=1e-8)) / 2

    kl_total = torch.zeros((), device=x.device, dtype=x.dtype)
    log_dict: Dict[str, torch.Tensor] = {
        "loss/distortion": distortion.detach(),
        "loss/mse": mse.detach(),
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
