from typing import List, Optional

from torch import nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.deterministic_vq import DeterministicVQQuantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import GaussianSQQuantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.lfq_adapter import LFQAdapter


def _resolve_quantizer_types(
    global_type: str,
    num_layers: int,
    per_layer: Optional[List[str]] = None,
) -> List[str]:
    if per_layer is not None:
        if len(per_layer) != num_layers:
            raise ValueError(
                f"per_layer length {len(per_layer)} != num_layers {num_layers}"
            )
        return list(per_layer)
    return [global_type] * num_layers


def build_layer_quantizer(
    qtype: str,
    size_dict: int,
    dim_dict: int,
    flg_loss_continuous: bool = False,
    temperature: float = 1.0,
    commitment_weight: float = 0.25,
    lfq_sample_min_weight: float = 1.0,
    lfq_batch_max_weight: float = 1.0,
    prior: str = "zero",
    usage_reg_weight: float = 0.0,
    usage_reg_target_perplexity: float = 0.0,
    in_channels: int | None = None,
) -> nn.Module:
    qtype = qtype.lower()
    if qtype == "sq":
        return GaussianSQQuantizer(
            size_dict=size_dict,
            dim_dict=dim_dict,
            flg_loss_continuous=flg_loss_continuous,
            temperature=temperature,
            prior=prior,
            usage_reg_weight=usage_reg_weight,
            usage_reg_target_perplexity=usage_reg_target_perplexity,
            in_channels=in_channels,
        )
    if qtype == "vq":
        return DeterministicVQQuantizer(
            size_dict=size_dict,
            dim_dict=dim_dict,
            commitment_weight=commitment_weight,
            in_channels=in_channels,
        )
    if qtype == "lfq":
        return LFQAdapter(
            dim=dim_dict,
            codebook_size=size_dict,
            sample_minimization_weight=lfq_sample_min_weight,
            batch_maximization_weight=lfq_batch_max_weight,
        )
    raise ValueError(f"Unknown quantizer type: {qtype}")
