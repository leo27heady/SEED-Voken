from typing import List, Optional

from torch import nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import GaussianSQQuantizer


def build_layer_quantizer(
    qtype: str,
    size_dict: int,
    dim_dict: int,
    flg_loss_continuous: bool = False,
    temperature: float = 1.0,
    prior: str = "zero",
    usage_reg_weight: float = 0.0,
    usage_reg_target_perplexity: float = 0.0,
    in_channels: int | None = None,
    temporal_kl_weight: float = 0.0,
    levels: Optional[List[int]] = None,
    **_,
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
            temporal_kl_weight=temporal_kl_weight,
        )
    if qtype == "fsq":
        if levels is None:
            raise ValueError("fsq quantizer requires per-layer 'levels'")
        from src.Open_MAGVIT2.modules.vqvae.hierarchical.fsq import FSQLayerQuantizer

        return FSQLayerQuantizer(levels=levels, in_channels=in_channels)
    raise ValueError(f"Unknown quantizer type {qtype!r} (supported: sq, fsq)")
