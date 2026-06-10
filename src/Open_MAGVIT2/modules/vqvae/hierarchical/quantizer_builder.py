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
    **_,
) -> nn.Module:
    qtype = qtype.lower()
    if qtype != "sq":
        raise ValueError(f"Only sq quantizer is supported, got {qtype!r}")
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
