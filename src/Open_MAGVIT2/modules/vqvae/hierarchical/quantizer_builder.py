import math
from typing import List, Optional, Tuple

from torch import nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import GaussianSQQuantizer


def resolve_quantizer_sizes(quantizer_cfg: dict) -> Tuple[List[int], List[int]]:
    """Per-layer ``(size_dict, dim_dict)`` from a quantizer config, FSQ/LFQ-aware.

    SQ configs carry explicit ``size_dict``/``dim_dict``. FSQ/LFQ configs carry
    per-layer ``levels`` instead (the builder needs only those), so the predictor
    — which keys off ``size_dict``/``dim_dict`` — derives ``K=prod(levels)`` and
    ``d=len(levels)`` per layer. Without this, building a predictor on an FSQ
    tokenizer config raises ``KeyError: 'size_dict'``."""
    if "size_dict" in quantizer_cfg and "dim_dict" in quantizer_cfg:
        sizes = quantizer_cfg["size_dict"]
        dims = quantizer_cfg["dim_dict"]
        sizes = [sizes] if isinstance(sizes, int) else list(sizes)
        dims = [dims] if isinstance(dims, int) else list(dims)
        return sizes, dims
    qtype = str(quantizer_cfg.get("type", "sq")).lower()
    if qtype in ("fsq", "lfq"):
        levels = quantizer_cfg.get("levels")
        if levels is None:
            raise ValueError(f"{qtype} quantizer requires per-layer 'levels'")
        per_layer = levels if isinstance(levels[0], (list, tuple)) else [levels]
        sizes = [int(math.prod(lv)) for lv in per_layer]
        dims = [len(lv) for lv in per_layer]
        return sizes, dims
    raise KeyError(
        "quantizer config has neither size_dict/dim_dict nor fsq/lfq levels: "
        f"{sorted(quantizer_cfg)}"
    )


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
