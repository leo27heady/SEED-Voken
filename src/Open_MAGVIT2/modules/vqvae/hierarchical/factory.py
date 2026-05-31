from typing import Any, Dict, List, Optional

import torch
from torch import nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.quantizer_builder import (
    _resolve_quantizer_types,
    build_layer_quantizer,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.video_inj_topdown import SQVAE2TopDown
from src.Open_MAGVIT2.modules.vqvae.hierarchical.video_rsq_topdown import RSQTopDown


def build_top_down(
    hierarchy_cfg: Dict[str, Any],
    quantizer_cfg: Dict[str, Any],
    z_channels: int,
    width: int = 64,
) -> nn.Module:
    mode = hierarchy_cfg.get("mode", "sqvae2").lower()
    global_type = quantizer_cfg.get("type", "sq").lower()
    per_layer = quantizer_cfg.get("per_layer")
    temp_init = quantizer_cfg.get("temperature", {}).get("init", 1.0)

    if mode == "rsqvae":
        if per_layer and any(t.lower() == "lfq" for t in per_layer):
            raise ValueError("LFQ quantizer is not supported in RSQ mode")
        if global_type == "lfq":
            raise ValueError("LFQ quantizer is not supported in RSQ mode")

        num_layers = hierarchy_cfg.get("num_layers")
        if num_layers is None:
            blocks_sq = hierarchy_cfg.get("blocks_sq", "8x4")
            if "x" in blocks_sq:
                num_layers = int(blocks_sq.split("x")[-1])
            else:
                num_layers = 4

        size_dict = quantizer_cfg.get("size_dict", [512] * num_layers)
        dim_dict = quantizer_cfg.get("dim_dict", [64] * num_layers)
        if len(size_dict) == 1:
            size_dict = size_dict * num_layers
        if len(dim_dict) == 1:
            dim_dict = dim_dict * num_layers

        qtypes = _resolve_quantizer_types(global_type, num_layers, per_layer)
        log_init = quantizer_cfg.get(
            "log_param_q_init", [4.09434] * num_layers
        )
        if len(log_init) == 1:
            log_init = log_init * num_layers

        return RSQTopDown(
            num_layers=num_layers,
            z_channels=z_channels,
            size_dict=size_dict,
            dim_dict=dim_dict,
            quantizer_types=qtypes,
            log_param_q_init=log_init,
            temperature=temp_init,
            flg_codebook_share=quantizer_cfg.get("flg_codebook_share", False),
        )

    if mode == "sqvae2":
        return SQVAE2TopDown(
            hierarchy_cfg=hierarchy_cfg,
            quantizer_cfg=quantizer_cfg,
            z_channels=z_channels,
            width=width,
        )

    raise ValueError(f"Unknown hierarchy mode: {mode}")


def set_all_sq_temperature(module: nn.Module, tau: float) -> None:
    for m in module.modules():
        if hasattr(m, "set_temperature") and callable(m.set_temperature):
            m.set_temperature(tau)
