from typing import Any, Dict

import torch
from torch import nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.video_inj_topdown import SQVAE2TopDown


def build_top_down(
    hierarchy_cfg: Dict[str, Any],
    quantizer_cfg: Dict[str, Any],
    z_channels: int,
    width: int = 64,
) -> nn.Module:
    mode = hierarchy_cfg.get("mode", "sqvae2").lower()
    if mode != "sqvae2":
        raise ValueError(f"Only sqvae2 hierarchy mode is supported, got {mode!r}")
    return SQVAE2TopDown(
        hierarchy_cfg=hierarchy_cfg,
        quantizer_cfg=quantizer_cfg,
        z_channels=z_channels,
        width=width,
    )


def set_all_sq_temperature(module: nn.Module, tau: float) -> None:
    for m in module.modules():
        if hasattr(m, "set_temperature") and callable(m.set_temperature):
            m.set_temperature(tau)
