from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class QuantizerResult:
    z_q: torch.Tensor
    aux_loss: torch.Tensor
    perplexity: torch.Tensor
    indices: torch.Tensor
    log_stats: Dict[str, Any] = field(default_factory=dict)
    resolution_key: Optional[str] = None
    grid_shape: Optional[Tuple[int, int, int]] = None


class LayerQuantizer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        *,
        var_q_pos: Optional[torch.Tensor] = None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> QuantizerResult:
        raise NotImplementedError()

    @abstractmethod
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def set_temperature(self, tau: float) -> None:
        pass
