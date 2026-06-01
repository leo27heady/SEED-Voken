from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import QuantizerResult
from src.Open_MAGVIT2.modules.vqvae.hierarchical.quantizer_builder import build_layer_quantizer


class ResSQBlock(nn.Module):
    def __init__(self, quantizer: nn.Module):
        super().__init__()
        self.quantizer = quantizer

    def forward(
        self,
        z_cur: torch.Tensor,
        z_res: torch.Tensor,
        var_q_prefix: torch.Tensor,
        flg_train: bool,
        flg_quant_det: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, QuantizerResult]:
        result = self.quantizer(
            z_res,
            var_q_pos=var_q_prefix,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det,
        )
        z_q = result.z_q
        z_res = z_res - z_q
        z_cur = z_cur + z_q
        return z_cur, z_res, result


class RSQTopDown(nn.Module):
    def __init__(
        self,
        num_layers: int,
        z_channels: int,
        size_dict: List[int],
        dim_dict: List[int],
        quantizer_types: List[str],
        log_param_q_init: List[float],
        temperature: float = 1.0,
        flg_codebook_share: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.z_channels = z_channels
        self.flg_codebook_share = flg_codebook_share

        self.log_param_q_scalar = nn.Parameter(
            torch.tensor(log_param_q_init, dtype=torch.float32)
        )

        blocks = []
        for i in range(num_layers):
            flg_continuous = i == num_layers - 1
            idx_cb = 0 if flg_codebook_share else i
            q = build_layer_quantizer(
                quantizer_types[i],
                size_dict=size_dict[idx_cb],
                dim_dict=dim_dict[idx_cb],
                flg_loss_continuous=flg_continuous,
                temperature=temperature,
            )
            blocks.append(ResSQBlock(q))
        self.blocks = nn.ModuleList(blocks)

    def set_temperature(self, tau: float) -> None:
        for block in self.blocks:
            block.quantizer.set_temperature(tau)

    def forward(
        self,
        h: torch.Tensor,
        *,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> Tuple[torch.Tensor, List[QuantizerResult]]:
        z_cur = torch.zeros_like(h)
        z_res = h
        results: List[QuantizerResult] = []
        for i, block in enumerate(self.blocks):
            var_prefix = self.log_param_q_scalar[: i + 1].exp()
            z_cur, z_res, result = block(
                z_cur, z_res, var_prefix, flg_train, flg_quant_det
            )
            results.append(result)
        return z_cur, results

    def decode_from_indices(self, indices_list: List[torch.Tensor]) -> torch.Tensor:
        z_sum = None
        for i, block in enumerate(self.blocks):
            z_q = block.quantizer.decode_indices(indices_list[i])
            z_sum = z_q if z_sum is None else z_sum + z_q
        return z_sum

    def level_metadata(self, layer_index: int) -> Dict[str, Any]:
        block = self.blocks[layer_index]
        size = int(block.quantizer.size_dict)
        return {
            "resolution_key": f"layer_{layer_index}",
            "codebook_size": size,
        }

    def forward_progressive(
        self,
        h: torch.Tensor,
        flg_quant_det: bool = True,
        flg_train: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z_cur = torch.zeros_like(h)
        z_res = h
        partial_sums: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            var_prefix = self.log_param_q_scalar[: i + 1].exp()
            z_cur, z_res, _ = block(
                z_cur, z_res, var_prefix, flg_train=flg_train, flg_quant_det=flg_quant_det
            )
            partial_sums.append(z_cur.clone())
        return z_cur, partial_sums
