from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import (
    ConvBlock3D,
    ResBlock,
    Upsampler,
    swish,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import QuantizerResult
from src.Open_MAGVIT2.modules.vqvae.hierarchical.quantizer_builder import build_layer_quantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.layer_string import (
    flatten_layer_specs,
    parse_blocks_sq,
)


class InjSQBlock(nn.Module):
    def __init__(
        self,
        z_channels: int,
        act_channels: int,
        width: int,
        upsample: bool,
        quantizer: nn.Module,
    ):
        super().__init__()
        self.upsample = upsample
        self.quantizer = quantizer
        if upsample:
            self.spatial_up = Upsampler(z_channels, block_size=(1, 2, 2))
        else:
            self.spatial_up = None
        self.act_proj = (
            nn.Identity()
            if act_channels == z_channels
            else ConvBlock3D(
                act_channels, z_channels, kernel_size=(1, 1, 1), causal=True, padding=0
            )
        )
        self.posterior = nn.Sequential(
            ConvBlock3D(
                z_channels * 2, width, kernel_size=(3, 3, 3), causal=True, padding=1
            ),
            nn.ReLU(True),
            ConvBlock3D(
                width, z_channels, kernel_size=(1, 1, 1), causal=True, padding=0
            ),
        )

    def _upsample(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_up is None:
            return z
        return self.spatial_up(z)

    def forward(
        self,
        z_cur: torch.Tensor,
        activation: torch.Tensor,
        var_q: torch.Tensor,
        flg_train: bool,
        flg_quant_det: bool,
    ) -> Tuple[torch.Tensor, QuantizerResult]:
        z_pass = self._upsample(z_cur)
        act = self.act_proj(activation)
        if act.shape[2:] != z_pass.shape[2:]:
            act = F.interpolate(
                act, size=z_pass.shape[2:], mode="trilinear", align_corners=False
            )
        if z_pass.shape[2:] != act.shape[2:]:
            z_pass = F.interpolate(
                z_pass,
                size=act.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        z_feat = self.posterior(torch.cat([z_pass, act], dim=1))
        result = self.quantizer(
            z_feat,
            var_q_pos=var_q,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det,
        )
        z_cur = z_pass + result.z_q
        return z_cur, result

    def pass_through(self, z_cur: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        z_pass = self._upsample(z_cur)
        act = self.act_proj(activation)
        if act.shape[2:] != z_pass.shape[2:]:
            act = F.interpolate(
                act, size=z_pass.shape[2:], mode="trilinear", align_corners=False
            )
        if z_pass.shape[2:] != act.shape[2:]:
            z_pass = F.interpolate(
                z_pass,
                size=act.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        z_feat = self.posterior(torch.cat([z_pass, torch.zeros_like(act)], dim=1))
        return z_pass + z_feat


class SQVAE2TopDown(nn.Module):
    def __init__(
        self,
        hierarchy_cfg: Dict[str, Any],
        quantizer_cfg: Dict[str, Any],
        z_channels: int,
        width: int = 64,
    ):
        super().__init__()
        self.z_channels = z_channels
        blocks_sq = hierarchy_cfg.get("blocks_sq", "t3_h8_w8_x1,t3_h16_w16_u2")
        layer_specs = flatten_layer_specs(parse_blocks_sq(blocks_sq))
        self.num_layers = len(layer_specs)

        global_type = quantizer_cfg.get("type", "sq")
        per_layer = quantizer_cfg.get("per_layer")
        qtypes = (
            per_layer
            if per_layer
            else [global_type] * self.num_layers
        )
        size_dict = quantizer_cfg.get("size_dict", [512] * self.num_layers)
        dim_dict = quantizer_cfg.get("dim_dict", [64] * self.num_layers)
        if len(size_dict) == 1:
            size_dict = size_dict * self.num_layers
        if len(dim_dict) == 1:
            dim_dict = dim_dict * self.num_layers

        temp_init = quantizer_cfg.get("temperature", {}).get("init", 1.0)
        log_init = quantizer_cfg.get("log_param_q_init", [4.09434] * self.num_layers)
        if len(log_init) == 1:
            log_init = log_init * self.num_layers
        self.log_param_q_scalar = nn.Parameter(
            torch.tensor(log_init, dtype=torch.float32)
        )

        self.resolution_keys = [spec[0] for spec in layer_specs]
        self.layer_upsample = [spec[1] for spec in layer_specs]
        tap_channels = hierarchy_cfg.get("tap_channels", {})

        blocks = []
        for i, (res_key, upsample) in enumerate(layer_specs):
            act_ch = tap_channels.get(res_key, z_channels)
            idx_end = []
            for j, (rk, up) in enumerate(layer_specs):
                if up:
                    idx_end.append(j - 1)
            idx_end.append(len(layer_specs) - 1)
            flg_continuous = i in idx_end

            q = build_layer_quantizer(
                qtypes[i],
                size_dict=size_dict[i],
                dim_dict=dim_dict[i],
                flg_loss_continuous=flg_continuous,
                temperature=temp_init,
            )
            blocks.append(
                InjSQBlock(
                    z_channels=z_channels,
                    act_channels=act_ch,
                    width=width,
                    upsample=upsample,
                    quantizer=q,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def set_temperature(self, tau: float) -> None:
        for block in self.blocks:
            block.quantizer.set_temperature(tau)

    def _get_activation(
        self,
        activations: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        if key not in activations:
            available = ", ".join(sorted(activations.keys()))
            raise KeyError(
                f"Activation key '{key}' not found. Available: {available}"
            )
        return activations[key]

    def forward(
        self,
        activations: Union[Dict[str, torch.Tensor], torch.Tensor],
        *,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> Tuple[torch.Tensor, List[QuantizerResult]]:
        if isinstance(activations, torch.Tensor):
            raise ValueError("SQVAE2TopDown expects activation dict from encoder taps")

        first_key = self.resolution_keys[0]
        act0 = self._get_activation(activations, first_key)
        z_cur = act0.new_zeros((act0.shape[0], self.z_channels, *act0.shape[2:]))
        results: List[QuantizerResult] = []

        idx_start = 0
        for i, (res_key, _) in enumerate(
            zip(self.resolution_keys, self.layer_upsample)
        ):
            if i == 0 or self.layer_upsample[i]:
                idx_start = i
            var_q = self.log_param_q_scalar[idx_start : i + 1].exp()
            act = self._get_activation(activations, res_key)
            z_cur, result = self.blocks[i](
                z_cur, act, var_q, flg_train, flg_quant_det
            )
            results.append(result)
        return z_cur, results

    def forward_progressive(
        self,
        activations: Dict[str, torch.Tensor],
        flg_quant_det: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        first_key = self.resolution_keys[0]
        act0 = self._get_activation(activations, first_key)
        z_cur = act0.new_zeros((act0.shape[0], self.z_channels, *act0.shape[2:]))
        partial: List[torch.Tensor] = []
        idx_start = 0

        for i, res_key in enumerate(self.resolution_keys):
            if i == 0 or self.layer_upsample[i]:
                idx_start = i
            var_q = self.log_param_q_scalar[idx_start : i + 1].exp()
            act = self._get_activation(activations, res_key)
            z_cur, _ = self.blocks[i](
                z_cur, act, var_q, flg_train=False, flg_quant_det=flg_quant_det
            )
            partial.append(z_cur.clone())
        return z_cur, partial

    def pass_through_from(self, z_cur: torch.Tensor, activations: Dict[str, torch.Tensor], start_idx: int) -> torch.Tensor:
        for i in range(start_idx + 1, len(self.blocks)):
            act = self._get_activation(activations, self.resolution_keys[i])
            z_cur = self.blocks[i].pass_through(z_cur, act)
        return z_cur

    def decode_from_indices(
        self,
        indices_list: List[torch.Tensor],
        activations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        first_key = self.resolution_keys[0]
        act0 = self._get_activation(activations, first_key)
        z_cur = act0.new_zeros((act0.shape[0], self.z_channels, *act0.shape[2:]))
        for i, block in enumerate(self.blocks):
            z_q = block.quantizer.decode_indices(indices_list[i])
            z_pass = block._upsample(z_cur)
            act = self._get_activation(activations, self.resolution_keys[i])
            if z_pass.shape[2:] != act.shape[2:]:
                z_pass = F.interpolate(
                    z_pass, size=act.shape[2:], mode="trilinear", align_corners=False
                )
            z_cur = z_pass + z_q
        return z_cur
