from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import (
    ConvBlock3D,
    Upsampler,
)
from src.Open_MAGVIT2.modules.numerical_debug import check_tensor, numerical_debug_enabled
from src.Open_MAGVIT2.modules.vqvae.hierarchical.base import QuantizerResult
from src.Open_MAGVIT2.modules.vqvae.hierarchical.tap_keys import (
    normalize_resolution_key,
    resolve_activation_key,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.layer_string import (
    flatten_layer_specs,
    parse_blocks_sq,
)
from src.Open_MAGVIT2.modules.vqvae.hierarchical.prior_net import GaussianPriorHead
from src.Open_MAGVIT2.modules.vqvae.hierarchical.quantizer_builder import build_layer_quantizer


def align_spatial(z: torch.Tensor, target_thw: Tuple[int, int, int]) -> torch.Tensor:
    if z.shape[2:] == target_thw:
        return z
    return F.interpolate(
        z, size=target_thw, mode="trilinear", align_corners=False
    )


def fuse_to_latent(z_q: torch.Tensor, latent_thw: Tuple[int, int, int]) -> torch.Tensor:
    return align_spatial(z_q, latent_thw)


def _tag_result(
    result: QuantizerResult,
    resolution_key: str,
    grid_shape: Tuple[int, int, int],
) -> QuantizerResult:
    result.resolution_key = resolution_key
    result.grid_shape = grid_shape
    return result


def _expand_scalar_or_list(value, num_layers: int, name: str):
    if value is None:
        return [0.0] * num_layers
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return [value[0]] * num_layers
        if len(value) != num_layers:
            raise ValueError(f"{name} length {len(value)} != num_layers {num_layers}")
        return list(value)
    return [value] * num_layers


def _resolve_sq_prior_string(prior_cfg) -> str:
    if isinstance(prior_cfg, dict):
        mode = prior_cfg.get("mode", "zero").lower()
    else:
        mode = str(prior_cfg).lower()
    if mode in ("learned", "learned_chain"):
        return "learned"
    if mode == "uniform":
        return "uniform"
    return "zero"


def _codebook_size(quantizer: nn.Module) -> int:
    if hasattr(quantizer, "size_dict"):
        return int(quantizer.size_dict)
    if hasattr(quantizer, "lfq"):
        return int(quantizer.lfq.codebook_size)
    raise AttributeError("Cannot infer codebook size from quantizer")


class InjSQBlock(nn.Module):
    """Upsample z_state then inject via posterior MLP (HQ u2 path)."""

    def __init__(
        self,
        z_channels: int,
        act_channels: int,
        width: int,
        quantizer: nn.Module,
    ):
        super().__init__()
        self.quantizer = quantizer
        self.spatial_up = Upsampler(z_channels, block_size=(1, 2, 2))
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

    def forward(
        self,
        z_state: torch.Tensor,
        activation: torch.Tensor,
        var_q: torch.Tensor,
        flg_train: bool,
        flg_quant_det: bool,
        z_pri: Optional[torch.Tensor] = None,
        var_q_pri: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, QuantizerResult]:
        z_pass = self.spatial_up(z_state)
        act = self.act_proj(activation)
        if act.shape[2:] != z_pass.shape[2:]:
            act = align_spatial(act, z_pass.shape[2:])
        z_feat = self.posterior(torch.cat([z_pass, act], dim=1))
        q_kwargs = {}
        if getattr(self.quantizer, "prior", None) == "learned":
            q_kwargs["z_pri"] = z_pri
            q_kwargs["var_q_pri"] = var_q_pri
        result = self.quantizer(
            z_feat,
            var_q_pos=var_q,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det,
            **q_kwargs,
        )
        z_state = z_pass + result.z_q
        return z_state, result

    def pass_through(self, z_state: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        z_pass = self.spatial_up(z_state)
        act = self.act_proj(activation)
        if act.shape[2:] != z_pass.shape[2:]:
            act = align_spatial(act, z_pass.shape[2:])
        z_feat = self.posterior(torch.cat([z_pass, torch.zeros_like(act)], dim=1))
        return z_pass + z_feat


class SQResSQBlock(nn.Module):
    """Residual SQ at native tap grid (xN); fuses contributions into latent_key grid."""

    def __init__(
        self,
        z_channels: int,
        act_channels: int,
        quantizer: nn.Module,
        token_grid: str,
    ):
        super().__init__()
        self.token_grid = token_grid
        self.quantizer = quantizer
        self.act_proj = (
            nn.Identity()
            if act_channels == z_channels
            else ConvBlock3D(
                act_channels, z_channels, kernel_size=(1, 1, 1), causal=True, padding=0
            )
        )

    def forward(
        self,
        z_latent: torch.Tensor,
        z_state: Optional[torch.Tensor],
        activation: torch.Tensor,
        var_q: torch.Tensor,
        latent_thw: Tuple[int, int, int],
        flg_train: bool,
        flg_quant_det: bool,
        layer_index: int,
        z_pri: Optional[torch.Tensor] = None,
        var_q_pri: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], QuantizerResult]:
        act = self.act_proj(activation)
        native_thw = act.shape[2:]

        if self.token_grid == "native":
            if layer_index == 0:
                z_res = act
            else:
                z_on_act = align_spatial(z_latent, native_thw)
                z_res = act - z_on_act
            q_kwargs = {}
            if getattr(self.quantizer, "prior", None) == "learned":
                q_kwargs["z_pri"] = z_pri
                q_kwargs["var_q_pri"] = var_q_pri
            result = self.quantizer(
                z_res,
                var_q_pos=var_q,
                flg_train=flg_train,
                flg_quant_det=flg_quant_det,
                **q_kwargs,
            )
            z_latent = z_latent + fuse_to_latent(result.z_q, latent_thw)
            return z_latent, z_state, result

        if z_state is None:
            raise ValueError("pyramid mode requires z_state for ResSQ blocks")
        act_aligned = align_spatial(act, z_state.shape[2:])
        z_res = act_aligned - z_state
        q_kwargs = {}
        if getattr(self.quantizer, "prior", None) == "learned":
            q_kwargs["z_pri"] = z_pri
            q_kwargs["var_q_pri"] = var_q_pri
        result = self.quantizer(
            z_res,
            var_q_pos=var_q,
            flg_train=flg_train,
            flg_quant_det=flg_quant_det,
            **q_kwargs,
        )
        z_state = z_state + result.z_q
        z_latent = z_latent + fuse_to_latent(result.z_q, latent_thw)
        return z_latent, z_state, result


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
        self.token_grid = hierarchy_cfg.get("token_grid", "native").lower()
        if self.token_grid not in ("native", "pyramid"):
            raise ValueError(f"token_grid must be 'native' or 'pyramid', got {self.token_grid}")

        blocks_sq = hierarchy_cfg.get("blocks_sq", "t3_h8_w8_x1,t3_h16_w16_u2")
        layer_specs = flatten_layer_specs(parse_blocks_sq(blocks_sq))
        self.num_layers = len(layer_specs)
        self.resolution_keys = [spec[0] for spec in layer_specs]
        self.layer_upsample = [spec[1] for spec in layer_specs]
        self.has_u2 = any(self.layer_upsample)

        latent_key = hierarchy_cfg.get("latent_key")
        if latent_key is None:
            latent_key = self.resolution_keys[0]
        self.latent_key = normalize_resolution_key(latent_key)
        self.tap_key_format = hierarchy_cfg.get("tap_key_format", "spatial").lower()
        if latent_key not in self.resolution_keys and hierarchy_cfg.get("tap_channels"):
            pass
        tap_channels = hierarchy_cfg.get("tap_channels", {})

        global_type = quantizer_cfg.get("type", "sq")
        per_layer = quantizer_cfg.get("per_layer")
        qtypes = per_layer if per_layer else [global_type] * self.num_layers
        size_dict = quantizer_cfg.get("size_dict", [512] * self.num_layers)
        dim_dict = quantizer_cfg.get("dim_dict", [64] * self.num_layers)
        if len(size_dict) == 1:
            size_dict = size_dict * self.num_layers
        if len(dim_dict) == 1:
            dim_dict = dim_dict * self.num_layers

        temp_init = quantizer_cfg.get("temperature", {}).get("init", 1.0)
        prior_cfg = quantizer_cfg.get("prior", "zero")
        sq_prior = _resolve_sq_prior_string(prior_cfg)
        if isinstance(prior_cfg, dict):
            self.prior_detach = bool(prior_cfg.get("detach_conditioning", False))
            prior_width = int(prior_cfg.get("width", width))
        else:
            self.prior_detach = False
            prior_width = width
        self.use_learned_prior = sq_prior == "learned"
        usage_reg_weights = _expand_scalar_or_list(
            quantizer_cfg.get("usage_reg_weight", 0.0), self.num_layers, "usage_reg_weight"
        )
        usage_reg_targets = _expand_scalar_or_list(
            quantizer_cfg.get("usage_reg_target_perplexity", 0.0),
            self.num_layers,
            "usage_reg_target_perplexity",
        )
        lfq_sample = quantizer_cfg.get("sample_minimization_weight", 1.0)
        lfq_batch = quantizer_cfg.get("batch_maximization_weight", 1.0)
        log_init = quantizer_cfg.get("log_param_q_init", [4.09434] * self.num_layers)
        if len(log_init) == 1:
            log_init = log_init * self.num_layers
        self._has_sq_layers = any(q == "sq" for q in qtypes)
        log_tensor = torch.tensor(log_init, dtype=torch.float32)
        if self._has_sq_layers:
            self.log_param_q_scalar = nn.Parameter(log_tensor)
        else:
            self.register_buffer("log_param_q_scalar", log_tensor)

        idx_end: List[int] = []
        for j, (_, up) in enumerate(layer_specs):
            if up:
                idx_end.append(j - 1)
        idx_end.append(len(layer_specs) - 1)
        self._idx_end_set = set(idx_end)

        lc_cfg = quantizer_cfg.get("flg_loss_continuous", "auto")

        prior_heads = nn.ModuleDict()
        blocks = []
        for i, (res_key, upsample) in enumerate(layer_specs):
            act_ch = tap_channels.get(res_key, z_channels)
            if lc_cfg == "auto":
                flg_continuous = i in self._idx_end_set
            elif isinstance(lc_cfg, bool):
                flg_continuous = lc_cfg
            elif isinstance(lc_cfg, (list, tuple)):
                if len(lc_cfg) != self.num_layers:
                    raise ValueError(
                        f"flg_loss_continuous list length {len(lc_cfg)} != num_layers {self.num_layers}"
                    )
                flg_continuous = bool(lc_cfg[i])
            else:
                raise ValueError(
                    f"flg_loss_continuous must be 'auto', bool, or list, got {lc_cfg!r}"
                )
            q = build_layer_quantizer(
                qtypes[i],
                size_dict=size_dict[i],
                dim_dict=dim_dict[i],
                flg_loss_continuous=flg_continuous,
                temperature=temp_init,
                commitment_weight=quantizer_cfg.get("commitment_weight", 0.25),
                lfq_sample_min_weight=lfq_sample,
                lfq_batch_max_weight=lfq_batch,
                prior=sq_prior if qtypes[i] == "sq" else "zero",
                usage_reg_weight=usage_reg_weights[i],
                usage_reg_target_perplexity=usage_reg_targets[i],
                in_channels=z_channels,
            )
            if self.use_learned_prior and qtypes[i] == "sq":
                # Conditioning tensors use act_proj → z_channels (see _compute_prior_fields).
                if upsample:
                    prior_in_ch = z_channels * 2
                else:
                    prior_in_ch = z_channels if i == 0 else z_channels * 2
                prior_heads[str(i)] = GaussianPriorHead(
                    prior_in_ch, dim_dict[i], width=prior_width
                )
            if upsample:
                if self.token_grid == "native":
                    raise ValueError(
                        "u2 layers require token_grid: pyramid (or use xN-only native config)"
                    )
                blocks.append(
                    InjSQBlock(
                        z_channels=z_channels,
                        act_channels=act_ch,
                        width=width,
                        quantizer=q,
                    )
                )
            else:
                blocks.append(
                    SQResSQBlock(
                        z_channels=z_channels,
                        act_channels=act_ch,
                        quantizer=q,
                        token_grid=self.token_grid,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.prior_heads = prior_heads

        if self.token_grid == "pyramid" and not self.has_u2:
            raise ValueError("token_grid pyramid requires at least one u2 layer in blocks_sq")

    def set_temperature(self, tau: float) -> None:
        for block in self.blocks:
            block.quantizer.set_temperature(tau)

    def _get_activation(
        self,
        activations: Dict[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        resolved = resolve_activation_key(
            activations, key, tap_key_format=self.tap_key_format
        )
        return activations[resolved]

    def _latent_thw(self, activations: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        act = self._get_activation(activations, self.latent_key)
        return act.shape[2], act.shape[3], act.shape[4]

    def _resolve_latent_thw(
        self,
        activations: Dict[str, torch.Tensor],
        encoder_bottleneck: Optional[torch.Tensor] = None,
    ) -> Tuple[int, int, int]:
        if encoder_bottleneck is not None:
            return encoder_bottleneck.shape[2], encoder_bottleneck.shape[3], encoder_bottleneck.shape[4]
        return self._latent_thw(activations)

    def _compute_prior_fields(
        self,
        layer_index: int,
        act: torch.Tensor,
        z_latent: torch.Tensor,
        z_state: Optional[torch.Tensor],
        block: nn.Module,
        var_q: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        key = str(layer_index)
        if key not in self.prior_heads:
            return None, var_q
        head = self.prior_heads[key]
        if isinstance(block, InjSQBlock):
            if z_state is None:
                return None, var_q
            z_pass = block.spatial_up(z_state)
            act_a = block.act_proj(act)
            if act_a.shape[2:] != z_pass.shape[2:]:
                act_a = align_spatial(act_a, z_pass.shape[2:])
            cond = torch.cat([z_pass, act_a], dim=1)
        else:
            act_a = block.act_proj(act)
            if layer_index == 0:
                cond = act_a
            else:
                cond = torch.cat(
                    [align_spatial(z_latent, act_a.shape[2:]), act_a], dim=1
                )
        if self.prior_detach:
            cond = cond.detach()
        return head(cond), var_q

    def forward(
        self,
        activations: Union[Dict[str, torch.Tensor], torch.Tensor],
        *,
        encoder_bottleneck: Optional[torch.Tensor] = None,
        flg_train: bool = True,
        flg_quant_det: bool = False,
    ) -> Tuple[torch.Tensor, List[QuantizerResult]]:
        if isinstance(activations, torch.Tensor):
            raise ValueError("SQVAE2TopDown expects activation dict from encoder taps")

        latent_thw = self._resolve_latent_thw(activations, encoder_bottleneck)
        ref = encoder_bottleneck if encoder_bottleneck is not None else self._get_activation(
            activations, self.latent_key
        )
        z_latent = ref.new_zeros((ref.shape[0], self.z_channels, *latent_thw))

        z_state: Optional[torch.Tensor] = None
        if self.has_u2:
            act0 = self._get_activation(activations, self.resolution_keys[0])
            z_state = act0.new_zeros((act0.shape[0], self.z_channels, *act0.shape[2:]))

        results: List[QuantizerResult] = []
        idx_start = 0

        for i, res_key in enumerate(self.resolution_keys):
            if i == 0 or self.layer_upsample[i]:
                idx_start = i
            var_q = self.log_param_q_scalar[idx_start : i + 1].exp()
            act = self._get_activation(activations, res_key)
            block = self.blocks[i]
            grid_shape = act.shape[2:]

            z_pri, var_q_pri = self._compute_prior_fields(
                i, act, z_latent, z_state, block, var_q
            )
            if isinstance(block, InjSQBlock):
                z_state, result = block(
                    z_state,
                    act,
                    var_q,
                    flg_train,
                    flg_quant_det,
                    z_pri=z_pri,
                    var_q_pri=var_q_pri,
                )
                z_latent = z_latent + fuse_to_latent(
                    align_spatial(result.z_q, latent_thw), latent_thw
                )
                grid_shape = result.z_q.shape[2:]
            else:
                z_latent, z_state, result = block(
                    z_latent,
                    z_state,
                    act,
                    var_q,
                    latent_thw,
                    flg_train,
                    flg_quant_det,
                    layer_index=i,
                    z_pri=z_pri,
                    var_q_pri=var_q_pri,
                )
                grid_shape = result.indices.shape[1:]

            results.append(_tag_result(result, res_key, grid_shape))
            if numerical_debug_enabled():
                layer = i + 1
                check_tensor(
                    f"hier/L{layer}_{res_key}/z_latent",
                    z_latent,
                    extra={"var_q": float(var_q[-1].item())},
                )
                check_tensor(
                    f"hier/L{layer}_{res_key}/aux_loss",
                    result.aux_loss.reshape(1),
                )

        return z_latent, results

    def forward_progressive(
        self,
        activations: Dict[str, torch.Tensor],
        encoder_bottleneck: Optional[torch.Tensor] = None,
        flg_quant_det: bool = True,
        flg_train: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        latent_thw = self._resolve_latent_thw(activations, encoder_bottleneck)
        ref = encoder_bottleneck if encoder_bottleneck is not None else self._get_activation(
            activations, self.latent_key
        )
        z_latent = ref.new_zeros((ref.shape[0], self.z_channels, *latent_thw))
        z_state: Optional[torch.Tensor] = None
        if self.has_u2:
            act0 = self._get_activation(activations, self.resolution_keys[0])
            z_state = act0.new_zeros((act0.shape[0], self.z_channels, *act0.shape[2:]))

        partial: List[torch.Tensor] = []
        idx_start = 0

        for i, res_key in enumerate(self.resolution_keys):
            if i == 0 or self.layer_upsample[i]:
                idx_start = i
            var_q = self.log_param_q_scalar[idx_start : i + 1].exp()
            act = self._get_activation(activations, res_key)
            block = self.blocks[i]

            z_pri, var_q_pri = self._compute_prior_fields(
                i, act, z_latent, z_state, block, var_q
            )
            if isinstance(block, InjSQBlock):
                z_state, result = block(
                    z_state,
                    act,
                    var_q,
                    flg_train=flg_train,
                    flg_quant_det=flg_quant_det,
                    z_pri=z_pri,
                    var_q_pri=var_q_pri,
                )
                z_latent = z_latent + fuse_to_latent(
                    align_spatial(result.z_q, latent_thw), latent_thw
                )
            else:
                z_latent, z_state, _ = block(
                    z_latent,
                    z_state,
                    act,
                    var_q,
                    latent_thw,
                    flg_train=flg_train,
                    flg_quant_det=flg_quant_det,
                    layer_index=i,
                    z_pri=z_pri,
                    var_q_pri=var_q_pri,
                )
            partial.append(z_latent.clone())

        return z_latent, partial

    def decode_from_indices(
        self,
        indices_list: List[torch.Tensor],
        activations: Dict[str, torch.Tensor],
        encoder_bottleneck: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_thw = self._resolve_latent_thw(activations, encoder_bottleneck)
        ref = encoder_bottleneck if encoder_bottleneck is not None else self._get_activation(
            activations, self.latent_key
        )
        z_latent = ref.new_zeros((ref.shape[0], self.z_channels, *latent_thw))
        z_state: Optional[torch.Tensor] = None
        if self.has_u2:
            act0 = self._get_activation(activations, self.resolution_keys[0])
            z_state = act0.new_zeros((act0.shape[0], self.z_channels, *act0.shape[2:]))

        for i, block in enumerate(self.blocks):
            indices = indices_list[i]
            act = self._get_activation(activations, self.resolution_keys[i])
            z_q = block.quantizer.decode_indices(indices)

            if isinstance(block, InjSQBlock):
                z_pass = block.spatial_up(z_state)
                if z_pass.shape[2:] != z_q.shape[2:]:
                    z_pass = align_spatial(z_pass, z_q.shape[2:])
                z_state = z_pass + z_q
                z_latent = z_latent + fuse_to_latent(
                    align_spatial(z_q, latent_thw), latent_thw
                )
            else:
                if block.token_grid == "pyramid" and z_state is not None:
                    z_state = z_state + z_q
                z_latent = z_latent + fuse_to_latent(z_q, latent_thw)

        return z_latent

    def level_metadata(self, layer_index: int) -> Dict[str, Any]:
        block = self.blocks[layer_index]
        return {
            "resolution_key": self.resolution_keys[layer_index],
            "codebook_size": _codebook_size(block.quantizer),
        }
