from src.Open_MAGVIT2.modules.diffusionmodules.norm import FrameWiseGroupNorm
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Callable, Any, List
from collections import OrderedDict
from torch.nn.modules.utils import _triple, _pair
from torch import Tensor

def swish(x):
    # swish
    return x*torch.sigmoid(x)

class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int, int]],
        causal: bool,
        padding: Union[int, Tuple[int, int, int]] = 0,
        stride: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        **kwargs: Any
    ):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        if causal:
            padding = 0
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        time_pad = kernel_size[0] - 1 if causal else kernel_size[0] // 2
        
        if causal:
            self.padding = (
                width_pad,
                width_pad,
                height_pad,
                height_pad,
                time_pad,
                0
            )
        else:
            self.padding = (
                width_pad,
                width_pad,
                height_pad,
                height_pad,
                time_pad,
                time_pad,
            )

        self.conv_1 = nn.Conv3d(in_planes, out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                        **kwargs)
        
    def forward(self, x):
        """
        CasuelConv3D
        """
        x = F.pad(x, self.padding, mode="constant")
        x = self.conv_1(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 use_agn = False,
                 num_groups: int = 32,
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn: ## agn is GroupNorm likewise skip it if has agn before
            self.norm1 = FrameWiseGroupNorm(num_groups, in_filters, eps=1e-6)
        self.norm2 = FrameWiseGroupNorm(num_groups, out_filters, eps=1e-6)

        self.conv1 = ConvBlock3D(in_filters, out_filters, kernel_size=(3, 3, 3), causal=True, padding=1, bias=False)
        self.conv2 = ConvBlock3D(out_filters, out_filters, kernel_size=(3, 3, 3), causal=True, padding=1, bias=False)
        

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = ConvBlock3D(in_filters, out_filters, kernel_size=(3, 3, 3), causal=True, padding=1, bias=False)
            else:
                self.nin_shortcut = ConvBlock3D(in_filters, out_filters, kernel_size=(1, 1, 1), causal=True, padding=0, bias=False)
    

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual
    
def _resolve_temporal_strides(
    ch_mult, temporal_downsample, num_downsamples: int
) -> List[bool]:
    """Per-downsample-level temporal-stride flags.

    Legacy behaviour couples temporal stride to ``ch_mult`` (a level with
    ``ch_mult[i] == 1`` downsamples spatially only; otherwise spatial+temporal).
    ``temporal_downsample`` (an explicit list of bools, one per downsample level)
    decouples the two so width and temporal stride can be chosen independently
    (required by the big-step 16/4/1 pyramid). ``None`` reproduces legacy exactly.
    """
    if temporal_downsample is None:
        return [ch_mult[i] != 1 for i in range(num_downsamples)]
    flags = [bool(t) for t in temporal_downsample]
    if len(flags) != num_downsamples:
        raise ValueError(
            f"temporal_downsample length {len(flags)} != number of downsample "
            f"levels {num_downsamples} (len(ch_mult) - 1)"
        )
    return flags


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4),
                resolution, double_z=False, num_groups: int = 32,
                temporal_downsample: Optional[List[bool]] = None,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution
        self.num_groups = num_groups

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        t_down = _resolve_temporal_strides(ch_mult, temporal_downsample, self.num_blocks - 1)

        self.conv_in = ConvBlock3D(in_channels,
                                   ch,
                                   kernel_size=(3, 3, 3),
                                   causal = True,
                                   padding=1,
                                   bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level] #[1, 1, 2, 2, 4]
            block_out = ch*ch_mult[i_level] #[1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, num_groups=self.num_groups))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                # spatial is always halved; temporal halves only when flagged
                # (decoupled from ch_mult — see _resolve_temporal_strides).
                t_stride = 2 if t_down[i_level] else 1
                down.downsample = ConvBlock3D(
                    block_out, block_out, kernel_size=(3, 3, 3), causal=True,
                    stride=(t_stride, 2, 2), padding=1,
                )

            self.down.append(down)
        
        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, num_groups=self.num_groups))
        
        ### end
        self.norm_out = FrameWiseGroupNorm(self.num_groups, block_out, eps=1e-6)
        self.conv_out = ConvBlock3D(block_out, z_channels, kernel_size=(1, 1, 1), causal=True)

    @staticmethod
    def _shape_key(x: torch.Tensor) -> str:
        h, w = x.shape[3], x.shape[4]
        return f"h{h}_w{w}"

    def forward(self, x, return_intermediates: bool = False):
        activations = {}

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)

            activations[self._shape_key(x)] = x

            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
                activations[self._shape_key(x)] = x

        ## mid
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        activations[self._shape_key(x)] = x

        if return_intermediates:
            return x, activations
        return x

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4),
                resolution, double_z=False, num_groups: int = 32,
                temporal_upsample: Optional[List[bool]] = None) -> None:
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        # Mirror of the encoder's per-level temporal stride; index j corresponds
        # to the upsample that reverses encoder downsample level j. None = legacy
        # (coupled to ch_mult). Unused for finest_state decoders (single 2x).
        t_up = _resolve_temporal_strides(ch_mult, temporal_upsample, self.num_blocks - 1)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_groups = num_groups

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = ConvBlock3D(
            z_channels, block_in, kernel_size=(3, 3, 3), causal=True, padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in, num_groups=self.num_groups))
        
        self.up = nn.ModuleList()

        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in, num_groups=self.num_groups))
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out, num_groups=self.num_groups))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                # spatial always doubled; temporal doubled only when flagged.
                t_factor = 2 if t_up[i_level - 1] else 1
                up.upsample = Upsampler(block_in, block_size=(t_factor, 2, 2))
            self.up.insert(0, up)
        
        self.norm_out = FrameWiseGroupNorm(self.num_groups, block_in, eps=1e-6)

        self.conv_out = ConvBlock3D(block_in, out_ch, kernel_size=(3, 3, 3), causal=True, padding=1)
    
    def forward(self, z):
        
        style = z.clone() #for adaptive groupnorm

        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z

def depth_to_space3d(x: torch.Tensor, block_size: List) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, t, h, w = x.shape[-4:]

    time_block, h_block, w_block = block_size
    s = time_block * h_block * w_block
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-4]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, time_block, h_block, w_block, c // s, t, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, t * time_block, h * h_block,
                            w * w_block)

    return x

class Upsampler(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        
        dim_out = dim
        for block_si in block_size:
            dim_out = dim_out * block_si
        
        self.block_size = block_size
        self.conv1 = ConvBlock3D(dim, dim_out, kernel_size=(3, 3, 3), causal=True, padding=1)
        self.depth2space = depth_to_space3d
    
    def forward(self, x):
        """
        input_video: [B C T H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, self.block_size)
        if self.block_size[0] > 1:
            ### drop the first s-1 frames
            if out.shape[2] > 2: #video input
                out = torch.concat([out[:, :, [0], ...], out[:, :, 2:, ...]], dim=2)
            else:
                out = out[:, :, [0], ...] #only take the first frame
        return out

class FrameWiseVideoAdaptiveGroupNorm(nn.Module):
    """Adaptive GroupNorm with per-frame style pooling (video v2)."""

    def __init__(self, z_channel: int, in_filters: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.gn = FrameWiseGroupNorm(num_groups, in_filters, eps=eps)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        style: [B, C_z, T', H', W'] — interpolated to x grid when shapes differ
        """
        b, c, t, h, w = x.shape
        if style.shape[2:] != (t, h, w):
            style = F.interpolate(
                style, size=(t, h, w), mode="trilinear", align_corners=False
            )
        style_hw = style.permute(0, 2, 1, 3, 4).reshape(b * t, style.shape[1], h, w)
        scale = style_hw.var(dim=(-2, -1), unbiased=False) + self.eps
        scale = scale.sqrt()
        scale = self.gamma(scale).view(b, t, c, 1, 1).permute(0, 2, 1, 3, 4)

        bias = style_hw.mean(dim=(-2, -1))
        bias = self.beta(bias).view(b, t, c, 1, 1).permute(0, 2, 1, 3, 4)

        x = self.gn(x)
        return scale * x + bias


# Backward-compatible alias (2D path unused in video decoder forward)
AdaptiveGroupNorm = FrameWiseVideoAdaptiveGroupNorm

if __name__ == "__main__":
    x = torch.randn(size = (2, 3, 17, 128, 128))
    encoder = Encoder(ch=128, in_channels=3, num_res_blocks=4, z_channels=18, out_ch=3, resolution=128)
    decoder = Decoder(out_ch=3, z_channels=18, num_res_blocks=4, ch=128, in_channels=3, resolution=128)
    z = encoder(x)
    out = decoder(z)