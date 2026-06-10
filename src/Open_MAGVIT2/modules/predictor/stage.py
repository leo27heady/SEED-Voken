"""Single predictor stage with reversible shifts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.Open_MAGVIT2.modules.predictor.attention.factorized_layer import FactorizedPredictorLayer
from src.Open_MAGVIT2.modules.predictor.parent_condition import ParentCondition
from src.Open_MAGVIT2.modules.predictor.reversible_block import ReversibleCouplingBlock
from src.Open_MAGVIT2.modules.predictor.schedule import ShiftMasks
from src.Open_MAGVIT2.modules.predictor.stream_fusion import build_stream_fusion


@dataclass
class ShiftOutput:
    o1: torch.Tensor
    o2: torch.Tensor
    logits: torch.Tensor


class PredictorStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_layers: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        h: int,
        w: int,
        max_t: int,
        max_shifts: int,
        has_parent: bool = False,
        parent_dim: int | None = None,
        parent_mode: str = "dual_stream",
        stream_fusion_mode: str = "conv",
        attention_type: str = "full",
        t_window: int = -1,
        spatial_window: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.n_spatial = h * w
        self.codebook_size = codebook_size
        self.codebook_dim = int(codebook_dim if codebook_dim is not None else dim)
        self.max_t = max_t
        self.has_parent = has_parent
        self.parent_mode = parent_mode
        self.attention_type = attention_type

        self.input_proj = nn.Linear(dim, dim)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.n_spatial, dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_t, dim) * 0.02)
        self.shift_embed = nn.Embedding(max_shifts, dim)

        if attention_type == "factorized":
            tw = max_t if t_window == -1 else t_window
            self.layers = nn.ModuleList([
                FactorizedPredictorLayer(
                    dim=dim,
                    n_heads=n_heads,
                    n_spatial=self.n_spatial,
                    t_window=tw,
                    spatial_window=spatial_window,
                    has_cross_attn=has_parent,
                    parent_dim=parent_dim or dim,
                )
                for _ in range(n_layers)
            ])
            self.rev_layers = None
        else:
            self.layers = None
            self.rev_layers = nn.ModuleList([
                ReversibleCouplingBlock(
                    dim=dim,
                    n_heads=n_heads,
                    has_cross_attn=has_parent,
                    parent_dim=parent_dim or dim,
                )
                for _ in range(n_layers)
            ])
        self.stream_fusion = build_stream_fusion(stream_fusion_mode, dim)
        self.output_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, codebook_size))
        self.codebook_embed = nn.Embedding(codebook_size, self.codebook_dim)
        if self.codebook_dim == dim:
            self.codebook_proj = nn.Identity()
        else:
            self.codebook_proj = nn.Linear(self.codebook_dim, dim)

    def _project_codebook_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.codebook_proj(vectors)

    def embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Flat or (B, N) token ids -> (..., dim) in predictor space."""
        return self._project_codebook_vectors(self.codebook_embed(token_ids))

    def pos_encoding(self, t_len: int) -> torch.Tensor:
        spatial = self.spatial_pos[:, : self.n_spatial]
        temporal = self.temporal_pos[:, :t_len]
        return temporal.repeat_interleave(self.n_spatial, dim=1) + spatial.repeat(1, t_len, 1)

    def execute_shift(
        self,
        shift_idx: int,
        *,
        context_embed: Optional[torch.Tensor],
        stream_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        parent: Optional[ParentCondition],
        masks: ShiftMasks,
        t_len: int,
    ) -> ShiftOutput:
        if stream_state is None:
            if context_embed is None:
                raise ValueError("shift 0 requires context_embed")
            x = self.input_proj(context_embed)
            x = x + self.pos_encoding(t_len)
            shift_vec = self.shift_embed(torch.tensor([shift_idx], device=x.device))
            x = x + shift_vec.unsqueeze(1)
            x = torch.cat([x, x], dim=-1)
        else:
            o1, o2 = stream_state
            x = torch.cat([o1, o2], dim=-1)
            shift_vec = self.shift_embed(torch.tensor([shift_idx], device=x.device))
            x[:, :, : self.dim] = x[:, :, : self.dim] + shift_vec.unsqueeze(1)
            x[:, :, self.dim :] = x[:, :, self.dim :] + shift_vec.unsqueeze(1)

        if self.rev_layers is not None:
            for layer_idx, block in enumerate(self.rev_layers):
                block.set_masks(masks.self_attn, masks.cross_attn)
                if parent is not None and parent.mode != "none":
                    block.set_parent(
                        parent_mode=parent.mode,
                        o1=parent.o1,
                        o2=parent.o2,
                        fused=parent.fused,
                        layer_idx=layer_idx,
                    )
                else:
                    block.set_parent(
                        parent_mode="none", o1=None, o2=None, fused=None, layer_idx=layer_idx
                    )
                x = block(x)
        else:
            for layer_idx, block in enumerate(self.layers):
                pmode = parent.mode if parent is not None else "none"
                x = block(
                    x,
                    self_attn_mask=masks.self_attn,
                    cross_kv=None,
                    cross_attn_mask=masks.cross_attn,
                    layer_idx=layer_idx,
                    parent_mode=pmode,
                    parent_o1=parent.o1 if parent else None,
                    parent_o2=parent.o2 if parent else None,
                    parent_fused=parent.fused if parent else None,
                )

        o1, o2 = torch.chunk(x, 2, dim=-1)
        fused = self.stream_fusion(x)
        logits = self.output_head(fused)
        return ShiftOutput(o1=o1, o2=o2, logits=logits)

    def embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B,T,H,W) -> (B, T*H*W, dim)"""
        b, t, h, w = indices.shape
        flat = indices.reshape(b, t * h * w)
        return self.embed_token_ids(flat)
