"""Single predictor stage with reversible shifts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        use_reversible_backprop: bool = False,
        output_mode: str = "composite",
        levels: list[int] | None = None,
        pos_encoding: str = "absolute",
        rope_axes: tuple[int, int, int] | None = None,
        rope_base: float = 10000.0,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp",
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        from src.Open_MAGVIT2.modules.predictor.attention.blocks_common import BlockKit
        kit = BlockKit(norm_type=norm_type, mlp_type=mlp_type, qk_norm=qk_norm)
        self.dim = dim
        self.n_heads = n_heads
        self.h = h
        self.w = w
        self.n_spatial = h * w
        self.codebook_size = codebook_size
        self.codebook_dim = int(codebook_dim if codebook_dim is not None else dim)
        self.max_t = max_t
        self.has_parent = has_parent
        self.parent_mode = parent_mode
        self.attention_type = attention_type
        self.use_reversible_backprop = use_reversible_backprop

        # Positional encoding: learned-absolute (legacy default) or factorized 3D
        # RoPE (relative; length-generalizing for AR; no untrained absolute slots).
        self.pos_encoding_type = pos_encoding
        self.input_proj = nn.Linear(dim, dim)
        if pos_encoding == "absolute":
            self.spatial_pos = nn.Parameter(torch.randn(1, self.n_spatial, dim) * 0.02)
            self.temporal_pos = nn.Parameter(torch.randn(1, max_t, dim) * 0.02)
            self.rope = None
        elif pos_encoding == "rope":
            from src.Open_MAGVIT2.modules.predictor.attention.rope import Rotary3D
            head_dim = dim // n_heads
            self.rope = Rotary3D(
                head_dim, axes=rope_axes, has_spatial=self.n_spatial > 1, base=rope_base
            )
        else:
            raise ValueError(f"unknown pos_encoding {pos_encoding!r}")
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
                    kit=kit,
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
                    custom_backward=use_reversible_backprop,
                    kit=kit,
                )
                for _ in range(n_layers)
            ])
        # Output head: a single K-way softmax over the composite index ("composite"),
        # or per-FSQ-channel softmaxes ("factorized_fsq"). Factorized matches FSQ's
        # mixed-radix structure: it predicts each scalar channel (4-8 way) and gives
        # partial credit, instead of one ~impossible 1-of-K composite classification.
        self.output_mode = output_mode
        if output_mode == "factorized_fsq":
            if not levels:
                raise ValueError("output_mode='factorized_fsq' requires per-stage FSQ 'levels'")
            self.levels = [int(L) for L in levels]
            prod = 1
            for L in self.levels:
                prod *= L
            if prod != codebook_size:
                raise ValueError(
                    f"prod(levels)={prod} != codebook_size={codebook_size}"
                )
            # FSQ basis (mixed-radix): index = sum_i code_i * basis_i (matches fsq.py)
            basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.long), dim=0)
            self.register_buffer("_fsq_basis", basis, persistent=False)
            head_out_dim = int(sum(self.levels))
        elif output_mode == "composite":
            self.levels = None
            head_out_dim = codebook_size
        else:
            raise ValueError(f"unknown output_mode {output_mode!r}")
        self.head_out_dim = head_out_dim

        self.stream_fusion = build_stream_fusion(stream_fusion_mode, dim)
        self.output_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, head_out_dim))
        self.codebook_embed = nn.Embedding(codebook_size, self.codebook_dim)
        if self.codebook_dim == dim:
            self.codebook_proj = nn.Identity()
        else:
            self.codebook_proj = nn.Linear(self.codebook_dim, dim)
        self._init_stage_weights()

    def _init_stage_weights(self) -> None:
        head_linear = self.output_head[-1]
        nn.init.zeros_(head_linear.weight)
        if head_linear.bias is not None:
            nn.init.zeros_(head_linear.bias)

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
        # RoPE appliers for this shift (None under absolute PE). Built per call so
        # t_len growth during AR re-entry is handled (length generalization).
        rope_temporal = rope_spatial = rope_full = None
        if self.rope is not None:
            if self.attention_type == "factorized":
                rope_temporal = self.rope.temporal_applier(t_len)
                rope_spatial = self.rope.spatial_applier(self.h, self.w)
            else:
                rope_full = self.rope.full_applier(t_len, self.h, self.w)

        if stream_state is None:
            if context_embed is None:
                raise ValueError("shift 0 requires context_embed")
            x = self.input_proj(context_embed)
            if self.pos_encoding_type == "absolute":
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
                block.set_rope(rope_full)
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
            if self.use_reversible_backprop and self.training:
                from src.Open_MAGVIT2.modules.predictor.rev_back_prop import EfficientRevBackProp
                x = EfficientRevBackProp.apply(x, 1, list(self.rev_layers))
            else:
                for block in self.rev_layers:
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
                    rope_temporal=rope_temporal,
                    rope_spatial=rope_spatial,
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

    # ---- output-head semantics (composite vs factorized FSQ) ----------------
    def shift_cross_entropy(
        self, logits: torch.Tensor, target_index: torch.Tensor
    ) -> torch.Tensor:
        """CE for one shift's query positions. Composite: K-way CE. Factorized
        FSQ: SUM of per-channel CEs (== composite-scale CE at uniform, so the
        loss magnitude and ce_over_baseline stay comparable across modes)."""
        flat = logits.reshape(-1, logits.shape[-1]).float()
        tgt = target_index.reshape(-1)
        if self.output_mode != "factorized_fsq":
            return F.cross_entropy(flat, tgt)
        total = flat.new_zeros(())
        off = 0
        for i, L in enumerate(self.levels):
            code_i = (tgt // int(self._fsq_basis[i])) % L
            total = total + F.cross_entropy(flat[:, off : off + L], code_i)
            off += L
        return total

    def commit_index(self, logits: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        """Argmax (or sample) -> composite index. Factorized: per-channel pick
        then recompose via the FSQ basis (so the decoded value is per-channel
        best, degrading gracefully instead of landing on a far composite cell)."""
        lead = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])
        if self.output_mode != "factorized_fsq":
            if sample:
                idx = torch.multinomial(flat.softmax(dim=-1), 1).squeeze(-1)
            else:
                idx = flat.argmax(dim=-1)
            return idx.reshape(lead)
        idx = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
        off = 0
        for i, L in enumerate(self.levels):
            ch = flat[:, off : off + L]
            code_i = (
                torch.multinomial(ch.softmax(dim=-1), 1).squeeze(-1)
                if sample
                else ch.argmax(dim=-1)
            )
            idx = idx + code_i * int(self._fsq_basis[i])
            off += L
        return idx.reshape(lead)

    def mean_per_channel_accuracy(
        self, logits: torch.Tensor, target_index: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Factorized only: mean over channels of per-channel top-1 accuracy
        (the honest training-health signal; composite top-1 stays ~product-low)."""
        if self.output_mode != "factorized_fsq":
            return None
        flat = logits.reshape(-1, logits.shape[-1])
        tgt = target_index.reshape(-1)
        accs = []
        off = 0
        for i, L in enumerate(self.levels):
            code_i = (tgt // int(self._fsq_basis[i])) % L
            accs.append((flat[:, off : off + L].argmax(dim=-1) == code_i).float().mean())
            off += L
        return torch.stack(accs).mean()
