import re
import warnings
from dataclasses import dataclass
from typing import List, Tuple

from src.Open_MAGVIT2.modules.vqvae.hierarchical.tap_keys import normalize_resolution_key


@dataclass
class SQBlockSpec:
    resolution_key: str
    upsample: bool
    count: int = 1
    upsample_factor: int = 2


def shape_to_key(h: int, w: int) -> str:
    return f"h{h}_w{w}"


def legacy_shape_to_key(t: int, h: int, w: int) -> str:
    return f"t{t}_h{h}_w{w}"


def _validate_upsample_factor(factor: int, token: str) -> int:
    """Spatial upsample factor must be a power of two >= 2 (the encoder/Upsampler
    are 2**n: ``depth_to_space3d`` with ``block_size`` H/W and the encoder's
    stride-2 downsample chain)."""
    if factor < 2 or (factor & (factor - 1)) != 0:
        raise ValueError(
            f"Upsample factor {factor} in token {token!r} must be a power of two >= 2"
        )
    return factor


def parse_blocks_sq(blocks_sq: str) -> List[SQBlockSpec]:
    """
    Parse DSL like: h8_w8_x1,h16_w16_x1 or legacy t3_h8_w8_x1,t3_h16_w16_u2
    - xN: N inject layers at fixed resolution (no upsample before inject)
    - uF: upsample xF (spatial, power-of-two) then inject once (e.g. u2, u4)
    """
    specs: List[SQBlockSpec] = []
    for token in blocks_sq.split(","):
        token = token.strip()
        if not token:
            continue
        m_spatial = re.match(r"h(\d+)_w(\d+)_(?:x(\d+)|u(\d+))$", token)
        m_legacy = re.match(r"t(\d+)_h(\d+)_w(\d+)_(?:x(\d+)|u(\d+))$", token)
        if m_spatial:
            h, w = int(m_spatial.group(1)), int(m_spatial.group(2))
            key = shape_to_key(h, w)
            if m_spatial.group(3) is not None:  # xN
                count = int(m_spatial.group(3))
                specs.append(SQBlockSpec(resolution_key=key, upsample=False, count=count))
            else:  # uF
                factor = _validate_upsample_factor(int(m_spatial.group(4)), token)
                specs.append(
                    SQBlockSpec(resolution_key=key, upsample=True, count=1, upsample_factor=factor)
                )
            continue
        if m_legacy:
            warnings.warn(
                f"Legacy temporal blocks_sq token {token!r}; use spatial h{{H}}_w{{W}} format",
                DeprecationWarning,
                stacklevel=2,
            )
            key = normalize_resolution_key(f"t{m_legacy.group(1)}_h{m_legacy.group(2)}_w{m_legacy.group(3)}")
            if m_legacy.group(4) is not None:  # xN
                count = int(m_legacy.group(4))
                specs.append(SQBlockSpec(resolution_key=key, upsample=False, count=count))
            else:  # uF
                factor = _validate_upsample_factor(int(m_legacy.group(5)), token)
                specs.append(
                    SQBlockSpec(resolution_key=key, upsample=True, count=1, upsample_factor=factor)
                )
            continue
        raise ValueError(f"Invalid blocks_sq token: {token}")
    return specs


def flatten_layer_specs(specs: List[SQBlockSpec]) -> List[Tuple[str, bool, int]]:
    """Expand counts into per-layer (resolution_key, upsample, upsample_factor) entries.

    ``upsample_factor`` is the spatial x-factor applied before injection for u-layers
    (2 for legacy ``u2``); it is unused for x-layers (no upsample) but reported as 1
    there for clarity."""
    layers: List[Tuple[str, bool, int]] = []
    for spec in specs:
        factor = spec.upsample_factor if spec.upsample else 1
        for _ in range(spec.count):
            layers.append((spec.resolution_key, spec.upsample, factor))
    return layers
