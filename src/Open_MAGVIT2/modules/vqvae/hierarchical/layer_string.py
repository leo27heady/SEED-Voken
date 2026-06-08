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


def shape_to_key(h: int, w: int) -> str:
    return f"h{h}_w{w}"


def legacy_shape_to_key(t: int, h: int, w: int) -> str:
    return f"t{t}_h{h}_w{w}"


def parse_blocks_sq(blocks_sq: str) -> List[SQBlockSpec]:
    """
    Parse DSL like: h8_w8_x1,h16_w16_x1 or legacy t3_h8_w8_x1,t3_h16_w16_u2
    - xN: N inject layers at fixed resolution (no upsample before inject)
    - u2: upsample x2 (spatial) then inject once
    """
    specs: List[SQBlockSpec] = []
    for token in blocks_sq.split(","):
        token = token.strip()
        if not token:
            continue
        m_spatial = re.match(r"h(\d+)_w(\d+)_(x(\d+)|u2)", token)
        m_legacy = re.match(r"t(\d+)_h(\d+)_w(\d+)_(x(\d+)|u2)", token)
        if m_spatial:
            h, w = int(m_spatial.group(1)), int(m_spatial.group(2))
            key = shape_to_key(h, w)
            if m_spatial.group(3).startswith("x"):
                count = int(m_spatial.group(4))
                specs.append(SQBlockSpec(resolution_key=key, upsample=False, count=count))
            else:
                specs.append(SQBlockSpec(resolution_key=key, upsample=True, count=1))
            continue
        if m_legacy:
            warnings.warn(
                f"Legacy temporal blocks_sq token {token!r}; use spatial h{{H}}_w{{W}} format",
                DeprecationWarning,
                stacklevel=2,
            )
            key = normalize_resolution_key(f"t{m_legacy.group(1)}_h{m_legacy.group(2)}_w{m_legacy.group(3)}")
            if m_legacy.group(4).startswith("x"):
                count = int(m_legacy.group(5))
                specs.append(SQBlockSpec(resolution_key=key, upsample=False, count=count))
            else:
                specs.append(SQBlockSpec(resolution_key=key, upsample=True, count=1))
            continue
        raise ValueError(f"Invalid blocks_sq token: {token}")
    return specs


def flatten_layer_specs(specs: List[SQBlockSpec]) -> List[Tuple[str, bool]]:
    """Expand counts into per-layer (resolution_key, upsample) entries."""
    layers: List[Tuple[str, bool]] = []
    for spec in specs:
        for _ in range(spec.count):
            layers.append((spec.resolution_key, spec.upsample))
    return layers
