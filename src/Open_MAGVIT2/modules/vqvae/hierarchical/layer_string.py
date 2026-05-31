import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SQBlockSpec:
    resolution_key: str
    upsample: bool
    count: int = 1


def shape_to_key(t: int, h: int, w: int) -> str:
    return f"t{t}_h{h}_w{w}"


def parse_blocks_sq(blocks_sq: str) -> List[SQBlockSpec]:
    """
    Parse DSL like: t3_h8_w8_x1,t3_h16_w16_u2
    - xN: N inject layers at fixed resolution (no upsample before inject)
    - u2: upsample x2 (spatial) then inject once
    """
    specs: List[SQBlockSpec] = []
    for token in blocks_sq.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.match(r"t(\d+)_h(\d+)_w(\d+)_(x(\d+)|u2)", token)
        if not m:
            raise ValueError(f"Invalid blocks_sq token: {token}")
        t, h, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
        key = shape_to_key(t, h, w)
        if m.group(4).startswith("x"):
            count = int(m.group(5))
            specs.append(SQBlockSpec(resolution_key=key, upsample=False, count=count))
        else:
            specs.append(SQBlockSpec(resolution_key=key, upsample=True, count=1))
    return specs


def flatten_layer_specs(specs: List[SQBlockSpec]) -> List[Tuple[str, bool]]:
    """Expand counts into per-layer (resolution_key, upsample) entries."""
    layers: List[Tuple[str, bool]] = []
    for spec in specs:
        for _ in range(spec.count):
            layers.append((spec.resolution_key, spec.upsample))
    return layers
