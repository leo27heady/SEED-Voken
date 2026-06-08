"""Derive per-shift supervision indices for 64_S pyramid."""
from __future__ import annotations


def causal_groups(T_in: int, k: int = 3, s: int = 2) -> list[list[int]]:
    pad = k - 1
    T_out = (T_in - 1) // s + 1
    groups = []
    for p in range(T_out):
        start = p * s - pad
        groups.append(list(range(max(0, start), min(T_in - 1, start + k - 1) + 1)))
    return groups


def compose_token_to_frames(T_frames: int, links: list[tuple[int, int]]) -> list[list[set[int]]]:
    """Return per-stage list mapping token_idx -> set of frame indices (0-based)."""
    stages: list[list[set[int]]] = [[{i} for i in range(T_frames)]]
    grid = T_frames
    for k, s in links:
        parent_groups = causal_groups(grid, k, s)
        child = stages[-1]
        parent = []
        for grp in parent_groups:
            acc: set[int] = set()
            for ci in grp:
                acc |= child[ci]
            parent.append(acc)
        stages.append(parent)
        grid = len(parent)
    return stages


def main() -> None:
    links = [(3, 2), (3, 2)]
    T_total = 13
    T_ctx = 9

    rf = compose_token_to_frames(T_total, links)
    names = ["bot", "mid", "top"]
    for si, stage in enumerate(rf):
        print(f"\n{names[si]} T={len(stage)} (T_total={T_total})")
        for ti, frames in enumerate(stage):
            in_ctx = frames <= set(range(T_ctx))
            tag = "CTX" if in_ctx else "HOR"
            print(f"  [{ti}] frames {[f+1 for f in sorted(frames)]} {tag}")

    print("\n=== Context token counts (frames subset of 1..9) ===")
    for si, name in enumerate(names):
        ctx = [i for i, fr in enumerate(rf[si]) if fr <= set(range(T_ctx))]
        print(f"  {name}: indices {ctx} (count={len(ctx)})")

    print("\n=== Shift targets (native +1 per shift from context end) ===")
    shifts = [1, 2, 4]
    for si, (name, n_shifts) in enumerate(zip(names, shifts)):
        ctx_tokens = [i for i, fr in enumerate(rf[si]) if fr <= set(range(T_ctx))]
        ctx_end = max(ctx_tokens) if ctx_tokens else -1
        print(f"  {name} context_end_idx={ctx_end}")
        for k in range(n_shifts):
            tgt = ctx_end + k + 1
            fr = rf[si][tgt] if tgt < len(rf[si]) else None
            print(f"    shift {k} -> token {tgt} frames {[x+1 for x in sorted(fr)] if fr else 'OOR'}")

    print("\n=== Cross-attn: mid corners -> top parents ===")
    top = rf[2]
    mid = rf[1]
    for mi in [0, 2, 4]:
        parents = [pi for pi, pf in enumerate(top) if mid[mi] & pf]
        print(f"  m{mi+1} frames {[x+1 for x in sorted(mid[mi])]} <- top {[f't{pi+1}' for pi in parents]}")

    print("\n=== KEY RESOLUTION ISSUE ===")
    print("At T=13 encoder keys: t13_h32_w32, t7_h16_w16, t4_h8_w8")
    print("Checkpoint blocks_sq:   t9_h32_w32,  t5_h16_w16,  t3_h8_w8")
    print("=> Need spatial-only key resolution (h32_w32 etc.) for T_total != T_train")


if __name__ == "__main__":
    main()
