"""Verify receptive field, envelope, and horizon math for predictor spec."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def causal_conv_groups(T_in: int, k: int = 3, s: int = 2, prefix: str = "x") -> tuple[int, list]:
    pad = k - 1
    T_out = (T_in - 1) // s + 1
    groups = []
    for p in range(T_out):
        start = p * s - pad
        window = []
        for i in range(start, start + k):
            if i < 0:
                window.append("pad")
            else:
                window.append(f"{prefix}{i + 1}")
        real = list(range(max(0, start), min(T_in - 1, start + k - 1) + 1))
        groups.append({"p": p, "window": window, "real": real})
    return T_out, groups


def top_to_mid_frames(p: int, T_mid: int = 5, k: int = 3, s: int = 2) -> list[int]:
    pad = k - 1
    start = p * s - pad
    return list(range(max(0, start), min(T_mid - 1, start + k - 1) + 1))


def mid_to_bot_frames(m: int, T_bot: int = 9, k: int = 3, s: int = 2) -> list[int]:
    pad = k - 1
    start = m * s - pad
    return list(range(max(0, start), min(T_bot - 1, start + k - 1) + 1))


def envelope_order(S: int, ratio: int = 2) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []

    def visit(s: int, k: int) -> None:
        order.append((s, k))
        if s < S - 1:
            for j in range(ratio):
                visit(s + 1, k * ratio + j)

    visit(0, 0)
    return order


def main() -> None:
    print("=== User manual table: bot -> mid (T=9) ===")
    _, mid_groups = causal_conv_groups(9, prefix="b")
    expected_mid = [
        ["pad", "pad", "b1"],
        ["b1", "b2", "b3"],
        ["b3", "b4", "b5"],
        ["b5", "b6", "b7"],
        ["b7", "b8", "b9"],
    ]
    for g, exp in zip(mid_groups, expected_mid):
        ok = g["window"] == exp
        print(f"  m{g['p']+1}: {g['window']}  {'OK' if ok else 'MISMATCH expected ' + str(exp)}")

    print("\n=== User manual table: mid -> top (T=5) ===")
    _, top_groups = causal_conv_groups(5, prefix="m")
    expected_top = [
        ["pad", "pad", "m1"],
        ["m1", "m2", "m3"],
        ["m3", "m4", "m5"],
    ]
    for g, exp in zip(top_groups, expected_top):
        ok = g["window"] == exp
        print(f"  t{g['p']+1}: {g['window']}  {'OK' if ok else 'MISMATCH expected ' + str(exp)}")

    print("\n=== Top -> bot frame ranges (0-indexed) ===")
    for p in range(3):
        mid_idx = top_to_mid_frames(p)
        bot_union = sorted({f for m in mid_idx for f in mid_to_bot_frames(m)})
        print(f"  t{p+1}: mid {mid_idx} -> bot {bot_union} (1-indexed: {[x+1 for x in bot_union]})")

    print("\n=== Corner overlap: which top tokens cover m1, m3, m5? ===")
    for m in [0, 2, 4]:
        parents = [p for p in range(3) if m in top_to_mid_frames(p)]
        print(f"  m{m+1} covered by top tokens: {[f't{p+1}' for p in parents]}")

    print("\n=== Envelope order S=3 ===")
    env3 = envelope_order(3, 2)
    print(env3)
    assert env3 == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3)]

    print("\n=== Envelope order S=4 ===")
    env4 = envelope_order(4, 2)
    print(f"length={len(env4)}, expected={sum(2**s for s in range(4))}")

    print("\n=== Parent ancestry check S=3 ===")
    for s, k in env3:
        if s > 0:
            parent_k = k // 2
            print(f"  ({s},{k}) parent shift idx at s-1 = {parent_k}")

    print("\n=== T_total for T_context=9, 4 bot shifts ===")
    print("Need bot tokens at indices 9,10,11,12 -> frames 10-13 -> T_total=13")
    print(f"13 % 4 == 1: {13 % 4 == 1}")

    from Open_MAGVIT2.modules.vqvae.hierarchical.shape_audit import audit_encoder_taps

    ddconfig = dict(
        double_z=False,
        z_channels=32,
        resolution=64,
        in_channels=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
    )
    for T in [9, 13]:
        audit = audit_encoder_taps(ddconfig, T)
        keys = ["t9_h32_w32", "t5_h16_w16", "t3_h8_w8"]
        print(f"\nEncoder audit T={T}:")
        for key in keys:
            if key in audit:
                sh = audit[key]
                print(f"  {key}: T={sh[2]}, H={sh[3]}, W={sh[4]}")

    print("\n=== Shift target alignment sketch ===")
    # Bot context ends at token index 8 (frame 9). Shift k supervises token 8+k+1?
    # Spec says: native-time index equals context_end_s + k + 1
    T_ctx = 9
    for k in range(4):
        bot_token_idx = T_ctx - 1 + k + 1  # 0-indexed: after shift k, predict token at 9+k
        print(f"  bot shift {k}: target bot token index {bot_token_idx} (frame {bot_token_idx + 1})")


if __name__ == "__main__":
    main()
