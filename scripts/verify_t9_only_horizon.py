"""Explore T=9-only encoding strategies for horizon GT (no T=13 encode)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel


def build_model():
    ddconfig = dict(
        double_z=False, z_channels=32, resolution=64, in_channels=3, out_ch=3,
        ch=64, ch_mult=[1, 2, 2, 4], num_res_blocks=2,
    )
    hierarchy = dict(
        mode="sqvae2", token_grid="native", sequence_length=9, latent_key="t3_h8_w8",
        blocks_sq="t3_h8_w8_x1,t5_h16_w16_x1,t9_h32_w32_x1",
        tap_channels=dict(t3_h8_w8=32, t5_h16_w16=128, t9_h32_w32=128),
    )
    quantizer = dict(
        type="sq", prior="zero", size_dict=[1536, 768, 384], dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3, temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )
    return VideoHierVQModel(
        ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4
    )


def encode_clip(model, video_btchw: torch.Tensor):
    """video: (1, T, C, H, W)"""
    x = video_btchw.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
    with torch.no_grad():
        return model.encode_tokens(x, flg_quant_det=True)


def main():
    torch.manual_seed(42)
    model = build_model()
    model.eval()

    # Full 13-frame synthetic clip (B,T,C,H,W)
    full = torch.randn(1, 13, 3, 64, 64)

    # Context encode: frames 0..8 (1-9)
    ctx = encode_clip(model, full[:, :9])
    bot_ctx = ctx["levels"][2]["indices"]  # t9_h32_w32

    # Bot horizon via sliding T=9 windows (last token = target frame)
    # b10=frame idx 9 -> window start 1 (frames 1..9 in 0-indexed: 1..9)? 
    # frame 10 (1-indexed) = index 9 -> window [1..9] 0-indexed = start 1, end 10
    bot_horizon = []
    for target_frame in range(9, 13):  # frames 10-13 (0-indexed 9-12)
        start = target_frame - 8  # 9 frames ending at target_frame
        win = full[:, start : target_frame + 1]
        tok = encode_clip(model, win)
        idx = tok["levels"][2]["indices"][:, -1]  # last bot token
        bot_horizon.append(idx)

    print("=== Strategy: T=9-only encodes (matches jq0ynj5m config) ===")
    print(f"Context bot shape: {tuple(bot_ctx.shape)} keys: {[lv['key'] for lv in ctx['levels']]}")

    # Compare: does context bot match first 9 of window ending at frame 9?
    win9 = encode_clip(model, full[:, :9])
    match = torch.equal(bot_ctx, win9["levels"][2]["indices"])
    print(f"Context == encode(frames 1-9): {match}")

    # GroupNorm check: is bot[8] from ctx same as last token of window ending at frame 9?
    win_to_9 = encode_clip(model, full[:, 1:10])  # frames 2-10
    b9_from_slide = win_to_9["levels"][2]["indices"][:, -2]  # token at frame 9 position
    b9_ctx = bot_ctx[:, 8]
    print(f"b9 context vs b9 from slide window [2-10]: equal={torch.equal(b9_ctx, b9_from_slide)}")

    # Index stability: horizon tokens are valid int indices
    for i, h in enumerate(bot_horizon):
        print(f"  bot frame {10+i} horizon idx shape {tuple(h.shape)} range [{h.min()}, {h.max()}]")


if __name__ == "__main__":
    main()
