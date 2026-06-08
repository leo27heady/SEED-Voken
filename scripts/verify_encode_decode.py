"""Verify encode_tokens / decode_from_indices for predictor spec."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel


def main() -> None:
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
    hierarchy = dict(
        mode="sqvae2",
        token_grid="native",
        sequence_length=13,
        latent_key="t3_h8_w8",
        blocks_sq="t3_h8_w8_x1,t5_h16_w16_x1,t9_h32_w32_x1",
        tap_channels=dict(t3_h8_w8=32, t5_h16_w16=128, t9_h32_w32=128),
    )
    quantizer = dict(
        type="sq",
        prior="zero",
        size_dict=[1536, 768, 384],
        dim_dict=[32, 32, 32],
        log_param_q_init=[4.09434] * 3,
        temperature=dict(init=1.0, decay=1e-5, min=0.3),
    )

    model = VideoHierVQModel(
        ddconfig=ddconfig,
        hierarchy=hierarchy,
        quantizer=quantizer,
        learning_rate=1e-4,
    )
    model.eval()

    for T in [9, 13]:
        print(f"\n=== T={T} ===")
        x = torch.randn(1, 3, T, 64, 64)
        with torch.no_grad():
            tokens = model.encode_tokens(x, flg_quant_det=True)

        for lv in tokens["levels"]:
            idx = lv["indices"]
            print(f"  key={lv['key']} shape={tuple(idx.shape)} K={lv['codebook_size']}")

        level_indices = [lv["indices"] for lv in tokens["levels"]]
        acts = tokens["activations"]
        h = tokens["encoder_bottleneck"]

        with torch.no_grad():
            recon = model.decode_from_indices(level_indices, acts, encoder_bottleneck=h)

        zero_acts = {k: torch.zeros_like(v) for k, v in acts.items()}
        with torch.no_grad():
            recon_zero = model.decode_from_indices(
                level_indices, zero_acts, encoder_bottleneck=torch.zeros_like(h)
            )

        mse = ((recon - recon_zero) ** 2).mean().item()
        print(f"  recon {tuple(recon.shape)} | GT-acts vs zero-acts MSE diff: {mse:.8f}")
        print(f"  -> activation VALUES unused in decode: {mse < 1e-10}")


if __name__ == "__main__":
    main()
