"""Verify context tokens match when encoding T=9 vs first 9 of T=13."""
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
    return VideoHierVQModel(ddconfig=ddconfig, hierarchy=hierarchy, quantizer=quantizer, learning_rate=1e-4)


def encode_with_spatial_keys(model, x):
    """Encode bypassing fixed t9 keys by aliasing activations."""
    with torch.no_grad():
        h, activations = model.encoder(x, return_intermediates=True)
    # Map canonical keys to actual keys via spatial suffix
    canonical = ["t3_h8_w8", "t5_h16_w16", "t9_h32_w32"]
    aliased = dict(activations)
    for ck in canonical:
        suffix = ck.split("_", 1)[1]  # h8_w8 etc
        match = [k for k in activations if k.endswith(suffix) and k != ck]
        if match and ck not in activations:
            aliased[ck] = activations[match[0]]
    with torch.no_grad():
        z_q, layer_results = model.hier_quant(
            aliased, encoder_bottleneck=h, flg_train=False, flg_quant_det=True
        )
    return model._pack_token_levels(layer_results)


def main():
    torch.manual_seed(0)
    model = build_model()
    model.eval()

    x9 = torch.randn(1, 3, 9, 64, 64)
    x13 = torch.randn(1, 3, 13, 64, 64)
    x13[..., :9, :, :] = x9  # same first 9 frames

    with torch.no_grad():
        lv9 = model.encode_tokens(x9, flg_quant_det=True)["levels"]
    lv13 = encode_with_spatial_keys(model, x13)

    print("=== Context token match (first 9 frames identical input) ===")
    for a, b in zip(lv9, lv13):
        key = a["key"]
        idx9 = a["indices"]
        idx13 = b["indices"]
        # compare context portion
        if key == "t3_h8_w8":
            ctx = idx9.shape[1]
            match = torch.equal(idx9, idx13[:, :ctx])
        elif key == "t5_h16_w16":
            ctx = idx9.shape[1]
            match = torch.equal(idx9, idx13[:, :ctx])
        else:
            ctx = 9
            match = torch.equal(idx9, idx13[:, :ctx])
        print(f"  {key}: context match={match} shapes {tuple(idx9.shape)} vs {tuple(idx13.shape)}")


if __name__ == "__main__":
    main()
