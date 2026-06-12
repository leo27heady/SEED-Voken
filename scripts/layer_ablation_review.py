"""Empirical check: how much does each hierarchy layer contribute to reconstruction?

For each layer: replace its indices with random ones and measure reconstruction MSE
against (a) ground-truth video and (b) the unablated reconstruction.
Also measures the norm of each layer's fused contribution to z_latent.
"""
import sys

import torch
import yaml

sys.path.insert(0, ".")

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.data.shape_video import ShapeVideoDataset
from src.Open_MAGVIT2.modules.vqvae.hierarchical.video_inj_topdown import fuse_to_latent

CFG = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml"
CKPT = r"C:\Users\leoni\Documents\checkpoints\vqgan\shapes3d_sqvae2_32_S_lite\epoch=67-step=70856.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

with open(CFG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]["init_args"]
model = VideoHierVQModel(**model_cfg)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
model = model.to(device).eval()

# use EMA weights as in validation
if model.use_ema:
    model.model_ema.copy_to(model)

data_cfg = cfg["data"]["init_args"]["validation"]["params"]["config"]
ds = ShapeVideoDataset(config=data_cfg)
batch = torch.stack([ds[i]["video"] for i in range(16)]).to(device)
print("batch:", tuple(batch.shape))

with torch.no_grad():
    tokens = model.encode_tokens(batch, flg_train=False, flg_quant_det=True)
    levels = tokens["levels"]
    acts = tokens["activations"]
    bottleneck = tokens["encoder_bottleneck"]

    for i, lv in enumerate(levels):
        idx = lv["indices"]
        uniq = idx.unique().numel()
        print(f"layer {i+1}: key={lv['key']} grid={tuple(idx.shape[1:])} K={lv['codebook_size']} "
              f"unique_codes_in_batch={uniq}")

    # per-layer contribution norm to z_latent
    latent_thw = bottleneck.shape[2:]
    contribs = []
    for i, block in enumerate(model.hier_quant.blocks):
        z_q = block.quantizer.decode_indices(levels[i]["indices"])
        fused = fuse_to_latent(z_q, latent_thw)
        contribs.append(fused)
        print(f"layer {i+1}: ||fused contribution|| (rms) = {fused.pow(2).mean().sqrt().item():.4f}")
    z_latent = sum(contribs)
    print(f"z_latent rms = {z_latent.pow(2).mean().sqrt().item():.4f}")

    gt_indices = [lv["indices"] for lv in levels]
    base = model.decode_from_indices(gt_indices, acts, encoder_bottleneck=bottleneck).clamp(-1, 1)
    base_mse = torch.mean((base - batch) ** 2).item()
    print(f"\nbaseline recon MSE vs GT video: {base_mse:.6f}")

    g = torch.Generator(device="cpu").manual_seed(0)
    for i, lv in enumerate(levels):
        ab = [x.clone() for x in gt_indices]
        rand = torch.randint(0, lv["codebook_size"], lv["indices"].shape, generator=g).to(device)
        ab[i] = rand
        rec = model.decode_from_indices(ab, acts, encoder_bottleneck=bottleneck).clamp(-1, 1)
        mse_gt = torch.mean((rec - batch) ** 2).item()
        mse_base = torch.mean((rec - base) ** 2).item()
        print(f"ablate layer {i+1} (random codes): MSE vs GT = {mse_gt:.6f}  "
              f"(delta {mse_gt - base_mse:+.6f}), MSE vs baseline recon = {mse_base:.6f}")

    # also: keep ONLY one layer (others random)
    for i, lv in enumerate(levels):
        ab = []
        for j, lv2 in enumerate(levels):
            if j == i:
                ab.append(gt_indices[j].clone())
            else:
                ab.append(torch.randint(0, lv2["codebook_size"], lv2["indices"].shape, generator=g).to(device))
        rec = model.decode_from_indices(ab, acts, encoder_bottleneck=bottleneck).clamp(-1, 1)
        mse_gt = torch.mean((rec - batch) ** 2).item()
        print(f"keep ONLY layer {i+1}: MSE vs GT = {mse_gt:.6f}")
