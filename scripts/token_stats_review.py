"""Token stream statistics: temporal stability + marginal entropy per layer.

If coarse tokens flip chaotically frame-to-frame, the predictor cannot do better
than the marginal entropy -- evidence that the tokenizer (not the predictor) is
the bottleneck.
"""
import math
import sys

import torch
import yaml

sys.path.insert(0, ".")

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.data.shape_video import ShapeVideoDataset

CFG = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml"
CKPT = r"C:\Users\leoni\Documents\checkpoints\vqgan\shapes3d_sqvae2_32_S_lite\epoch=67-step=70856.ckpt"

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(CFG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
model = VideoHierVQModel(**cfg["model"]["init_args"])
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"], strict=False)
model = model.to(device).eval()
if model.use_ema:
    model.model_ema.copy_to(model)

ds = ShapeVideoDataset(config=cfg["data"]["init_args"]["validation"]["params"]["config"])

all_idx = {0: [], 1: [], 2: []}
with torch.no_grad():
    for start in range(0, 128, 16):
        batch = torch.stack([ds[i]["video"] for i in range(start, start + 16)]).to(device)
        tokens = model.encode_tokens(batch, flg_train=False, flg_quant_det=True)
        for s, lv in enumerate(tokens["levels"]):
            all_idx[s].append(lv["indices"].cpu())

for s in range(3):
    idx = torch.cat(all_idx[s])  # (B, T, H, W)
    b, t, h, w = idx.shape
    K = [4096, 2048, 1024][s]
    same = (idx[:, 1:] == idx[:, :-1]).float().mean().item()
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=K).float()
    p = counts / counts.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum().item()
    uniq = (counts > 0).sum().item()
    # conditional entropy of token given its previous-frame token at same position
    pairs = torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)], dim=1)
    # estimate H(next | prev) via empirical joint on observed pairs
    key = pairs[:, 0].long() * K + pairs[:, 1].long()
    uk, cnt = key.unique(return_counts=True)
    joint = cnt.float() / cnt.sum()
    prev_ids = (uk // K)
    prev_counts = torch.zeros(K)
    prev_counts.scatter_add_(0, prev_ids, cnt.float())
    p_prev = prev_counts[prev_ids] / cnt.sum()
    cond_ent = -(joint * (joint / p_prev).log()).sum().item()
    print(f"layer {s+1}: grid=({t},{h},{w}) K={K} unique={uniq} "
          f"P(token unchanged vs prev frame)={same:.3f} "
          f"H(marginal)={ent:.3f} nats (ln K={math.log(K):.2f}) "
          f"H(next|prev@same-pos)={cond_ent:.3f}")
