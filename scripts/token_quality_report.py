"""Path A token-quality gate (plan §4.2).

Consolidates the review's layer-ablation and token-statistics probes into one
CLI with pass/fail gates. Run after every tokenizer training run, BEFORE any
predictor work:

    python scripts/token_quality_report.py \
        --config configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_pyr.yaml \
        --ckpt <best.ckpt> [--n-videos 128] [--json out.json]

Gates (see docs/wandb_analysis/baseline_lite_2026-06-11.md for the baseline):
  G1 perplexity / K            >= --min-perplexity-frac   for every layer
  G2 token persistence         coarse > mid > fine AND coarse >= --min-coarse-persistence
  G3 ablation delta-MSE order  coarse >= mid >= fine
  G4 recon MSE (per pixel)     <= --max-recon-mse

Exit code 0 = all gates pass, 1 = at least one fails, 2 = error.
"""

import argparse
import json
import math
import sys

import torch
import yaml

sys.path.insert(0, ".")

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel  # noqa: E402
from src.Open_MAGVIT2.data.shape_video import ShapeVideoDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-videos", type=int, default=128, help="videos for token stats")
    p.add_argument("--ablation-videos", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--json", default=None, help="write metrics to this JSON file")
    p.add_argument("--min-perplexity-frac", type=float, default=0.10)
    p.add_argument("--min-coarse-persistence", type=float, default=0.5)
    # 0.0028 was the KL-free baseline AE's distortion; a properly KL-regularized
    # ELBO model sits at a different rate-distortion point. 0.005 keeps shapes
    # clearly recognizable while leaving room for real KL pressure.
    p.add_argument("--max-recon-mse", type=float, default=0.005)
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_model(config_path, ckpt_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = VideoHierVQModel(**cfg["model"]["init_args"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = model.to(device).eval()
    if model.use_ema:
        model.model_ema.copy_to(model)
        print("[info] using EMA weights")
    return model, cfg


def collect_indices(model, ds, n_videos, batch_size, device):
    all_idx = None
    sizes = None
    for start in range(0, n_videos, batch_size):
        batch = torch.stack(
            [ds[i]["video"] for i in range(start, min(start + batch_size, n_videos))]
        ).to(device)
        with torch.no_grad():
            tokens = model.encode_tokens(batch, flg_quant_det=True)
        if all_idx is None:
            all_idx = {s: [] for s in range(len(tokens["levels"]))}
            sizes = [lv["codebook_size"] for lv in tokens["levels"]]
        for s, lv in enumerate(tokens["levels"]):
            all_idx[s].append(lv["indices"].cpu())
    return {s: torch.cat(v) for s, v in all_idx.items()}, sizes


def token_stats(idx, K):
    b, t, h, w = idx.shape
    persistence = (idx[:, 1:] == idx[:, :-1]).float().mean().item() if t > 1 else float("nan")
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=K).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    marginal_entropy = -(nz * nz.log()).sum().item()
    perplexity = math.exp(marginal_entropy)
    unique = int((counts > 0).sum().item())
    # conditional entropy H(next | prev token at same position)
    cond_entropy = float("nan")
    if t > 1:
        pairs_prev = idx[:, :-1].reshape(-1).long()
        pairs_next = idx[:, 1:].reshape(-1).long()
        key = pairs_prev * K + pairs_next
        uk, cnt = key.unique(return_counts=True)
        joint = cnt.float() / cnt.sum()
        prev_ids = uk // K
        prev_counts = torch.zeros(K)
        prev_counts.scatter_add_(0, prev_ids, cnt.float())
        p_prev = prev_counts[prev_ids] / cnt.sum()
        cond_entropy = -(joint * (joint / p_prev).log()).sum().item()
    mi = marginal_entropy - cond_entropy if cond_entropy == cond_entropy else float("nan")
    return {
        "grid": [t, h, w],
        "K": K,
        "unique_codes": unique,
        "perplexity": perplexity,
        "perplexity_frac": perplexity / K,
        "persistence": persistence,
        "marginal_entropy_nats": marginal_entropy,
        "cond_entropy_nats": cond_entropy,
        # temporal mutual information: how many nats the previous token at the
        # same position gives about the next one. More robust than exact-match
        # persistence when posteriors are sharp over many neighboring codes.
        "temporal_mi_nats": mi,
    }


def ablation(model, ds, n_videos, device, sizes):
    batch = torch.stack([ds[i]["video"] for i in range(n_videos)]).to(device)
    with torch.no_grad():
        tokens = model.encode_tokens(batch, flg_quant_det=True)
        gt_indices = [lv["indices"] for lv in tokens["levels"]]
        acts, bottleneck = tokens["activations"], tokens["encoder_bottleneck"]
        base = model.decode_from_indices(
            gt_indices, acts, encoder_bottleneck=bottleneck
        ).clamp(-1, 1)
        base_mse = torch.mean((base - batch) ** 2).item()

        g = torch.Generator(device="cpu").manual_seed(0)
        deltas, keep_only = [], []
        for i in range(len(gt_indices)):
            ab = [x.clone() for x in gt_indices]
            ab[i] = torch.randint(0, sizes[i], gt_indices[i].shape, generator=g).to(device)
            rec = model.decode_from_indices(ab, acts, encoder_bottleneck=bottleneck).clamp(-1, 1)
            deltas.append(torch.mean((rec - batch) ** 2).item() - base_mse)

            ko = [
                gt_indices[j].clone() if j == i
                else torch.randint(0, sizes[j], gt_indices[j].shape, generator=g).to(device)
                for j in range(len(gt_indices))
            ]
            rec = model.decode_from_indices(ko, acts, encoder_bottleneck=bottleneck).clamp(-1, 1)
            keep_only.append(torch.mean((rec - batch) ** 2).item())
    return base_mse, deltas, keep_only


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    model, cfg = load_model(args.config, args.ckpt, device)
    qtype = str(
        cfg["model"]["init_args"].get("quantizer", {}).get("type", "sq")
    ).lower()
    if qtype in ("fsq", "lfq"):
        print(
            f"[note] quantizer={qtype}: G1 (perplexity/K) is INFORMATIONAL — usage is "
            "~uniform by construction; G2/G3/G4 are the binding signal."
        )
    data_cfg = cfg["data"]["init_args"]["validation"]["params"]["config"]
    ds = ShapeVideoDataset(config=data_cfg)
    n_videos = min(args.n_videos, len(ds))

    idx_by_layer, sizes = collect_indices(model, ds, n_videos, args.batch_size, device)
    stats = {s: token_stats(idx_by_layer[s], sizes[s]) for s in idx_by_layer}
    base_mse, deltas, keep_only = ablation(
        model, ds, min(args.ablation_videos, len(ds)), device, sizes
    )

    S = len(stats)
    print(f"\n## Token quality report ({n_videos} val videos)\n")
    print("| layer | grid | K | unique | ppl | ppl/K | persistence | H(marg) | H(next\\|prev) | MI | abl dMSE | keep-only MSE |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in range(S):
        st = stats[s]
        print(
            f"| L{s + 1} | {tuple(st['grid'])} | {st['K']} | {st['unique_codes']} "
            f"| {st['perplexity']:.1f} | {st['perplexity_frac']:.3f} "
            f"| {st['persistence']:.3f} | {st['marginal_entropy_nats']:.2f} "
            f"| {st['cond_entropy_nats']:.2f} | {st['temporal_mi_nats']:.2f} "
            f"| {deltas[s]:+.4f} | {keep_only[s]:.4f} |"
        )
    print(f"\nrecon MSE (per pixel, {args.ablation_videos} videos): {base_mse:.5f}")

    gates = {}
    gates["G1_perplexity_frac"] = all(
        stats[s]["perplexity_frac"] >= args.min_perplexity_frac for s in range(S)
    )
    pers = [stats[s]["persistence"] for s in range(S)]
    gates["G2_persistence"] = (
        all(pers[i] > pers[i + 1] for i in range(S - 1))
        and pers[0] >= args.min_coarse_persistence
    )
    gates["G3_ablation_order"] = all(deltas[i] >= deltas[i + 1] for i in range(S - 1))
    gates["G4_recon_mse"] = base_mse <= args.max_recon_mse

    print("\n## Gates\n")
    for name, ok in gates.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    result = {
        "config": args.config,
        "ckpt": args.ckpt,
        "n_videos": n_videos,
        "stats": {f"layer_{s + 1}": stats[s] for s in range(S)},
        "recon_mse": base_mse,
        "ablation_delta_mse": deltas,
        "keep_only_mse": keep_only,
        "gates": gates,
        "all_pass": all(gates.values()),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[info] wrote {args.json}")

    print(f"\nRESULT: {'ALL GATES PASS' if result['all_pass'] else 'GATE FAILURE'}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"[error] {exc}")
        raise SystemExit(2)
