"""Fetch W&B history for runs for review analysis.

Usage:
    python scripts/fetch_wandb_review.py [label=entity/project/runs/id ...] [--out DIR]

With no args, fetches the 2026-06 review baseline runs.
"""
import argparse
import json
import sys

import pandas as pd
import wandb

DEFAULT_RUNS = {
    "encoder_idomjiua": "/leo27heady/seed-voken-shapes3d/runs/idomjiua",
    "predictor_wgm3d46g": "/leo27heady/seed-voken-shapes3d/runs/wgm3d46g",
}

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="*", help="label=entity/project/runs/id")
parser.add_argument("--out", default="docs/wandb_analysis")
args = parser.parse_args()

RUNS = (
    {spec.split("=", 1)[0]: spec.split("=", 1)[1] for spec in args.runs}
    if args.runs
    else DEFAULT_RUNS
)
OUT_DIR = args.out

api = wandb.Api(timeout=60)

for label, path in RUNS.items():
    run = api.run(path.lstrip("/"))
    print(f"=== {label} ===")
    print("name:", run.name, "| state:", run.state)
    print("created:", run.created_at)
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    print("summary keys:", sorted(summary.keys()))
    with open(f"{OUT_DIR}/{label}_summary.json", "w") as f:
        json.dump({k: str(v) for k, v in summary.items()}, f, indent=2)
    cfg = {k: v for k, v in run.config.items()}
    with open(f"{OUT_DIR}/{label}_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    # full scan history (downsampled to ~10k rows)
    hist = run.history(samples=10000)
    hist.to_csv(f"{OUT_DIR}/{label}_history.csv", index=False)
    print("history shape:", hist.shape)
    print("history columns:", sorted(hist.columns.tolist()))
    print()

print("DONE")
