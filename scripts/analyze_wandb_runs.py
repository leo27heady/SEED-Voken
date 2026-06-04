"""Compare W&B runs for 64_S vs 128_S baseline."""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

RUNS = {
    "64_S (ch76xeh0)": "leo27heady/seed-voken-shapes3d/ch76xeh0",
    "128_S baseline (9v4vq5rv)": "leo27heady/seed-voken-shapes3d/9v4vq5rv",
}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "wandb_analysis")


def last_valid(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    api = wandb.Api()
    all_hist = {}

    for label, path in RUNS.items():
        run = api.run(path)
        hist = run.history(samples=10000, pandas=True)
        if "_step" in hist.columns:
            hist = hist.sort_values("_step")
        all_hist[label] = hist
        hist.to_csv(os.path.join(OUT_DIR, f"history_{run.id}.csv"), index=False)

    metrics = [
        "loss/mse_epoch",
        "loss/distortion_epoch",
        "loss/total_epoch",
        "loss/kl_total_epoch",
        "val/mse",
        "val_ema/mse",
        "val/distortion",
        "val_ema/distortion",
        "train/perplexity_layer_1_epoch",
        "train/perplexity_layer_2_epoch",
        "train/perplexity_layer_3_epoch",
        "train/active_codes_layer_1_epoch",
        "train/active_codes_layer_2_epoch",
        "train/active_codes_layer_3_epoch",
        "train/code_usage_frac_layer_1_epoch",
        "train/code_usage_frac_layer_2_epoch",
        "train/code_usage_frac_layer_3_epoch",
        "train/temperature_epoch",
        "lr-AdamW",
    ]

    rows = []
    for label, hist in all_hist.items():
        row = {
            "run": label,
            "steps": len(hist),
            "max_step": hist["_step"].max() if "_step" in hist.columns else np.nan,
        }
        if "epoch" in hist.columns:
            row["max_epoch"] = hist["epoch"].max()
        for m in metrics:
            if m in hist.columns:
                row[m] = last_valid(hist[m])
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(OUT_DIR, "run_summary_last.csv"), index=False)

    plot_specs = [
        ("loss/mse_epoch", "Train MSE (epoch)"),
        ("val_ema/mse", "Val EMA MSE"),
        ("loss/distortion_epoch", "Train ARELBO distortion"),
        ("loss/kl_total_epoch", "Total KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flat, plot_specs):
        for label, hist in all_hist.items():
            if col not in hist.columns:
                continue
            x = hist["_step"] if "_step" in hist.columns else range(len(hist))
            y = hist[col]
            mask = y.notna()
            ax.plot(x[mask], y[mask], label=label, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_core_losses.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, hist in all_hist.items():
        for i in (1, 2, 3):
            col = f"train/perplexity_layer_{i}_epoch"
            if col in hist.columns and hist[col].notna().any():
                ax.plot(hist["_step"], hist[col], label=f"{label} L{i}", alpha=0.85)
    ax.set_title("Perplexity per layer (epoch)")
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_perplexity.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, hist in all_hist.items():
        for col, style in (("val_ema/mse", "-"), ("val/mse", "--")):
            if col in hist.columns and "epoch" in hist.columns:
                g = hist.dropna(subset=[col, "epoch"]).groupby("epoch")[col].last()
                if len(g):
                    ax.plot(g.index, g.values, style, label=f"{label} {col}")
    ax.set_title("Validation MSE by epoch")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_val_mse_by_epoch.png"), dpi=150)
    plt.close()

    # KL layers
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, hist in all_hist.items():
        for i in (1, 2, 3):
            col = f"loss/kl_layer_{i}_epoch"
            if col in hist.columns and hist[col].notna().any():
                ax.plot(hist["_step"], hist[col], label=f"{label} kl L{i}")
    ax.set_title("Weighted KL per layer")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_kl_layers.png"), dpi=150)
    plt.close()

    print(df_summary.to_string())
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
