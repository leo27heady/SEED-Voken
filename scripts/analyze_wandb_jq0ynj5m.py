"""Deep analysis of W&B run jq0ynj5m vs prior 64_S runs."""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

RUN_NEW = "leo27heady/seed-voken-shapes3d/jq0ynj5m"
RUN_OLD = "leo27heady/seed-voken-shapes3d/ch76xeh0"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "wandb_analysis")


def load_run(path):
    run = wandb.Api().run(path)
    hist = run.history(samples=20000, pandas=True)
    if "_step" in hist.columns:
        hist = hist.sort_values("_step")
    return run, hist


def epoch_series(hist, col):
    if col not in hist.columns or "epoch" not in hist.columns:
        return pd.Series(dtype=float)
    return hist.dropna(subset=[col, "epoch"]).groupby("epoch")[col].last()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_new, h_new = load_run(RUN_NEW)
    run_old, h_old = load_run(RUN_OLD)

    # Save CSV
    h_new.to_csv(os.path.join(OUT_DIR, "history_jq0ynj5m.csv"), index=False)

    print("=" * 60)
    print("RUN jq0ynj5m:", run_new.name, "state:", run_new.state)
    print("steps:", h_new["_step"].max(), "epochs:", h_new["epoch"].max())
    print("=" * 60)

    # Core metrics at key epochs
    core = [
        "loss/mse_epoch",
        "loss/distortion_epoch",
        "loss/total_epoch",
        "loss/kl_total_epoch",
        "val/mse",
        "val_ema/mse",
        "val/distortion",
        "val_ema/distortion",
        "train/temperature_epoch",
        "lr-AdamW",
    ]
    kl_cols = [c for c in h_new.columns if c.startswith("loss/kl_layer_") and c.endswith("_epoch")]
    ppl_cols = [c for c in h_new.columns if "perplexity_layer" in c and c.endswith("_epoch")]
    ac_cols = [c for c in h_new.columns if "active_codes_layer" in c and c.endswith("_epoch")]
    uf_cols = [c for c in h_new.columns if "code_usage_frac_layer" in c and c.endswith("_epoch")]
    pv_cols = [c for c in h_new.columns if "posterior_var_layer" in c and c.endswith("_epoch")]

    codebook_sizes = [1536, 768, 384]

    rows = []
    for ep in [4, 20, 40, 60, 80]:
        row = {"epoch": ep}
        for col in core + kl_cols + ppl_cols + ac_cols + uf_cols:
            s = epoch_series(h_new, col)
            row[col] = s.get(ep, np.nan)
        rows.append(row)
    df_ep = pd.DataFrame(rows)
    df_ep.to_csv(os.path.join(OUT_DIR, "jq0ynj5m_by_epoch.csv"), index=False)
    print("\n--- Metrics at epochs 4, 20, 40, 60, 80 ---")
    print(df_ep.to_string())

    # Best val
    v = epoch_series(h_new, "val_ema/mse")
    if len(v):
        print(f"\nBest val_ema/mse: {v.min():.2f} at epoch {v.idxmin():.0f}")
        print(f"Last val_ema/mse: {v.iloc[-1]:.2f} at epoch {v.index[-1]:.0f}")

    # Perplexity vs codebook ratio
    print("\n--- Perplexity / codebook (epoch 80, expect << 1, not pinned at 1) ---")
    s80 = df_ep[df_ep.epoch == 80]
    if len(s80):
        for i, K in enumerate(codebook_sizes, 1):
            col = f"train/perplexity_layer_{i}_epoch"
            if col in s80.columns:
                p = s80[col].values[0]
                if not np.isnan(p):
                    print(f"  L{i}: perplexity={p:.1f} / K={K} => ratio={p/K:.3f}")

    # KL sign and magnitude
    print("\n--- KL layers at epoch 80 (negative = typical for zero prior) ---")
    for col in sorted(kl_cols):
        if col in s80.columns:
            print(f"  {col}: {s80[col].values[0]}")

    # Inconsistency checks
    print("\n--- Consistency checks ---")
    issues = []

    # 1. val vs train gap
    if len(v) and "loss/mse_epoch" in h_new.columns:
        tm = epoch_series(h_new, "loss/mse_epoch")
        common = v.index.intersection(tm.index)
        if len(common):
            last_ep = min(80, max(common))
            if last_ep in v.index and last_ep in tm.index:
                if v.loc[last_ep] > 3 * tm.loc[last_ep]:
                    issues.append(
                        f"Large val/train MSE gap at ep {last_ep}: val_ema={v.loc[last_ep]:.0f} vs train={tm.loc[last_ep]:.0f}"
                    )

    # 2. perplexity collapse
    for i, K in enumerate(codebook_sizes, 1):
        col = f"train/perplexity_layer_{i}_epoch"
        s = epoch_series(h_new, col)
        if len(s) and s.iloc[-1] < 0.05 * K:
            issues.append(f"L{i} perplexity very low ({s.iloc[-1]:.1f} vs K={K}) — possible collapse")
        if len(s) and s.iloc[-1] > 0.95 * K:
            issues.append(f"L{i} perplexity near uniform ({s.iloc[-1]:.1f} / {K}) — under-using codes")

    # 3. active codes vs size
    for i, K in enumerate(codebook_sizes, 1):
        ac = f"train/active_codes_layer_{i}_epoch"
        uf = f"train/code_usage_frac_layer_{i}_epoch"
        s_ac = epoch_series(h_new, ac)
        s_uf = epoch_series(h_new, uf)
        if len(s_ac) and len(s_uf):
            if abs(s_ac.iloc[-1] / K - s_uf.iloc[-1]) > 0.02:
                issues.append(
                    f"L{i} active_codes/K ({s_ac.iloc[-1]/K:.3f}) != usage_frac ({s_uf.iloc[-1]:.3f})"
                )

    # 4. kl_total vs sum of layers
    if kl_cols:
        s_tot = epoch_series(h_new, "loss/kl_total_epoch")
        s_sum = sum(epoch_series(h_new, c) for c in kl_cols if "_raw" not in c)
        if len(s_tot) and len(s_sum):
            ep = s_tot.index[-1]
            if ep in s_sum.index:
                diff = abs(s_tot.loc[ep] - s_sum.loc[ep])
                if diff > 0.5:
                    issues.append(
                        f"kl_total ({s_tot.loc[ep]:.3f}) != sum kl layers ({s_sum.loc[ep]:.3f})"
                    )

    # 5. distortion vs mse relationship (arelbo)
    dist = epoch_series(h_new, "loss/distortion_epoch")
    mse = epoch_series(h_new, "loss/mse_epoch")
    if len(dist) and len(mse):
        ep = 80 if 80 in dist.index else dist.index[-1]
        if ep in mse.index:
            # distortion should be >> mse for arelbo (log term)
            if dist.loc[ep] < mse.loc[ep]:
                issues.append(f"distortion < mse at ep {ep} — unusual for ARELBO")

    # 6. temperature
    temp = epoch_series(h_new, "train/temperature_epoch")
    if len(temp) and (temp.iloc[-1] < 0.31 or temp.iloc[-1] > 1.0):
        issues.append(f"temperature out of expected range: {temp.iloc[-1]:.3f}")

    # 7. compare old run at ~80 epochs
    v_old = epoch_series(h_old, "val_ema/mse")
    if len(v) and len(v_old):
        if 80 in v.index and len(v_old):
            old_near = v_old[v_old.index <= 85]
            if len(old_near):
                print(f"\nCompare val_ema/mse @80: new={v.loc[80]:.1f} old~80={old_near.iloc[-1]:.1f} (ch76xeh0)")

    if issues:
        print("ISSUES:")
        for x in issues:
            print("  [!]", x)
    else:
        print("No major inconsistencies flagged.")

    # Plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    specs = [
        ("val_ema/mse", "Val EMA MSE"),
        ("loss/mse_epoch", "Train MSE"),
        ("loss/distortion_epoch", "Train distortion"),
        ("loss/kl_total_epoch", "KL total"),
        ("train/perplexity_layer_1_epoch", "PPL L1"),
        ("train/perplexity_layer_3_epoch", "PPL L3"),
    ]
    for ax, (col, title) in zip(axes.flat, specs):
        for label, hist, style in [
            ("jq0ynj5m", h_new, "-"),
            ("ch76xeh0", h_old, "--"),
        ]:
            s = epoch_series(hist, col)
            if len(s):
                ax.plot(s.index, s.values, style, label=label, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "jq0ynj5m_analysis.png"), dpi=150)
    plt.close()

    # Layer panel
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, ax in enumerate(axes.flat):
        if i >= 3:
            ax.axis("off")
            continue
        for col, lbl in [
            (f"train/perplexity_layer_{i+1}_epoch", "perplexity"),
            (f"train/active_codes_layer_{i+1}_epoch", "active_codes"),
            (f"loss/kl_layer_{i+1}_epoch", "kl"),
        ]:
            s = epoch_series(h_new, col)
            if len(s):
                ax2 = ax.twinx() if lbl == "active_codes" else None
                ax.plot(s.index, s.values, label=lbl)
        ax.set_title(f"Layer {i+1} (K={codebook_sizes[i]})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "jq0ynj5m_layers.png"), dpi=150)
    plt.close()
    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
