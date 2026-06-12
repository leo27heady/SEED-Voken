"""Summarize trends for encoder and predictor W&B histories."""
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

D = "docs/wandb_analysis"


def trend(df, col, n=8):
    s = df[["_step", col]].dropna()
    if s.empty:
        return None
    qs = s.iloc[[int(i * (len(s) - 1) / (n - 1)) for i in range(n)]]
    return qs.set_index("_step")[col].round(5).to_dict()


def show(df, cols, title):
    print(f"\n--- {title} ---")
    for c in cols:
        t = trend(df, c)
        if t is None:
            print(f"{c}: <no data>")
            continue
        vals = " | ".join(f"{k}:{v:g}" for k, v in t.items())
        print(f"{c}\n   {vals}")


enc = pd.read_csv(f"{D}/encoder_idomjiua_history.csv")
pred = pd.read_csv(f"{D}/predictor_wgm3d46g_history.csv")

print("=== ENCODER (idomjiua) ===")
print("max epoch:", enc["epoch"].max(), "max step:", enc["trainer/global_step"].max())
show(enc, [
    "loss/total_epoch", "loss/distortion_epoch", "loss/mse_per_pixel_epoch",
    "loss/kl_total_epoch",
    "loss/kl_layer_1_raw_epoch", "loss/kl_layer_2_raw_epoch", "loss/kl_layer_3_raw_epoch",
    "train/temperature_epoch",
    "train/perplexity_layer_1_epoch", "train/perplexity_layer_2_epoch", "train/perplexity_layer_3_epoch",
    "train/perplexity_frac_layer_1_epoch", "train/perplexity_frac_layer_2_epoch", "train/perplexity_frac_layer_3_epoch",
    "train/code_usage_frac_layer_1_epoch", "train/code_usage_frac_layer_2_epoch", "train/code_usage_frac_layer_3_epoch",
    "train/active_codes_layer_1_epoch", "train/active_codes_layer_2_epoch", "train/active_codes_layer_3_epoch",
    "train/posterior_var_layer_1_epoch", "train/posterior_var_layer_2_epoch", "train/posterior_var_layer_3_epoch",
], "encoder train trends")
show(enc, [
    "val_ema/mse_per_pixel", "val_ema/total",
    "val_ema/perplexity_layer_1", "val_ema/perplexity_layer_2", "val_ema/perplexity_layer_3",
    "val_ema/code_usage_frac_layer_1", "val_ema/code_usage_frac_layer_2", "val_ema/code_usage_frac_layer_3",
    "val_ema/active_codes_layer_1", "val_ema/active_codes_layer_2", "val_ema/active_codes_layer_3",
], "encoder val trends")

print("\n\n=== PREDICTOR (wgm3d46g) ===")
print("max epoch:", pred["epoch"].max(), "max step:", pred["trainer/global_step"].max())
show(pred, [
    "train/loss_total_epoch", "train/loss_ce_epoch", "train/loss_pred_mse_epoch",
    "train/ce_over_baseline_epoch",
    "train/ce_stage_0_epoch", "train/ce_stage_1_epoch", "train/ce_stage_2_epoch",
    "train/ce_s0_k0_epoch", "train/ce_s1_k0_epoch", "train/ce_s1_k1_epoch",
    "train/ce_s2_k0_epoch", "train/ce_s2_k1_epoch", "train/ce_s2_k2_epoch", "train/ce_s2_k3_epoch",
    "train/token_acc_stage_0_epoch", "train/token_acc_stage_1_epoch", "train/token_acc_stage_2_epoch",
], "predictor train trends")
show(pred, [
    "val/loss_total_ar", "val/loss_ce_ar", "val/loss_ce_parallel", "val/ce_gap_parallel_minus_ar",
    "val/ce_ar_stage_0", "val/ce_ar_stage_1", "val/ce_ar_stage_2",
    "val/ce_parallel_stage_0", "val/ce_parallel_stage_1", "val/ce_parallel_stage_2",
    "val/ce_s2_k0", "val/ce_s2_k1", "val/ce_s2_k2", "val/ce_s2_k3",
    "val_parallel/ce_s2_k0", "val_parallel/ce_s2_k3",
    "val/token_acc_ar_stage_0", "val/token_acc_ar_stage_1", "val/token_acc_ar_stage_2",
    "val/token_acc_parallel_stage_0", "val/token_acc_parallel_stage_1", "val/token_acc_parallel_stage_2",
    "val/pred_mse_ar_per_pixel", "val/pred_mse_parallel_per_pixel",
], "predictor val trends")

# magnitudes for loss balance
print("\n--- magnitude check (encoder, last valid rows) ---")
for c in ["loss/distortion_step", "loss/kl_total_step", "loss/kl_layer_1_raw_step"]:
    s = enc[c].dropna()
    if len(s):
        print(f"{c}: first={s.iloc[0]:.4g} last={s.iloc[-1]:.4g} min={s.min():.4g} max={s.max():.4g}")
print("\n--- predictor loss magnitudes ---")
for c in ["train/loss_ce_step", "train/loss_pred_mse_step"]:
    s = pred[c].dropna()
    if len(s):
        print(f"{c}: first={s.iloc[0]:.4g} last={s.iloc[-1]:.4g} min={s.min():.4g} max={s.max():.4g}")
