"""Health analysis for the elbofix run (s1vvokkt)."""
import pandas as pd

pd.set_option("display.width", 250)

df = pd.read_csv("docs/wandb_analysis/elbofix_pt5j4bxu_history.csv")
print("rows:", len(df), "| max step:", df["trainer/global_step"].max(),
      "| max epoch:", df["epoch"].max(),
      "| runtime h:", round(df["_runtime"].max() / 3600, 2))

COLS = [
    "loss/total_step", "loss/distortion_step", "loss/mse_per_pixel_step",
    "loss/kl_total_step",
    "loss/kl_layer_1_raw_step", "loss/kl_layer_2_raw_step", "loss/kl_layer_3_raw_step",
    "train/perplexity_layer_1_step", "train/perplexity_layer_2_step", "train/perplexity_layer_3_step",
    "train/perplexity_frac_layer_1_step", "train/perplexity_frac_layer_2_step", "train/perplexity_frac_layer_3_step",
    "train/active_codes_layer_1_step", "train/active_codes_layer_2_step", "train/active_codes_layer_3_step",
    "train/code_usage_frac_layer_1_step", "train/code_usage_frac_layer_2_step", "train/code_usage_frac_layer_3_step",
    "train/posterior_var_layer_1_step", "train/posterior_var_layer_2_step", "train/posterior_var_layer_3_step",
    "train/temperature_step", "train/kl_beta_step", "lr-Adam",
]

n = 8
for c in COLS:
    s = df[["_step", c]].dropna()
    if s.empty:
        print(f"{c}: <no data>")
        continue
    qs = s.iloc[[int(i * (len(s) - 1) / (n - 1)) for i in range(n)]]
    vals = " | ".join(f"{int(r['_step'])}:{r[c]:.5g}" for _, r in qs.iterrows())
    print(f"{c}\n   {vals}")

print("\n--- val_ema (all rows) ---")
val_cols = [c for c in df.columns if c.startswith("val_ema/")]
v = df[["_step"] + val_cols].dropna(subset=val_cols, how="all")
for c in ["val_ema/mse_per_pixel", "val_ema/total", "val_ema/kl_total",
          "val_ema/perplexity_layer_1", "val_ema/perplexity_layer_2", "val_ema/perplexity_layer_3",
          "val_ema/active_codes_layer_1", "val_ema/active_codes_layer_2", "val_ema/active_codes_layer_3",
          "val_ema/posterior_var_layer_1", "val_ema/posterior_var_layer_2", "val_ema/posterior_var_layer_3"]:
    s = v[["_step", c]].dropna()
    if s.empty:
        print(f"{c}: <no data>")
        continue
    vals = " | ".join(f"{int(r['_step'])}:{r[c]:.5g}" for _, r in s.iterrows())
    print(f"{c}\n   {vals}")
