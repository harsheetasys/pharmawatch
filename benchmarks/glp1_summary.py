"""Lesson 4a check: does our random sample reproduce the TRUE drug mix?"""
import pandas as pd

from benchmarks.glp1_download_v2 import OUT_PATH

TRUE_SHARE = {   # from glp1_true_mix.py (FDA counts, 2015-2025)
    "tirzepatide": 0.387, "dulaglutide": 0.273, "semaglutide": 0.195,
    "liraglutide": 0.111, "exenatide": 0.023, "lixisenatide": 0.011,
}

df = pd.read_pickle(OUT_PATH)
df["weighted_serious"] = df["serious"] * df["sample_weight"]

mix = df.groupby("molecule").agg(
    sampled=("sample_weight", "size"),
    est_real_reports=("sample_weight", "sum"),
    weighted_serious=("weighted_serious", "sum"),
)
mix["est_real_share"] = mix["est_real_reports"] / mix["est_real_reports"].sum()
mix["true_share"] = mix.index.map(TRUE_SHARE)
mix["serious_share"] = mix["weighted_serious"] / mix["est_real_reports"]

overall = df["weighted_serious"].sum() / df["sample_weight"].sum()
print(f"{len(df):,} reports | weighted serious share: {overall:.1%}\n")
print(mix[["sampled", "est_real_share", "true_share", "serious_share"]]
      .sort_values("true_share", ascending=False).round(3).to_string())