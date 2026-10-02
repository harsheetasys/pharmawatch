"""Lesson 3e: is IC025's AUC gain over PRR05 real, or luck? (paired bootstrap)"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmarks.omop_prr import DATA_DIR


def paired_bootstrap(df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = df.iloc[rng.integers(0, len(df), len(df))]      # resample pairs WITH replacement
        if s.truth.nunique() < 2:                           # AUC needs both positives and negatives
            continue
        diffs.append(roc_auc_score(s.truth, s[a]) - roc_auc_score(s.truth, s[b]))
    diffs = np.array(diffs)
    observed = roc_auc_score(df.truth, df[a]) - roc_auc_score(df.truth, df[b])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return observed, low, high, (diffs > 0).mean()


if __name__ == "__main__":
    res = pd.read_csv(DATA_DIR / "omop_scores_compared.csv").dropna(subset=["prr05"])
    for name, df in [("A) same 346 pairs", res[res.original_match]), ("B) all 391 pairs", res)]:
        obs, low, high, share = paired_bootstrap(df, "ic025", "prr05")
        print(f"{name}: IC025 - PRR05 = {obs:+.3f}   95% CI [{low:+.3f}, {high:+.3f}]   IC025 wins in {share:.0%} of resamples")