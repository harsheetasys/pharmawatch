"""Lesson 5a + 5b: train the FINAL real-data severity model and save it for PharmaWatch.

- Clinical-only features (no country / report type / reporter / year): the honest model.
- XGBoost, early-stopped on a validation split.
- 5b: thresholds chosen on VALIDATION data (weighted):
      threshold        -> recall >= 90%  ("flag for review": catch 9 in 10 serious cases)
      threshold_high   -> precision >= 85% ("high risk": when we say serious, mostly right)
- 5b: calibration table on the TEST set: does "70% risk" really mean ~70% are serious?
Saves artifacts/glp1_severity.joblib + artifacts/glp1_metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from benchmarks.glp1_ablation import FEATURE_SETS, drop
from benchmarks.glp1_download_v2 import OUT_PATH
from benchmarks.glp1_model import LEAKY_TERMS, TOP_TERMS, build_features, fit, metrics, models
from pharmawatch import config

BUNDLE_PATH = config.ARTIFACTS / "glp1_severity.joblib"
METRICS_PATH = config.ARTIFACTS / "glp1_metrics.json"
TARGET_RECALL, TARGET_PRECISION = 0.90, 0.85


def threshold_for_recall(y, p, w, target):
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]])                      # highest threshold that still catches `target` of serious cases


def threshold_for_precision(y, p, w, target):
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.99


def confusion_at(y, p, w, thr) -> dict:
    flag = p >= thr
    tp, fn = w[(y == 1) & flag].sum(), w[(y == 1) & ~flag].sum()
    fp = w[(y == 0) & flag].sum()
    return {"threshold": round(thr, 3), "recall": round(float(tp / (tp + fn)), 3),
            "precision": round(float(tp / (tp + fp)), 3), "share_flagged": round(float(w[flag].sum() / w.sum()), 3)}


def calibration_table(y, p, w, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p, "w": w})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    d["wy"], d["wp"] = d["y"] * d["w"], d["p"] * d["w"]
    t = d.groupby("bin", observed=True).agg(w=("w", "sum"), wy=("wy", "sum"), wp=("wp", "sum"))
    return pd.DataFrame({"predicted_risk": t.wp / t.w, "actual_serious_rate": t.wy / t.w}).round(3)


if __name__ == "__main__":
    df = pd.read_pickle(OUT_PATH).reset_index(drop=True)
    tr_all, te = train_test_split(df.index, test_size=0.25, random_state=42, stratify=df["serious"])
    tr, va = train_test_split(tr_all, test_size=0.2, random_state=0, stratify=df.loc[tr_all, "serious"])
    counts = df.loc[tr, "reactions"].explode().value_counts()
    top_terms = [t for t in counts.index if t not in LEAKY_TERMS][:TOP_TERMS]
    X = drop(build_features(df, top_terms), FEATURE_SETS["3) minus reporting context"])
    X = X.reindex(columns=sorted(X.columns))
    y, w = df["serious"].to_numpy(), df["sample_weight"].to_numpy()

    model = fit("xgboost", models()["xgboost"], X.loc[tr], y[tr], w[tr], X.loc[va], y[va], w[va])
    p_va, p_te = model.predict_proba(X.loc[va])[:, 1], model.predict_proba(X.loc[te])[:, 1]

    thr = threshold_for_recall(y[va], p_va, w[va], TARGET_RECALL)
    thr_high = max(threshold_for_precision(y[va], p_va, w[va], TARGET_PRECISION), thr)
    test_metrics = {k: round(v, 4) for k, v in metrics(y[te], p_te, w[te]).items()}
    at_flag, at_high = confusion_at(y[te], p_te, w[te], thr), confusion_at(y[te], p_te, w[te], thr_high)
    calib = calibration_table(y[te], p_te, w[te])

    bundle = {"model": model, "columns": list(X.columns), "top_terms": top_terms,
              "threshold": thr, "threshold_high": thr_high, "model_name": "xgboost-glp1-faers-clinical"}
    joblib.dump(bundle, BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps({"n_reports": len(df), "test": test_metrics,
                                        "flag_for_review": at_flag, "high_risk": at_high,
                                        "calibration": calib.reset_index(drop=True).to_dict("records")}, indent=2))

    print(f"Saved model -> {BUNDLE_PATH}")
    print(f"\nTEST  ROC-AUC {test_metrics['roc_auc']}  PR-AUC {test_metrics['pr_auc']}  Brier {test_metrics['brier']}")
    print(f"\n'Flag for review' (recall target {TARGET_RECALL:.0%}):  {at_flag}")
    print(f"'High risk'       (precision target {TARGET_PRECISION:.0%}): {at_high}")
    print("\nCalibration on test set (10 equal-size groups by predicted risk):")
    print(calib.to_string(index=False))