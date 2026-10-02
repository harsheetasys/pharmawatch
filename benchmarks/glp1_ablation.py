"""Lesson 4c: is the model learning MEDICINE or REPORTING RULES? (feature ablation)

FAERS quirk: reports from outside the US, and reports from clinical studies, mostly reach FDA
only when they are serious - so "country" and "report type" predict seriousness for regulatory
reasons, not clinical ones. We retrain with groups of features removed and watch the AUC.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarks.bootstrap_auc import paired_bootstrap
from benchmarks.glp1_download_v2 import OUT_PATH
from benchmarks.glp1_model import LEAKY_TERMS, TOP_TERMS, build_features, fit, metrics, models

FEATURE_SETS = {
    "1) all features": [],
    "2) minus country": ["country_us"],
    "3) minus reporting context": ["country_us", "report_type_", "reporter_", "year"],
}


def drop(X: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    return X[[c for c in X.columns if not any(c.startswith(p) for p in prefixes)]]


if __name__ == "__main__":
    df = pd.read_pickle(OUT_PATH).reset_index(drop=True)
    tr_all, te = train_test_split(df.index, test_size=0.25, random_state=42, stratify=df["serious"])
    tr, va = train_test_split(tr_all, test_size=0.2, random_state=0, stratify=df.loc[tr_all, "serious"])
    counts = df.loc[tr, "reactions"].explode().value_counts()
    top_terms = [t for t in counts.index if t not in LEAKY_TERMS][:TOP_TERMS]
    X_all = build_features(df, top_terms)
    X_all = X_all.reindex(columns=sorted(X_all.columns))
    y, w = df["serious"], df["sample_weight"]

    rows, preds = [], {}
    for set_name, prefixes in FEATURE_SETS.items():
        X = drop(X_all, prefixes)
        for name in ["logistic_regression", "xgboost"]:
            m = fit(name, models()[name], X.loc[tr], y[tr], w[tr], X.loc[va], y[va], w[va])
            p = m.predict_proba(X.loc[te])[:, 1]
            preds[(set_name, name)] = p
            rows.append({"features": set_name, "model": name, "n_features": X.shape[1],
                         **{k: round(v, 3) for k, v in metrics(y[te], p, w[te]).items()}})
            print(f"   {set_name:30s} {name:20s} test ROC-AUC {rows[-1]['roc_auc']:.3f}", flush=True)

    print("\n" + pd.DataFrame(rows).to_string(index=False))

    test = pd.DataFrame({"truth": y[te].values,
                         "full": preds[("1) all features", "xgboost")],
                         "clinical": preds[("3) minus reporting context", "xgboost")]})
    obs, low, high, share = paired_bootstrap(test, "full", "clinical", n_boot=500)
    print(f"\nXGBoost: all features minus no-reporting-context = {obs:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]")
    print("Poster baseline (random holdout): logistic regression 0.876 | random forest 0.916")