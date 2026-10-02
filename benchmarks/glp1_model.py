"""Lesson 4b: predict whether a GLP-1 adverse-event report is SERIOUS.

Baseline to compare (ISPOR 2026 poster, ~480k FAERS GLP-1 cases, 25% random holdout):
    logistic regression ROC-AUC 0.876 | random forest ROC-AUC 0.916

We train logistic regression, random forest and XGBoost on the SAME features and split,
choose on a validation set, and report the held-out test set. Everything is weighted by
sample_weight so the metrics reflect the real report mix, not our sampling.
Two test designs:
    A) random 25% holdout  -> like-for-like with the poster
    B) time split (train <= 2023, test 2024-25) -> stricter "does it work on the future?"
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from benchmarks.glp1_download_v2 import OUT_PATH

# Reaction terms that ARE the outcome -> never allowed as inputs (leakage).
LEAKY_TERMS = {"death", "hospitalisation", "disability", "life threatening"}

# Adverse-event categories (like the poster): keyword -> category
AE_GROUPS = {
    "ae_gastrointestinal": ["nausea", "vomit", "diarrh", "constipat", "abdominal", "dyspeps", "gastroparesis", "eructation"],
    "ae_pancreatic": ["pancrea"],
    "ae_gallbladder": ["cholecyst", "cholelith", "gallbladder", "biliary"],
    "ae_renal": ["kidney", "renal"],
    "ae_psychiatric": ["suicid", "depress", "anxiety", "mood"],
    "ae_hypoglycaemia": ["hypoglyc"],
    "ae_injection_site": ["injection site"],
    "ae_dosing_or_admin_issue": ["dose", "medication error", "off label", "device", "incorrect", "wrong", "accidental", "product"],
    "ae_thyroid": ["thyroid"],
    "ae_eye": ["retinopathy", "vision", "visual", "eye"],
    "ae_drug_ineffective": ["ineffective", "no adverse event"],
}
TOP_TERMS = 100


def indication_bucket(text: str) -> str:
    t = text or ""
    if any(k in t for k in ["diabet", "glyc", "glucose", "insulin"]):
        return "diabetes"
    if any(k in t for k in ["weight", "obes", "overweight"]):
        return "weight_loss"
    return "unknown" if t in ("", "unknown", "product used for unknown indication") else "other"


def build_features(df: pd.DataFrame, top_terms: list[str]) -> pd.DataFrame:
    reactions = df["reactions"].apply(lambda rs: [r for r in rs if r not in LEAKY_TERMS])
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["age_missing"] = df["age"].isna().astype(int)
    X["weight_kg"] = df["weight"]
    X["country_us"] = df["country_us"]
    X["glp1_is_suspect"] = df["glp1_is_suspect"]
    X["n_drugs"] = df["n_drugs"]
    X["n_reactions"] = reactions.apply(len)
    X["year"] = df["received"].dt.year
    for col in ["sex", "reporter", "report_type", "molecule"]:
        X = X.join(pd.get_dummies(df[col], prefix=col, dtype=int))
    X = X.join(pd.get_dummies(df["indication"].apply(indication_bucket), prefix="indication", dtype=int))
    joined = reactions.apply(lambda rs: " | ".join(rs))
    for name, keys in AE_GROUPS.items():
        X[name] = joined.apply(lambda s: int(any(k in s for k in keys)))
    term_sets = reactions.apply(set)
    pt = pd.DataFrame({f"pt::{t}": term_sets.apply(lambda s, t=t: int(t in s)) for t in top_terms}, index=df.index)
    return pd.concat([X, pt], axis=1)          # add all 100 columns in one go (fast, no warning)


def models():
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=3000, C=0.5)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                              n_jobs=-1, random_state=0)),
        "xgboost": XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=6, subsample=0.9,
                                 colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                                 early_stopping_rounds=50, n_jobs=-1, random_state=0),
    }


def fit(name, model, Xtr, ytr, wtr, Xva, yva, wva):
    if name == "xgboost":
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
    else:
        step = model.steps[-1][0]
        model.fit(Xtr, ytr, **{f"{step}__sample_weight": wtr})
    return model


def metrics(y, p, w) -> dict:
    return {"roc_auc": roc_auc_score(y, p, sample_weight=w),
            "pr_auc": average_precision_score(y, p, sample_weight=w),
            "brier": brier_score_loss(y, p, sample_weight=w)}


def run_design(df: pd.DataFrame, train_idx, test_idx, label: str) -> pd.DataFrame:
    train, test = df.loc[train_idx], df.loc[test_idx]
    tr, va = train_test_split(train.index, test_size=0.2, random_state=0, stratify=train["serious"])
    # Vocabulary of top reaction terms learned on TRAINING data only
    counts = train.loc[tr, "reactions"].explode().value_counts()
    top_terms = [t for t in counts.index if t not in LEAKY_TERMS][:TOP_TERMS]
    X_all = build_features(df, top_terms)
    X_all = X_all.reindex(columns=sorted(X_all.columns))
    y, w = df["serious"], df["sample_weight"]

    rows, fitted = [], {}
    for name, model in models().items():
        m = fit(name, model, X_all.loc[tr], y[tr], w[tr], X_all.loc[va], y[va], w[va])
        fitted[name] = m
        val = metrics(y[va], m.predict_proba(X_all.loc[va])[:, 1], w[va])
        tst = metrics(y[test_idx], m.predict_proba(X_all.loc[test_idx])[:, 1], w[test_idx])
        rows.append({"design": label, "model": name, "val_roc_auc": val["roc_auc"],
                     "test_roc_auc": tst["roc_auc"], "test_pr_auc": tst["pr_auc"], "test_brier": tst["brier"]})
        print(f"   {label:12s} {name:20s} val ROC-AUC {val['roc_auc']:.3f} | test ROC-AUC {tst['roc_auc']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    best = out.loc[out.val_roc_auc.idxmax(), "model"]
    out["selected"] = out.model == best

    if label.startswith("A"):          # explain the best model on the main design
        xgb = fitted["xgboost"]
        imp = pd.Series(xgb.feature_importances_, index=X_all.columns).sort_values(ascending=False).head(15)
        print("\n   Top 15 features (XGBoost importance):")
        for f, v in imp.items():
            print(f"      {f:40s} {v:.3f}")
    return out


if __name__ == "__main__":
    df = pd.read_pickle(OUT_PATH).reset_index(drop=True)
    print(f"{len(df):,} reports | serious (weighted) {np.average(df.serious, weights=df.sample_weight):.1%}\n")

    # A) random 25% holdout, stratified by outcome (poster's protocol)
    trA, teA = train_test_split(df.index, test_size=0.25, random_state=42, stratify=df["serious"])
    resA = run_design(df, trA, teA, "A) random")

    # B) time split: learn from the past, test on 2024-2025
    year = df["received"].dt.year
    resB = run_design(df, df.index[year <= 2023], df.index[year >= 2024], "B) time")

    res = pd.concat([resA, resB]).round(3)
    print("\n" + res.to_string(index=False))
    print("\nPoster baseline (random holdout): logistic regression 0.876 | random forest 0.916")