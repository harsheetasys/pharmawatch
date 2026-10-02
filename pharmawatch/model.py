"""Severity model: predicts P(report is serious) from complaint features.

Design choices worth defending in an interview:
* TIME-BASED split (train on the past, test on the future) - a random split
  would leak future drug-history information and overstate performance.
* Baseline first (logistic regression), then XGBoost; the winner is picked on
  VALIDATION PR-AUC, never on the test set.
* Operating threshold is chosen for RECALL, not accuracy: in drug safety a
  missed serious case (false negative) is far costlier than an extra review.
* Explanations: XGBoost's native SHAP values (pred_contribs) per prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .features import featurize, fit_vocab, load_training_frame


def time_split(df: pd.DataFrame, train=0.70, val=0.15):
    df = df.sort_values(["received_date", "report_id"]).reset_index(drop=True)
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def threshold_for_recall(y, p, target: float) -> float:
    """Highest threshold whose recall is still >= target (=> best precision at that recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def threshold_for_precision(y, p, target: float) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.99


def evaluate(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "threshold": round(float(thr), 4),
        "recall": round(float(recall_score(y, yhat)), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "false_negatives": int(fn), "false_positives": int(fp), "true_positives": int(tp), "true_negatives": int(tn),
        "n": int(len(y)), "prevalence": round(float(np.mean(y)), 4),
    }


def train(engine=None, verbose=True) -> dict:
    df, term_map = load_training_frame(engine)
    tr, va, te = time_split(df)
    terms = lambda part: [term_map.get(r, []) for r in part.report_id]
    vocab = fit_vocab(terms(tr), tr.serious.values)
    Xtr, Xva, Xte = (featurize(p, terms(p), vocab) for p in (tr, va, te))
    ytr, yva, yte = tr.serious.values, va.serious.values, te.serious.values

    candidates = {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=2000, C=0.5)),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                                 early_stopping_rounds=40, n_jobs=4, random_state=0),
    }
    results, fitted = {}, {}
    for name, m in candidates.items():
        if name == "xgboost":
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            m.fit(Xtr, ytr)
        pva = m.predict_proba(Xva)[:, 1]
        thr = threshold_for_recall(yva, pva, config.TARGET_RECALL)
        results[name] = {"validation": evaluate(yva, pva, thr)}
        fitted[name] = (m, thr)
        if verbose:
            print(f"[model] {name:20s} val PR-AUC={results[name]['validation']['pr_auc']:.3f} "
                  f"ROC-AUC={results[name]['validation']['roc_auc']:.3f}")

    best = max(results, key=lambda k: results[k]["validation"]["pr_auc"])
    model, thr = fitted[best]
    pva = model.predict_proba(Xva)[:, 1]
    thr_high = max(threshold_for_precision(yva, pva, 0.85), thr)
    for name, (m, t) in fitted.items():               # test reported for BOTH, selection already done
        results[name]["test"] = evaluate(yte, m.predict_proba(Xte)[:, 1], t)

    # Drug history for serving: smoothed serious rate over ALL known reports
    g = df.groupby("drug_name")["serious"].agg(["sum", "count"])
    drug_rate = ((g["sum"] + 1) / (g["count"] + 2)).to_dict()

    bundle = {
        "model_name": best, "model": model, "threshold": thr, "threshold_high": thr_high,
        "vocab": vocab, "columns": list(Xtr.columns), "drug_rate": drug_rate,
        "target_recall": config.TARGET_RECALL,
    }
    joblib.dump(bundle, config.MODEL_PATH)
    summary = {
        "selected_model": best, "target_recall": config.TARGET_RECALL,
        "threshold": round(thr, 4), "threshold_high": round(thr_high, 4),
        "split": {"train": len(tr), "validation": len(va), "test": len(te),
                  "train_end": str(tr.received_date.max().date()), "test_start": str(te.received_date.min().date())},
        "results": results,
        "top_features": top_features(bundle, Xva),
    }
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    if verbose:
        t = results[best]["test"]
        print(f"[model] selected={best} | TEST recall={t['recall']} precision={t['precision']} "
              f"PR-AUC={t['pr_auc']} FN={t['false_negatives']}")
    return summary


def top_features(bundle, X: pd.DataFrame, k=12) -> list[dict]:
    contrib = _contributions(bundle, X)
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(imp)[::-1][:k]
    return [{"feature": bundle["columns"][i], "mean_abs_contribution": round(float(imp[i]), 4)} for i in order]


def _contributions(bundle, X: pd.DataFrame) -> np.ndarray:
    m = bundle["model"]
    if bundle["model_name"] == "xgboost":
        import xgboost as xgb
        return m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # drop bias column
    imputer, scaler, lr = m.steps[0][1], m.steps[1][1], m.steps[2][1]
    return scaler.transform(imputer.transform(X)) * lr.coef_[0]


@dataclass
class SeverityResult:
    probability: float
    risk_band: str          # high / elevated / low
    flag_for_review: bool
    threshold: float
    top_factors: list[dict]
    model: str


_BUNDLE = None


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(config.MODEL_PATH)
    return _BUNDLE


def score(complaint: dict, bundle=None) -> SeverityResult:
    """Score one complaint dict: drugs (suspect first), reactions, age, sex, weight, reporter."""
    from . import real_model
    if bundle is None and real_model.handles(complaint):   # GLP-1 drug + trained real model -> use it
        return real_model.score(complaint)
    b = bundle or load_bundle()
    drugs = [d.strip() for d in complaint.get("drugs", []) if d and d.strip()]
    reactions = [r.lower() for r in complaint.get("reactions", [])]
    suspect = drugs[0] if drugs else None
    rate = next((v for k, v in b["drug_rate"].items() if suspect and k.lower() == suspect.lower()), 0.5)
    row = pd.DataFrame([{
        "patient_age": complaint.get("age"), "patient_sex": complaint.get("sex") or "U",
        "patient_weight": complaint.get("weight"),
        "reporter_qualification": complaint.get("reporter") or "consumer",
        "n_drugs": max(1, len(drugs)), "n_concomitant": max(0, len(drugs) - 1),
        "n_reactions": max(1, len(reactions)), "drug_prior_serious_rate": rate,
    }])
    X = featurize(row, [reactions], b["vocab"])[b["columns"]]
    p = float(b["model"].predict_proba(X)[:, 1][0])
    contrib = _contributions(b, X)[0]
    order = np.argsort(np.abs(contrib))[::-1][:5]
    factors = [{"feature": b["columns"][i], "value": _jsonable(X.iloc[0, i]),
                "contribution": round(float(contrib[i]), 3)} for i in order if abs(contrib[i]) > 1e-6]
    band = "high" if p >= b["threshold_high"] else "elevated" if p >= b["threshold"] else "low"
    return SeverityResult(round(p, 4), band, p >= b["threshold"], round(b["threshold"], 4), factors, b["model_name"])


def _jsonable(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v
