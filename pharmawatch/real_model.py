"""Real-data severity model for GLP-1 drugs (trained by benchmarks/glp1_train_final.py).

PharmaWatch uses this model whenever the suspect drug is a GLP-1 receptor agonist and the
trained bundle exists; otherwise it falls back to the demo model. Same output type
(SeverityResult), so the agent, API and dashboard do not need to change.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config

BUNDLE_PATH = config.ARTIFACTS / "glp1_severity.joblib"

# Brand names people actually write -> molecule
GLP1_NAMES = {
    "semaglutide": "semaglutide", "ozempic": "semaglutide", "wegovy": "semaglutide", "rybelsus": "semaglutide",
    "tirzepatide": "tirzepatide", "mounjaro": "tirzepatide", "zepbound": "tirzepatide",
    "dulaglutide": "dulaglutide", "trulicity": "dulaglutide",
    "liraglutide": "liraglutide", "victoza": "liraglutide", "saxenda": "liraglutide",
    "exenatide": "exenatide", "byetta": "exenatide", "bydureon": "exenatide",
    "lixisenatide": "lixisenatide", "adlyxin": "lixisenatide",
}

_BUNDLE = None


def molecule_of(drug: str | None) -> str | None:
    return GLP1_NAMES.get((drug or "").strip().lower())


def available() -> bool:
    return BUNDLE_PATH.exists()


def handles(complaint: dict) -> bool:
    drugs = complaint.get("drugs") or []
    return bool(drugs) and molecule_of(drugs[0]) is not None and available()


def _bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(BUNDLE_PATH)
    return _BUNDLE


def score(complaint: dict):
    from benchmarks.glp1_model import build_features   # same feature code as training -> no train/serve skew
    from .model import SeverityResult

    b = _bundle()
    drugs = complaint.get("drugs") or []
    row = pd.DataFrame([{
        "age": complaint.get("age"), "weight": complaint.get("weight"),
        "sex": complaint.get("sex") or "U", "reporter": "unknown", "report_type": "unknown",
        "country_us": 0, "received": pd.Timestamp.now(),
        "molecule": molecule_of(drugs[0]), "glp1_is_suspect": 1, "n_drugs": max(1, len(drugs)),
        "indication": (complaint.get("narrative") or "unknown").lower(),
        "reactions": [r.lower() for r in complaint.get("reactions") or []],
    }])
    X = build_features(row, b["top_terms"]).reindex(columns=b["columns"], fill_value=0).astype(float)
    p = float(b["model"].predict_proba(X)[:, 1][0])

    import xgboost as xgb
    contrib = b["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    order = np.argsort(np.abs(contrib))[::-1][:5]
    factors = [{"feature": b["columns"][i], "value": float(X.iloc[0, i]) if pd.notna(X.iloc[0, i]) else None,
                "contribution": round(float(contrib[i]), 3)} for i in order if abs(contrib[i]) > 1e-6]
    band = "high" if p >= b["threshold_high"] else "elevated" if p >= b["threshold"] else "low"
    return SeverityResult(round(p, 4), band, p >= b["threshold"], round(b["threshold"], 4), factors, b["model_name"])