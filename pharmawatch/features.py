"""Feature engineering.

Two halves:
1. SQL (CTEs + a window function) builds one row per report from the
   normalized tables, including a LEAKAGE-SAFE historical feature:
   "what share of this drug's *earlier* reports were serious?"
   (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING = only the past).
2. `featurize()` turns report rows + reaction lists into the model matrix.
   The SAME function is used at training time and at serving time (API/agent),
   which prevents train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db

REPORT_FEATURES_SQL = """
WITH primary_drug AS (
    SELECT report_id, MIN(drug_id) AS drug_id
    FROM report_drugs
    WHERE role = 'suspect'
    GROUP BY report_id
),
drug_counts AS (
    SELECT report_id,
           COUNT(*)                                           AS n_drugs,
           SUM(CASE WHEN role = 'suspect' THEN 1 ELSE 0 END)  AS n_suspect,
           SUM(CASE WHEN role <> 'suspect' THEN 1 ELSE 0 END) AS n_concomitant
    FROM report_drugs
    GROUP BY report_id
),
reaction_counts AS (
    SELECT report_id, COUNT(*) AS n_reactions
    FROM report_reactions
    GROUP BY report_id
),
base AS (
    SELECT r.report_id, r.received_date, r.serious, r.patient_age, r.patient_sex,
           r.patient_weight, r.reporter_qualification, r.country,
           p.drug_id, d.name AS drug_name,
           dc.n_drugs, dc.n_suspect, dc.n_concomitant, rc.n_reactions
    FROM reports r
    JOIN primary_drug    p  ON p.report_id  = r.report_id
    JOIN drugs           d  ON d.drug_id    = p.drug_id
    JOIN drug_counts     dc ON dc.report_id = r.report_id
    JOIN reaction_counts rc ON rc.report_id = r.report_id
)
SELECT b.*,
       COUNT(*)      OVER prior AS drug_prior_reports,
       SUM(serious)  OVER prior AS drug_prior_serious
FROM base b
WINDOW prior AS (PARTITION BY drug_id ORDER BY received_date, report_id
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY received_date, report_id
"""

REACTIONS_SQL = "SELECT report_id, reaction_term FROM report_reactions"

REPORTERS = ["consumer", "physician", "pharmacist", "other_hp", "lawyer"]
TOP_TERMS = 60


def load_training_frame(engine=None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = db.query(REPORT_FEATURES_SQL, engine=engine)
    df["received_date"] = pd.to_datetime(df["received_date"])
    df["drug_prior_serious"] = df["drug_prior_serious"].fillna(0)
    # Laplace-smoothed prior serious rate (a new drug with no history -> 0.5)
    df["drug_prior_serious_rate"] = (df["drug_prior_serious"] + 1) / (df["drug_prior_reports"] + 2)
    rx = db.query(REACTIONS_SQL, engine=engine)
    terms = rx.groupby("report_id")["reaction_term"].apply(list).to_dict()
    return df, terms


def fit_vocab(train_terms: list[list[str]], y: np.ndarray) -> dict:
    """Learn reaction vocabulary + smoothed per-term serious rate on TRAIN ONLY."""
    exploded = pd.DataFrame({"t": train_terms, "y": y}).explode("t")
    stats = exploded.groupby("t")["y"].agg(["sum", "count"])
    prior = float(y.mean())
    k = 20  # smoothing strength: rare terms shrink toward the global rate
    term_rate = ((stats["sum"] + k * prior) / (stats["count"] + k)).to_dict()
    top = stats.sort_values("count", ascending=False).head(TOP_TERMS).index.tolist()
    return {"term_rate": term_rate, "top_terms": top, "global_rate": prior}


def featurize(rows: pd.DataFrame, terms: list[list[str]], vocab: dict) -> pd.DataFrame:
    """rows needs: patient_age, patient_sex, patient_weight, reporter_qualification,
    n_drugs, n_concomitant, n_reactions, drug_prior_serious_rate."""
    X = pd.DataFrame(index=rows.index)
    X["age"] = pd.to_numeric(rows["patient_age"], errors="coerce")
    X["age_missing"] = X["age"].isna().astype(int)
    X["weight"] = pd.to_numeric(rows["patient_weight"], errors="coerce")
    X["sex_female"] = (rows["patient_sex"] == "F").astype(int)
    X["sex_unknown"] = (~rows["patient_sex"].isin(["F", "M"])).astype(int)
    for r in REPORTERS:
        X[f"reporter_{r}"] = (rows["reporter_qualification"] == r).astype(int)
    X["n_drugs"] = rows["n_drugs"].astype(float)
    X["n_concomitant"] = rows["n_concomitant"].astype(float)
    X["n_reactions"] = rows["n_reactions"].astype(float)
    X["drug_prior_serious_rate"] = rows["drug_prior_serious_rate"].astype(float)

    tr, g = vocab["term_rate"], vocab["global_rate"]
    rates = [[tr.get(t, g) for t in ts] or [g] for ts in terms]
    X["term_risk_max"] = [max(r) for r in rates]
    X["term_risk_mean"] = [float(np.mean(r)) for r in rates]
    X["n_unseen_terms"] = [sum(t not in tr for t in ts) for ts in terms]
    top = vocab["top_terms"]
    hot = np.zeros((len(rows), len(top)), dtype=np.int8)
    idx = {t: i for i, t in enumerate(top)}
    for r, ts in enumerate(terms):
        for t in ts:
            if t in idx:
                hot[r, idx[t]] = 1
    X = pd.concat([X, pd.DataFrame(hot, index=rows.index, columns=[f"rx::{t}" for t in top])], axis=1)
    return X
