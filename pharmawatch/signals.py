"""Safety-signal detection (pharmacovigilance statistics).

1. Disproportionality - PRR (Proportional Reporting Ratio), the standard
   screening statistic used by regulators (e.g. EMA / MHRA) on spontaneous reports:

                         reaction R   other reactions
       drug D               a               b
       other drugs          c               d

       PRR = [a/(a+b)] / [c/(c+d)]
   Flag (Evans criteria): PRR >= 2, chi-square >= 4, and a >= 3.

2. Volume spikes - weekly report counts per drug vs. an 8-week baseline that
   sits behind a 3-week guard band; flag when z >= 3.5 ("emerging signal" chart).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db

PRR_SQL = """
WITH suspect AS (
    SELECT DISTINCT report_id, drug_id FROM report_drugs WHERE role = 'suspect'
    {date_filter_suspect}
),
pairs AS (
    SELECT s.drug_id, rr.reaction_term, COUNT(DISTINCT s.report_id) AS a
    FROM suspect s
    JOIN report_reactions rr ON rr.report_id = s.report_id
    GROUP BY s.drug_id, rr.reaction_term
),
drug_tot AS (SELECT drug_id, COUNT(DISTINCT report_id) AS n_drug FROM suspect GROUP BY drug_id),
term_tot AS (
    SELECT rr.reaction_term, COUNT(DISTINCT rr.report_id) AS n_term
    FROM report_reactions rr JOIN (SELECT DISTINCT report_id FROM suspect) s ON s.report_id = rr.report_id
    GROUP BY rr.reaction_term
),
total AS (SELECT COUNT(DISTINCT report_id) AS n FROM suspect)
SELECT d.name AS drug, p.reaction_term, p.a,
       dt.n_drug - p.a                          AS b,
       tt.n_term - p.a                          AS c,
       t.n - dt.n_drug - (tt.n_term - p.a)      AS d
FROM pairs p
JOIN drugs d     ON d.drug_id = p.drug_id
JOIN drug_tot dt ON dt.drug_id = p.drug_id
JOIN term_tot tt ON tt.reaction_term = p.reaction_term
CROSS JOIN total t
"""


def prr_table(since: str | None = None, engine=None) -> pd.DataFrame:
    flt = ""
    params = {}
    if since:
        flt = "AND report_id IN (SELECT report_id FROM reports WHERE received_date >= :since)"
        params["since"] = since
    df = db.query(PRR_SQL.format(date_filter_suspect=flt), params, engine=engine)
    if df.empty:
        return df
    a, b, c, d = (df[k].astype(float) + 0.5 for k in "abcd")          # Haldane continuity correction
    df["prr"] = (a / (a + b)) / (c / (c + d))
    n = a + b + c + d
    # chi-square with Yates correction
    df["chi2"] = n * (np.abs(a * d - b * c) - n / 2).clip(lower=0) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    df["prr_ci_low"] = np.exp(np.log(df["prr"]) - 1.96 * se)
    df["signal"] = (df["prr"] >= 2) & (df["chi2"] >= 4) & (df["a"] >= 3)
    return df.sort_values(["signal", "chi2"], ascending=False).reset_index(drop=True)


def weekly_counts(engine=None) -> pd.DataFrame:
    engine = engine or db.get_engine()
    wk = db.week_expr(engine, "r.received_date")
    sql = f"""
        SELECT d.name AS drug, {wk} AS week,
               COUNT(*) AS reports, SUM(r.serious) AS serious
        FROM reports r
        JOIN report_drugs rd ON rd.report_id = r.report_id AND rd.role = 'suspect'
        JOIN drugs d ON d.drug_id = rd.drug_id
        GROUP BY d.name, {wk}
        ORDER BY d.name, week
    """
    df = db.query(sql, engine=engine)
    df["week"] = pd.to_datetime(df["week"])
    return df


def spikes(window: int = 8, guard: int = 3, z_thresh: float = 3.5, engine=None) -> pd.DataFrame:
    """Baseline = weeks [t-guard-window, t-guard). The guard band stops a slowly
    ramping signal from contaminating its own baseline (same idea as CDC EARS C2/C3)."""
    df = weekly_counts(engine)
    out = []
    for drug, g in df.groupby("drug"):
        g = g.set_index("week")[["reports", "serious"]].asfreq("W-MON", fill_value=0).reset_index()
        g["drug"] = drug
        base = g["reports"].shift(guard + 1).rolling(window, min_periods=4)
        g["baseline"] = base.mean()
        # Poisson-floor on the std so tiny-count drugs do not explode
        g["z"] = (g["reports"] - g["baseline"]) / np.maximum(base.std(), np.sqrt(g["baseline"].clip(lower=1)))
        g["spike"] = g["z"] >= z_thresh
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df


def term_trend(drug: str, term: str, weeks: int = 8, engine=None) -> dict:
    """Recent vs prior count of a (drug, reaction) pair - is it accelerating?"""
    engine = engine or db.get_engine()
    sql = """
        SELECT r.received_date FROM reports r
        JOIN report_drugs rd ON rd.report_id = r.report_id AND rd.role = 'suspect'
        JOIN drugs d ON d.drug_id = rd.drug_id
        JOIN report_reactions rr ON rr.report_id = r.report_id
        WHERE LOWER(d.name) = LOWER(:drug) AND rr.reaction_term = :term
    """
    dates = pd.to_datetime(db.query(sql, {"drug": drug, "term": term.lower()}, engine=engine)["received_date"])
    end = pd.to_datetime(db.query("SELECT MAX(received_date) AS m FROM reports", engine=engine)["m"].iloc[0])
    recent = int((dates > end - pd.Timedelta(weeks=weeks)).sum())
    prior = int(((dates <= end - pd.Timedelta(weeks=weeks)) & (dates > end - pd.Timedelta(weeks=2 * weeks))).sum())
    return {"recent_count": recent, "prior_count": prior, "all_time_count": int(len(dates)),
            "first_seen": str(dates.min().date()) if len(dates) else None, "window_weeks": weeks}


_CACHE: dict[str, pd.DataFrame] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _cached(key: str, fn, engine):
    if engine is not None:          # explicit engine (tests) -> no caching
        return fn(engine=engine)
    if key not in _CACHE:
        _CACHE[key] = fn(engine=None)
    return _CACHE[key]


def signal_context(drug: str, reactions: list[str], engine=None) -> dict:
    """Agent tool payload: disproportionality + trend for each reaction of one drug."""
    prr = _cached("prr", prr_table, engine)
    rows = prr[prr["drug"].str.lower() == drug.lower()]
    ctx = []
    for t in reactions:
        r = rows[rows["reaction_term"] == t.lower()]
        item = {"reaction": t.lower(), **term_trend(drug, t, engine=engine)}
        if len(r):
            r = r.iloc[0]
            item.update(prr=round(float(r.prr), 2), prr_ci_low=round(float(r.prr_ci_low), 2),
                        chi2=round(float(r.chi2), 1), cases=int(r.a), disproportionality_signal=bool(r.signal))
        else:
            item.update(prr=None, cases=0, disproportionality_signal=False)
        ctx.append(item)
    sp = _cached("spikes", spikes, engine)
    d = sp[sp["drug"].str.lower() == drug.lower()].tail(4)
    return {
        "drug": drug,
        "reactions": ctx,
        "volume_spike_last_4_weeks": bool(d["spike"].any()) if len(d) else False,
        "latest_week_z": round(float(d["z"].iloc[-1]), 2) if len(d) and pd.notna(d["z"].iloc[-1]) else None,
    }
