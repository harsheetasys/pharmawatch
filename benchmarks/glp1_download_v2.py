"""Lesson 4a (v2): download a RANDOM sample of real FAERS reports for GLP-1 drugs.

v1 took "the first 500 reports per month" - that is NOT random (FDA's index order groups
reports, e.g. by manufacturer batch), so the drug mix came out wrong. v2 fixes this:
for every month we ask how many reports exist, then pull several blocks from random
positions in that month's list. Each month is saved to its own file -> resumable.
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.openfda_counts import API_KEY, BASE, SESSION, total   # re-uses the IPv4 + retry fixes

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "glp1_raw_random"
OUT_PATH = DATA_DIR / "glp1_reports_v2.pkl"

GLP1 = ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide", "lixisenatide"]
DRUG_Q = "(" + " OR ".join(f'patient.drug.openfda.generic_name:"{d}"' for d in GLP1) + ")"
AGE_UNIT_TO_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 1 / 52, "804": 1 / 365, "805": 1 / 8760}
REPORTER = {"1": "physician", "2": "pharmacist", "3": "other_hp", "4": "lawyer", "5": "consumer"}
SEX = {"1": "M", "2": "F"}
MAX_SKIP = 25000          # openFDA will not page deeper than this


def month_query(year: int, month: int) -> str:
    first = pd.Timestamp(year, month, 1)
    last = first + pd.offsets.MonthEnd(0)
    return f"receivedate:[{first:%Y%m%d} TO {last:%Y%m%d}] AND {DRUG_Q}"


def get_page(search: str, skip: int, limit: int) -> list[dict]:
    params = {"search": search, "limit": limit, "skip": skip}
    if API_KEY:
        params["api_key"] = API_KEY
    r = SESSION.get(BASE, params=params, timeout=(10, 120))
    if r.status_code == 404:
        return []
    r.raise_for_status()
    time.sleep(0.3)
    return r.json().get("results", [])


def fetch_month(year: int, month: int, blocks: int = 5, block_size: int = 100, seed: int = 42) -> tuple[list[dict], int]:
    """Random sample of up to blocks*block_size reports from one month. Returns (reports, month_total)."""
    path = RAW_DIR / f"{year}-{month:02d}.json.gz"
    if path.exists():                                            # already downloaded -> skip
        saved = json.loads(gzip.decompress(path.read_bytes()))
        return saved["results"], saved["month_total"]
    search = month_query(year, month)
    n = total(search)
    reachable = min(n, MAX_SKIP)
    starts = np.arange(0, reachable, block_size)                  # possible block start positions
    rng = np.random.default_rng(seed + year * 100 + month)        # reproducible per month
    if len(starts) > blocks:
        starts = np.sort(rng.choice(starts, blocks, replace=False))
    results = []
    for s in starts:
        results.extend(get_page(search, int(s), block_size))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({"month_total": n, "results": results}).encode()))
    return results, n


def age_years(p: dict):
    try:
        years = float(p.get("patientonsetage")) * AGE_UNIT_TO_YEARS.get(str(p.get("patientonsetageunit", "801")), 1.0)
        return years if 0 <= years <= 110 else None
    except (TypeError, ValueError):
        return None


def parse(ev: dict) -> dict | None:
    p = ev.get("patient") or {}
    drugs = p.get("drug") or []
    glp1_drug = None
    for d in drugs:
        names = " ".join((d.get("openfda") or {}).get("generic_name", [])).lower()
        hit = next((g for g in GLP1 if g in names), None)
        if hit and (glp1_drug is None or str(d.get("drugcharacterization")) == "1"):
            glp1_drug = (hit, d)
    if glp1_drug is None:
        return None
    molecule, d = glp1_drug
    try:
        weight = float(p.get("patientweight")) if p.get("patientweight") else None
    except ValueError:
        weight = None
    return {
        "report_id": str(ev.get("safetyreportid")),
        "received": pd.to_datetime(ev.get("receivedate"), format="%Y%m%d", errors="coerce"),
        "serious": 1 if str(ev.get("serious")) == "1" else 0,                       # TARGET
        "death": 1 if str(ev.get("seriousnessdeath")) == "1" else 0,                # outcome detail (never a feature)
        "hospitalised": 1 if str(ev.get("seriousnesshospitalization")) == "1" else 0,
        "age": age_years(p),
        "sex": SEX.get(str(p.get("patientsex")), "U"),
        "weight": weight,
        "reporter": REPORTER.get(str((ev.get("primarysource") or {}).get("qualification")), "unknown"),
        "country_us": 1 if (ev.get("occurcountry") or "").upper() == "US" else 0,
        "report_type": str(ev.get("reporttype") or "unknown"),
        "molecule": molecule,
        "glp1_is_suspect": 1 if str(d.get("drugcharacterization")) == "1" else 0,
        "indication": (d.get("drugindication") or "unknown").lower(),
        "n_drugs": len(drugs),
        "reactions": sorted({(r.get("reactionmeddrapt") or "").lower() for r in p.get("reaction") or []} - {""}),
    }


def main(first_year: int = 2015, last_year: int = 2025) -> pd.DataFrame:
    rows, months = [], [(y, m) for y in range(first_year, last_year + 1) for m in range(1, 13)]
    for i, (y, m) in enumerate(months, 1):
        t0 = time.time()
        batch, n = fetch_month(y, m)
        parsed = [x for x in (parse(ev) for ev in batch) if x]
        for x in parsed:
            x["sample_weight"] = n / max(len(batch), 1)      # each sampled report "stands for" this many real ones
        rows.extend(parsed)
        print(f"{y}-{m:02d}: {len(parsed):4d} sampled of {n:6,d}  ({time.time() - t0:.1f}s)  [{i}/{len(months)}]", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("report_id")
    df.to_pickle(OUT_PATH)
    return df


if __name__ == "__main__":
    df = main()
    w = df.sample_weight
    print(f"\nSaved {len(df):,} reports to {OUT_PATH}")
    print(f"Serious share: {np.average(df.serious, weights=w):.1%} (weighted to real monthly volumes)\n")
    mix = df.groupby("molecule").apply(lambda g: pd.Series({
        "sampled": len(g), "est_real_reports": g.sample_weight.sum(),
        "serious_share": np.average(g.serious, weights=g.sample_weight)}), include_groups=False)
    mix["est_real_share"]