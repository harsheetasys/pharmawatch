"""Lesson 4a: download real FAERS reports for GLP-1 drugs (Ozempic, Wegovy, Mounjaro, Victoza...).

We take a sample per month (default 500) from 2015-2025, so the data covers the whole period
without downloading all ~480k reports. Each month is saved to its own file -> resumable.
"""
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd

from benchmarks.openfda_counts import API_KEY, BASE, SESSION   # re-uses the IPv4 + retry fixes

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "glp1_raw"
OUT_PATH = DATA_DIR / "glp1_reports.pkl"

GLP1 = ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide", "lixisenatide", "albiglutide"]
DRUG_Q = "(" + " OR ".join(f'patient.drug.openfda.generic_name:"{d}"' for d in GLP1) + ")"
AGE_UNIT_TO_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 1 / 52, "804": 1 / 365, "805": 1 / 8760}
REPORTER = {"1": "physician", "2": "pharmacist", "3": "other_hp", "4": "lawyer", "5": "consumer"}
SEX = {"1": "M", "2": "F"}


def fetch_month(year: int, month: int, limit: int) -> list[dict]:
    path = RAW_DIR / f"{year}-{month:02d}.json.gz"
    if path.exists():                                            # already downloaded -> skip
        return json.loads(gzip.decompress(path.read_bytes()))
    start = f"{year}{month:02d}01"
    end = f"{year + (month == 12)}{(month % 12) + 1:02d}01"
    params = {"search": f"receivedate:[{start} TO {end}] AND {DRUG_Q}", "limit": limit}
    if API_KEY:
        params["api_key"] = API_KEY
    r = SESSION.get(BASE, params=params, timeout=(10, 120))
    results = [] if r.status_code == 404 else r.json().get("results", [])
    if r.status_code not in (200, 404):
        r.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(results).encode()))
    time.sleep(0.3)
    return results


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


def main(limit: int = 500, first_year: int = 2015, last_year: int = 2025) -> pd.DataFrame:
    rows, months = [], [(y, m) for y in range(first_year, last_year + 1) for m in range(1, 13)]
    for i, (y, m) in enumerate(months, 1):
        t0 = time.time()
        batch = fetch_month(y, m, limit)
        parsed = [x for x in (parse(ev) for ev in batch) if x]
        rows.extend(parsed)
        print(f"{y}-{m:02d}: {len(parsed):4d} reports  ({time.time() - t0:.1f}s)  [{i}/{len(months)}]", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("report_id")
    df.to_pickle(OUT_PATH)
    return df


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    df = main(limit)
    print(f"\nSaved {len(df):,} reports to {OUT_PATH}")
    print(f"Serious share: {df.serious.mean():.1%}   |   by molecule:")
    print(df.groupby("molecule").agg(reports=("serious", "size"), serious_share=("serious", "mean")).round(3).to_string())