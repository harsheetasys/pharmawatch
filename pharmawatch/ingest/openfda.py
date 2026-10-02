"""Real-data ingester: FDA FAERS adverse-event reports + drug labels via openFDA.

openFDA is free and public (https://open.fda.gov). Without an API key you get
240 requests/minute and 1,000 requests/day; set OPENFDA_API_KEY to raise that.
Each request returns at most 1,000 records and `skip` is capped at 25,000,
so we page per drug *and* per date window to stay inside those limits.

Usage:  python -m scripts.build_all --source openfda --per-drug 3000
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "https://api.fda.gov/drug"
API_KEY = os.getenv("OPENFDA_API_KEY")

# Ten widely used drugs; three are Amgen biologics (denosumab, etanercept, evolocumab).
DEFAULT_DRUGS = [
    "denosumab", "etanercept", "evolocumab", "adalimumab", "semaglutide",
    "apixaban", "atorvastatin", "metformin", "sertraline", "montelukast",
]
LABEL_SECTIONS = ["boxed_warning", "warnings_and_cautions", "warnings", "adverse_reactions", "contraindications"]
QUALIFICATION = {"1": "physician", "2": "pharmacist", "3": "other_hp", "4": "lawyer", "5": "consumer"}
SEX = {"1": "M", "2": "F", "0": "U"}
ROLE = {"1": "suspect", "2": "concomitant", "3": "interacting"}
AGE_UNIT_TO_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 1 / 52, "804": 1 / 365, "805": 1 / 8760}


def _get(endpoint: str, params: dict, retries: int = 4) -> dict:
    if API_KEY:
        params = {**params, "api_key": API_KEY}
    for attempt in range(retries):
        r = requests.get(f"{BASE}/{endpoint}.json", params=params, timeout=60)
        if r.status_code == 404:          # openFDA returns 404 for "no matches"
            return {"results": []}
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def _age_years(patient: dict) -> float | None:
    try:
        age = float(patient.get("patientonsetage"))
    except (TypeError, ValueError):
        return None
    years = age * AGE_UNIT_TO_YEARS.get(str(patient.get("patientonsetageunit", "801")), 1.0)
    return round(years, 1) if 0 <= years <= 120 else None


def _drug_name(d: dict) -> str | None:
    gen = (d.get("openfda") or {}).get("generic_name")
    name = (gen[0] if gen else d.get("medicinalproduct")) or ""
    name = name.strip().lower()
    return name or None


def parse_event(ev: dict) -> tuple[dict, list[dict], list[str]] | None:
    """Map one openFDA event record -> (report row, drug rows, reaction terms)."""
    try:
        rd = ev["receivedate"]
        received = date(int(rd[:4]), int(rd[4:6]), int(rd[6:8]))
    except (KeyError, ValueError):
        return None
    patient = ev.get("patient") or {}
    w = patient.get("patientweight")
    try:
        weight = float(w) if w is not None else None
    except ValueError:
        weight = None
    drugs = []
    for d in patient.get("drug") or []:
        name = _drug_name(d)
        if name:
            drugs.append({"name": name, "role": ROLE.get(str(d.get("drugcharacterization")), "concomitant")})
    terms = sorted({(r.get("reactionmeddrapt") or "").strip().lower()
                    for r in patient.get("reaction") or [] if r.get("reactionmeddrapt")})
    if not drugs or not terms:
        return None
    report = dict(
        report_id=str(ev.get("safetyreportid")),
        received_date=received,
        serious=1 if str(ev.get("serious")) == "1" else 0,
        patient_age=_age_years(patient),
        patient_sex=SEX.get(str(patient.get("patientsex")), "U"),
        patient_weight=weight,
        reporter_qualification=QUALIFICATION.get(str((ev.get("primarysource") or {}).get("qualification")), "consumer"),
        country=(ev.get("occurcountry") or (ev.get("primarysource") or {}).get("reportercountry") or "")[:8],
        narrative=None,   # FAERS public data has no free-text narrative; we synthesize one from structured fields
    )
    return report, drugs, terms


def fetch_events(drug: str, per_drug: int, start: date, end: date) -> list[dict]:
    """Page through events for one suspect drug, walking the date range in windows."""
    out: list[dict] = []
    windows = 12
    span = (end - start) / windows
    per_window = max(1, per_drug // windows)
    for i in range(windows):
        a, b = start + span * i, start + span * (i + 1) - timedelta(days=1)
        search = (f'patient.drug.openfda.generic_name:"{drug}" AND patient.drug.drugcharacterization:1 '
                  f"AND receivedate:[{a:%Y%m%d} TO {b:%Y%m%d}]")
        skip = 0
        while skip < per_window:
            batch = _get("event", {"search": search, "limit": min(1000, per_window - skip), "skip": skip}).get("results", [])
            out.extend(batch)
            if len(batch) < 1000:
                break
            skip += len(batch)
            time.sleep(0.3)
    return out


def fetch_label(drug: str) -> dict[str, str]:
    res = _get("label", {"search": f'openfda.generic_name:"{drug}"', "limit": 1}).get("results", [])
    if not res:
        return {}
    lab = res[0]
    return {s: "\n".join(lab[s]) for s in LABEL_SECTIONS if lab.get(s)}


def build(drugs: list[str] | None = None, per_drug: int = 3000,
          start: date = date(2023, 1, 1), end: date = date(2025, 12, 31)) -> dict[str, pd.DataFrame]:
    drugs = [d.lower() for d in (drugs or DEFAULT_DRUGS)]
    reports, rdrugs, rreacts, seen = [], [], [], set()
    name_to_id: dict[str, int] = {}

    def drug_id(name: str) -> int:
        if name not in name_to_id:
            name_to_id[name] = len(name_to_id) + 1
        return name_to_id[name]

    for d in drugs:
        drug_id(d)
    for d in drugs:
        print(f"[openfda] events for {d} ...", flush=True)
        for ev in fetch_events(d, per_drug, start, end):
            parsed = parse_event(ev)
            if not parsed or parsed[0]["report_id"] in seen:
                continue
            report, dlist, terms = parsed
            seen.add(report["report_id"])
            # keep the tracked drug as the suspect; other drugs as recorded (dedupe per report)
            ids = {}
            for x in dlist:
                i = drug_id(x["name"])
                if i not in ids or x["role"] == "suspect":
                    ids[i] = x["role"]
            report["narrative"] = (f"{report['reporter_qualification'].replace('_', ' ').title()} report: "
                                   f"{report['patient_age'] or 'unknown-age'} y/o {report['patient_sex']} on "
                                   f"{', '.join(x['name'] for x in dlist[:4])}; reactions: {', '.join(terms)}.")
            reports.append(report)
            rdrugs.extend({"report_id": report["report_id"], "drug_id": i, "role": r} for i, r in ids.items())
            rreacts.extend({"report_id": report["report_id"], "reaction_term": t} for t in terms)

    labels = []
    for d in drugs:
        print(f"[openfda] label for {d} ...", flush=True)
        for section, body in fetch_label(d).items():
            labels.append({"drug_id": name_to_id[d], "section": section, "body": body})

    drugs_df = pd.DataFrame({"drug_id": list(name_to_id.values()), "name": list(name_to_id.keys()), "drug_class": None})
    return {
        "drugs": drugs_df,
        "reports": pd.DataFrame(reports),
        "report_drugs": pd.DataFrame(rdrugs),
        "report_reactions": pd.DataFrame(rreacts),
        "label_sections": pd.DataFrame(labels),
    }
