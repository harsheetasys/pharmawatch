"""Lesson 3: score all OMOP reference pairs with PRR on real FAERS data, then compute AUC.

Published baseline (Harpaz et al. 2013, FAERS through 2011 Q3): PRR05 AUC = 0.71.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmarks.openfda_counts import total

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "openfda_count_cache.json"
RESULTS_PATH = DATA_DIR / "omop_prr_results.csv"
END_DATE = "20110930"          # same cut-off as the paper (2011 Q3)
DATE_Q = f"receivedate:[19680101 TO {END_DATE}]"

# Each OMOP outcome is a GROUP of MedDRA preferred terms (our approximation).
OUTCOME_TERMS = {
    "OMOP Acute Liver Failure 1": [
        "hepatic failure", "acute hepatic failure", "hepatitis fulminant", "hepatic necrosis",
        "liver injury", "drug-induced liver injury", "hepatotoxicity", "hepatocellular injury",
        "hepatitis", "jaundice",
    ],
    "OMOP Acute Renal Failure 1": [
        "acute kidney injury", "renal failure acute", "renal failure", "renal impairment",
        "nephropathy toxic", "renal tubular necrosis",
    ],
    "OMOP Acute myocardial Infarction  1": [
        "myocardial infarction", "acute myocardial infarction",
    ],
    "HOI Upper GI #3": [
        "gastrointestinal haemorrhage", "upper gastrointestinal haemorrhage", "haematemesis",
        "melaena", "gastric haemorrhage", "gastric ulcer haemorrhage",
        "duodenal ulcer haemorrhage", "peptic ulcer haemorrhage",
    ],
}


def outcome_query(outcome: str) -> str:
    terms = " OR ".join(f'patient.reaction.reactionmeddrapt:"{t}"' for t in OUTCOME_TERMS[outcome])
    return f"({terms})"


def drug_query(drug: str) -> str:
    return f'patient.drug.openfda.generic_name:"{drug}"'


class CachedCounter:
    """Remembers every answer on disk, so a re-run costs zero API calls."""

    def __init__(self):
        self.cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        self.calls = 0

    def count(self, search: str) -> int:
        if search not in self.cache:
            self.cache[search] = total(search)
            self.calls += 1
            time.sleep(0.3)                              # stay under 240 requests/minute
            if self.calls % 25 == 0:
                CACHE_PATH.write_text(json.dumps(self.cache))
                print(f"   ... {self.calls} API calls so far")
        return self.cache[search]

    def save(self):
        CACHE_PATH.write_text(json.dumps(self.cache))


def prr_scores(a: int, n_drug: int, n_event: int, n_all: int) -> tuple[float, float]:
    b, c = max(n_drug - a, 0), max(n_event - a, 0)
    d = max(n_all - n_drug - c, 0)
    a, b, c, d = (x + 0.5 for x in (a, b, c, d))
    value = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    return value, math.exp(math.log(value) - 1.645 * se)


def run() -> pd.DataFrame:
    ref = pd.read_csv(DATA_DIR / "omop_reference_set.csv")
    counter = CachedCounter()
    n_all = counter.count(DATE_Q)
    n_event = {o: counter.count(f"{DATE_Q} AND {outcome_query(o)}") for o in OUTCOME_TERMS}

    rows = []
    for i, r in enumerate(ref.itertuples(), 1):
        n_drug = counter.count(f"{DATE_Q} AND {drug_query(r.exposureName)}")
        a = counter.count(f"{DATE_Q} AND {drug_query(r.exposureName)} AND {outcome_query(r.outcomeName)}") if n_drug else 0
        value, low = prr_scores(a, n_drug, n_event[r.outcomeName], n_all) if n_drug else (None, None)
        rows.append({"drug": r.exposureName, "outcome": r.outcomeName, "truth": r.groundTruth,
                     "a": a, "n_drug": n_drug, "prr": value, "prr05": low})
        if i % 50 == 0:
            print(f"{i}/{len(ref)} pairs scored")
    counter.save()
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_PATH, index=False)
    return out


if __name__ == "__main__":
    res = run()
    found = res.dropna(subset=["prr"])
    print(f"\nPairs in reference set: {len(res)} | drug found in openFDA: {len(found)}")
    print(f"AUC (PRR)   = {roc_auc_score(found.truth, found.prr):.3f}")
    print(f"AUC (PRR05) = {roc_auc_score(found.truth, found.prr05):.3f}   <- paper: 0.71\n")
    print("AUC by outcome (PRR05):")
    for o, g in found.groupby("outcome"):
        print(f"   {o:38s} {roc_auc_score(g.truth, g.prr05):.3f}   (n={len(g)})")
    print("\nDrugs not found in openFDA:", sorted(res[res.prr.isna()].drug.unique()))