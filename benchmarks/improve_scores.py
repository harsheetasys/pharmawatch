"""Lesson 3b + 3c: try to beat PRR05 on the OMOP reference set.

3c  Recover missing drugs: fix name mismatches, and fall back to the free-text
    drug field (medicinalproduct) when FDA's harmonised field has no match.
3b  Compare scoring methods on the SAME pairs:
      PRR05  - proportional reporting ratio, lower bound      (paper: 0.71)
      ROR05  - reporting odds ratio, lower bound              (paper: 0.71)
      IC025  - Bayesian "information component", lower bound (WHO / Uppsala method)
    Paper's Bayesian MGPS (EB05) scored 0.79; its best (logistic regression) 0.83.
"""
import math

import pandas as pd
from sklearn.metrics import roc_auc_score

from benchmarks.omop_prr import DATA_DIR, DATE_Q, OUTCOME_TERMS, CachedCounter, drug_query, outcome_query

# Name fixes: OMOP name -> names FDA actually uses (tried in order).
ALIASES = {
    "chlorazepate": ["clorazepate"],
    "mefenamate": ["mefenamic acid"],
    "estrogens, conjugated (usp)": ["estrogens, conjugated", "conjugated estrogens"],
    "lithium citrate": ["lithium"],
    "factor viia": ["coagulation factor viia"],
    "tetrahydrocannabinol": ["dronabinol"],
}


def candidate_queries(drug: str) -> list[str]:
    names = ALIASES.get(drug, []) + [drug]
    harmonised = [drug_query(n) for n in names]                                   # FDA's cleaned field
    free_text = [f'patient.drug.medicinalproduct:"{n}"' for n in names]           # what the reporter typed
    return harmonised + free_text


def resolve_drug(counter: CachedCounter, drug: str) -> tuple[str | None, int]:
    """First query that finds the drug in FAERS -> (query, number of reports)."""
    for q in candidate_queries(drug):
        n = counter.count(f"{DATE_Q} AND {q}")
        if n > 0:
            return q, n
    return None, 0


def scores(a: int, n_drug: int, n_event: int, n_all: int) -> dict:
    b, c = max(n_drug - a, 0), max(n_event - a, 0)
    d = max(n_all - n_drug - c, 0)
    A, B, C, D = (x + 0.5 for x in (a, b, c, d))
    z = 1.645                                                   # 90% interval -> 5th percentile

    prr = (A / (A + B)) / (C / (C + D))
    prr_se = math.sqrt(1 / A - 1 / (A + B) + 1 / C - 1 / (C + D))

    ror = (A * D) / (B * C)
    ror_se = math.sqrt(1 / A + 1 / B + 1 / C + 1 / D)

    expected = n_drug * n_event / n_all                         # reports we'd expect by chance
    ic = math.log2((a + 0.5) / (expected + 0.5))
    ic025 = ic - 3.3 * (a + 0.5) ** -0.5 - 2 * (a + 0.5) ** -1.5   # Noren et al. 2013 approximation

    return {"prr05": math.exp(math.log(prr) - z * prr_se),
            "ror05": math.exp(math.log(ror) - z * ror_se),
            "ic025": ic025, "expected": expected}


def run() -> pd.DataFrame:
    ref = pd.read_csv(DATA_DIR / "omop_reference_set.csv")
    counter = CachedCounter()
    n_all = counter.count(DATE_Q)
    n_event = {o: counter.count(f"{DATE_Q} AND {outcome_query(o)}") for o in OUTCOME_TERMS}

    rows = []
    for r in ref.itertuples():
        q, n_drug = resolve_drug(counter, r.exposureName)
        row = {"drug": r.exposureName, "outcome": r.outcomeName, "truth": r.groundTruth,
               "matched_by": q, "n_drug": n_drug,
               "original_match": q == drug_query(r.exposureName)}
        if q:
            a = counter.count(f"{DATE_Q} AND {q} AND {outcome_query(r.outcomeName)}")
            row.update(a=a, **scores(a, n_drug, n_event[r.outcomeName], n_all))
        rows.append(row)
    counter.save()
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "omop_scores_compared.csv", index=False)
    return out


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: [roc_auc_score(df.truth, df[m])] for m in ["prr05", "ror05", "ic025"]},
                        index=[f"n={len(df)}"]).round(3)


if __name__ == "__main__":
    res = run()
    scored = res.dropna(subset=["prr05"])
    same = scored[scored.original_match]
    print(f"\nCoverage: before = {int(res.original_match.sum())} pairs, after fixes = {len(scored)} of {len(res)}")
    print("Recovered drugs:", sorted(scored[~scored.original_match].drug.unique()))
    print("Still missing:  ", sorted(res[res.prr05.isna()].drug.unique()))

    print("\nA) Same pairs as Lesson 3 (fair method comparison):")
    print(auc_table(same).to_string())
    print("\nB) All recovered pairs:")
    print(auc_table(scored).to_string())
    print("\nPaper on FAERS: PRR05 0.71 | ROR05 0.71 | MGPS EB05 0.79 | logistic regression 0.83")
    print("\nAUC by outcome (B, IC025 vs PRR05):")
    for o, g in scored.groupby("outcome"):
        print(f"   {o:38s} IC025 {roc_auc_score(g.truth, g.ic025):.3f} | PRR05 {roc_auc_score(g.truth, g.prr05):.3f}  (n={len(g)})")