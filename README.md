# PharmaWatch: drug-safety signal detection and agentic triage on real FDA data

PharmaWatch helps a drug-safety team decide **which adverse-event reports need a human first, and why**. It combines three tools behind one agent:

1. **Severity model.** XGBoost trained on **63.6k real FDA (FAERS) reports** for GLP-1 drugs (Ozempic, Mounjaro, Trulicity...). It predicts whether a report is serious.
2. **Signal detection.** Disproportionality statistics (PRR, ROR, Bayesian IC and MGPS), **benchmarked against a published study** on the OMOP reference set.
3. **Label check (RAG).** Hybrid search over the **real FDA drug labels**. It says whether a reaction is already a known risk and quotes the label text.

A tool-calling agent (Gemini, with a deterministic fallback) combines the three into a triage decision: **URGENT / HIGH / ROUTINE**. Safety guardrails stop the LLM from downgrading a case, and every citation is checked against what the tools actually returned.

Every headline number below comes from **real public data** and is compared with a **published baseline**.

---

## Results

### 1. Signal detection: replicating and improving on a published benchmark

**Benchmark:** the OMOP reference set has 399 drug–outcome pairs labelled by experts (165 real side effects, 234 non-effects). The baseline is Harpaz et al., *Clin Pharmacol Ther* 2013, on FAERS through 2011 Q3.

| Method | Our AUC (same 346 pairs) | Published |
|---|---|---|
| PRR05 (standard disproportionality) | **0.708** | 0.71 ✅ replicated |
| ROR05 | 0.709 | 0.71 ✅ replicated |
| IC025 (Bayesian, WHO-UMC style) | **0.734** | n/a |
| MGPS EB05 (FDA's empirical-Bayes gamma–Poisson shrinker, implemented from scratch) | **0.729** | 0.79 |

- **The Bayesian methods beat PRR05 by a significant margin:** EB05 − PRR05 = **+0.022** (95% paired-bootstrap CI **+0.007 to +0.037**). IC025 and EB05 are statistically tied.
- **Lower-bound scores always beat point estimates** (PRR05 > PRR, EB05 > EBGM), as the paper reports.
- **Drug-name harmonisation** (aliases plus a free-text fallback field) raised coverage from **346 to 391 of 399 pairs** (paper: 394). The noisier free-text matches dilute the Bayesian gain, so it isn't significant on all 391 pairs.
- **Why we're below the paper's 0.79:** openFDA only covers 2004 onwards (2.5M vs 4.8M reports); our MGPS prior is fitted on about 390 pairs rather than millions; the counts can't separate suspect drugs from co-medications.

### 2. Severity model: GLP-1 adverse-event reports

**Baseline:** an ISPOR 2026 poster on ~480k FAERS GLP-1 cases (2015–2025) with a 25% random holdout: logistic regression ROC-AUC 0.876, random forest 0.916.

| Model (weighted test ROC-AUC) | Random split | Time split (train ≤2023 → test 2024–25) |
|---|---|---|
| Logistic regression | 0.944 | 0.947 |
| Random forest | 0.946 | 0.939 |
| **XGBoost** (selected on validation) | **0.960** | **0.952** |
| **XGBoost, clinical features only** (deployed) | **0.932** | n/a |
| Poster | LR 0.876, RF 0.916 | n/a |

**What makes these numbers trustworthy:**

- **Selection bias found and fixed.** The first sample ("first 500 reports per month") over-represented one drug at 43% when its true share is 27%. Random-block sampling plus inverse-probability weights reproduces FDA's true drug mix **within 0.2 percentage points**.
- **Leakage removed.** Reaction terms that *are* the outcome (death, hospitalisation, ...) are excluded, and so is the expedited-report flag.
- **Shortcut learning measured.** The top feature was *"report from outside the US"*, a regulatory artifact: foreign cases mostly reach FDA when they're serious. Removing reporting-context features cost **0.027 AUC** (CI +0.024 to +0.029). The **clinical-only model (0.932)** is the one deployed, and it still exceeds the poster.
- **Thresholds chosen on validation, confirmed on test.** Reviewing the **42.6% of reports the model flags catches 90.5% of serious cases**. At the "high risk" threshold, 85.7% of flagged cases are serious.
- **Well calibrated.** Predicted vs actual serious rate per decile: 0.12/0.14 · 0.42/0.43 · 0.63/0.66 · 0.87/0.85 · 0.99/0.99.

*This isn't a strict head-to-head with the poster.* The data extraction differs (openFDA-harmonised names vs RxNorm), we use a 63k weighted sample, and the poster's exact features are unknown.

### 3. Label check (RAG) on real FDA labels

Real labels for 6 GLP-1 drugs come from openFDA (366 chunks). Swapping in real labels exposed a **retrieval-recall failure**. On the long Ozempic label, top-4 vector retrieval missed the *Acute Pancreatitis* warning, so the agent wrongly escalated a case to URGENT. The fix is **hybrid search**: an exhaustive lexical scan over the drug's label, with vectors used only for ranking, a small synonym layer (e.g. *thyroid cancer → medullary thyroid carcinoma*), and citations that quote the text around the match. Ozempic + pancreatitis now returns `labeled_warning` with the exact label quote, and the agent's triage is the correct **HIGH**.

---

## How it fits together

```
openFDA (FAERS events, counts, labels) ──► Postgres/SQLite ──┬─► severity model (XGBoost, clinical-only) ─┐
OMOP reference set (OHDSI) ──► benchmark harness             ├─► signal stats (PRR / ROR / IC / MGPS)    ├─► Triage agent ─► FastAPI · Streamlit
                                                             └─► label RAG (hybrid lexical + vector)     ─┘     (Gemini + guardrails)
```

**Agent guardrails:**

- **Priority floor:** deterministic safety rules set a minimum priority, and the LLM can only escalate.
- **Citation check:** a quoted chunk must have been returned by a tool in this run.
- **Validation and fallback:** the output is checked against a Pydantic schema, and the offline policy takes over if the LLM fails.

## Reproduce

```bash
pip install -r requirements.txt pyreadr
python -m scripts.build_all            # demo world + tests
python -m pytest -q                    # 21 tests

# Signal-detection benchmark (openFDA; set OPENFDA_API_KEY for higher limits)
python -m benchmarks.load_reference    # OMOP reference set (OHDSI)
python -m benchmarks.omop_prr          # PRR05 replication  -> 0.708
python -m benchmarks.improve_scores    # name fixes + ROR05 / IC025
python -m benchmarks.bootstrap_auc     # significance
python -m benchmarks.mgps              # MGPS EB05

# Severity model (GLP-1)
python -m benchmarks.glp1_true_mix     # true drug mix from FDA counts
python -m benchmarks.glp1_download_v2  # random-block sample (~30 min, resumable)
python -m benchmarks.glp1_summary      # sample vs true mix
python -m benchmarks.glp1_model        # LR / RF / XGBoost, random + time splits
python -m benchmarks.glp1_ablation     # reporting-artifact ablation
python -m benchmarks.glp1_train_final  # deployed model + thresholds + calibration
python -m benchmarks.glp1_labels       # real FDA labels into the RAG index

uvicorn pharmawatch.api:app --port 8000      # API docs at /docs
streamlit run dashboard/app.py               # dashboard
```

## Repository map

| Path | What |
|---|---|
| `pharmawatch/` | the app: DB, features, demo model, `real_model.py` (GLP-1 model adapter), signals, RAG, agent, API |
| `benchmarks/` | real-data benchmarks and training (every number in this README) |
| `evals/` | agent golden-set evaluation |
| `tests/` | 21 unit/integration tests (scripted fake LLM; no network needed) |
| `dashboard/` | Streamlit UI |

## Engineering notes (bugs found on the way)

- **API calls took ~44 s each.** Windows tried IPv6 first and waited for it to time out. Forcing IPv4 and reusing one HTTP session brought it to **1.7 s**.
- **Resumability.** An on-disk cache of counts and per-month raw files means interrupted runs resume and experiments cost zero API calls.
- **SQLite on Windows** allows only 999 parameters per statement, so inserts switch to `executemany` there.
- **pandas versions differ** in what `groupby().apply` returns, so the code uses named `.agg()`.

## Limitations

- FAERS is voluntary reporting: under-reporting, duplicates, and media- or litigation-driven spikes. Disproportionality ≠ causation.
- The openFDA-harmonised drug field misses some reports (~313k vs ~480k GLP-1 cases).
- The lixisenatide label returned by openFDA is a combination product (Soliqua, with insulin glargine).
- The synonym layer is a small stand-in for the full MedDRA hierarchy.
- The demo world (fictional drugs) is used only for unit tests and the agent eval; it's never used for accuracy claims.