# PharmaWatch: real-data benchmark results

## Signal detection: OMOP reference set (replication of Harpaz et al. 2013)
- Data: openFDA FAERS, receivedate up to 2011-09-30 (2,487,057 reports; openFDA holds data from 2004 onward, while the paper used AERS from 1968 with 4.78M reports)
- Reference: OHDSI MethodEvaluation omopReferenceSet, 399 pairs (165 positive, 234 negative), 4 outcomes
- Outcomes defined as groups of MedDRA preferred terms (benchmarks/omop_prr.py OUTCOME_TERMS)
- Coverage: 346 pairs on harmonised generic names; 391/399 after name aliases plus free-text `medicinalproduct` fallback (paper: 394)

### Final comparison (AUC)
| Method | A) same 346 pairs | B) all 391 pairs | Paper (FAERS) |
|---|---|---|---|
| PRR (point estimate) | 0.653 | n/a | n/a |
| PRR05 | 0.708 | 0.715 | 0.71 |
| ROR05 | 0.709 | 0.714 | 0.71 |
| IC025 (BCPNN-style Bayesian) | **0.734** | 0.721 | n/a |
| MGPS EBGM | 0.680 | 0.691 | n/a |
| MGPS EB05 (our empirical-Bayes fit) | **0.729** | **0.725** | 0.79 |
| Logistic regression (report-level) | not attempted | not attempted | 0.83 |

### Paired bootstrap (2,000 resamples)
| Comparison | A) 346 pairs | B) 391 pairs |
|---|---|---|
| IC025 − PRR05 | +0.026 [+0.004, +0.048] ✅ | +0.007 [−0.016, +0.028] |
| EB05 − PRR05 | +0.022 [+0.007, +0.037] ✅ | +0.010 [−0.005, +0.024] |
| EB05 − IC025 | −0.004 [−0.016, +0.007] (tie) | +0.003 [−0.008, +0.015] (tie) |

Fitted MGPS prior (counts only, no labels): 0.25 × Gamma(7.76, 1.93) [mean ≈ 4.0, "elevated"] + 0.75 × Gamma(3.45, 2.83) [mean ≈ 1.2, "near baseline"]

By outcome (B), IC025 vs PRR05: Upper GI 0.837 vs 0.822 · Liver 0.775 vs 0.734 · Renal 0.715 vs 0.712 · MI 0.658 vs 0.704

### Conclusions
- PRR05 replicates the published 0.71, and ROR05 ties it, as in the paper.
- Both Bayesian shrinkage methods (IC025, EB05) beat PRR05 significantly on identical pairs. They are statistically indistinguishable from each other.
- Lower-bound scores beat point estimates every time (PRR05 > PRR, EB05 > EBGM), which matches the paper.
- Our EB05 (0.729) falls short of the paper's 0.79. Likely reasons: the prior was fit on ~390 hand-picked pairs (≈40% positives) rather than millions of database pairs; openFDA covers 2004+ only; outcome term groups are approximate; counts include non-suspect co-medication.
- Free-text drug matching increases coverage but dilutes gains (B not significant).

Defensible claim: "Replicated a published pharmacovigilance benchmark on real FDA data (PRR05 AUC 0.708 vs 0.71 reported). Implemented Bayesian disproportionality methods (IC025, MGPS/empirical-Bayes gamma-Poisson shrinker), which improved AUC to 0.73 on the same pairs (EB05 +0.022, 95% paired-bootstrap CI +0.007 to +0.037). Raised coverage from 346 to 391 of 399 pairs via drug-name harmonisation."

## Severity model: GLP-1 receptor agonist reports (vs ISPOR 2026 poster)
Poster baseline: ~480k FAERS GLP-1 cases 2015–2025, serious = any outcome code, 25% random holdout → LR ROC-AUC 0.876, RF 0.916.

### Data (benchmarks/glp1_download_v2.py)
- v1 sample ("first 500 reports per month") was **selection-biased** (dulaglutide 43% vs true 27%; tirzepatide 19% vs true 39%).
- v2: random blocks (5 × 100 from random offsets per month) plus inverse-probability `sample_weight` (month_total / sampled). This gives 63,606 reports and reproduces the true FDA molecule mix within 0.2 pp (tirzepatide 38.5% vs 38.7%, dulaglutide 27.3/27.3, semaglutide 19.5/19.5, liraglutide 11.3/11.1, exenatide 2.3/2.3, lixisenatide 1.0/1.1).
- openFDA harmonised names give ~313k GLP-1 reports, against the poster's ~480k (the poster used RxNorm mapping of raw names).
- Weighted serious share 30.8%. By molecule: tirzepatide 16%, dulaglutide 25%, semaglutide 53%, liraglutide 53%, exenatide 47%, lixisenatide 19%.
- Leakage guards: PTs death/hospitalisation/disability/life threatening removed from inputs; fulfillexpeditecriteria never downloaded; top-100 PT vocabulary learned on train split only.

### Results (weighted test ROC-AUC; benchmarks/glp1_model.py, glp1_ablation.py)
| Features | Logistic regression | XGBoost | Random forest |
|---|---|---|---|
| A) random split, all features | 0.944 | **0.960** (selected) | 0.946 |
| B) time split (train ≤2023, test 2024–25), all features | 0.947 | **0.952** (selected) | 0.939 |
| Ablation: minus country | 0.916 | 0.940 | n/a |
| Ablation: minus reporting context (country, report type, reporter, year) = **clinical-only** | **0.907** | **0.932** | n/a |
| Poster | 0.876 | n/a | 0.916 |

- Top XGBoost features (all-features model): country_us (0.135), ae_pancreatic, ae_injection_site, impaired gastric emptying, ae_renal, cerebrovascular accident, incorrect dose administered, report_type_2, ae_gallbladder, myocardial infarction, dehydration, diabetic ketoacidosis.
- Reporting-context features add +0.027 ROC-AUC (95% bootstrap CI +0.024 to +0.029, unweighted bootstrap). This is a **FAERS regulatory artifact**: non-US and study reports reach FDA mostly when serious. PR-AUC drops more (0.928 → 0.874).
- Clinical-only models still exceed the poster (XGB 0.932 vs RF 0.916; LR 0.907 vs LR 0.876). This is **not a head-to-head comparison**: different data extraction (openFDA-harmonised vs RxNorm), a 63k weighted sample, the poster's exact features are unknown, and the poster is a conference abstract.

### Final deployed model (benchmarks/glp1_train_final.py → artifacts/glp1_severity.joblib)
Clinical-only XGBoost, test ROC-AUC **0.932**, PR-AUC 0.874, Brier 0.095. Thresholds were chosen on validation (weighted) and checked on test:

| Mode | Threshold | Recall | Precision | Share flagged |
|---|---|---|---|---|
| Flag for review (target recall 90%) | 0.222 | **0.905** | 0.66 | 42.6% |
| High risk (target precision 85%) | 0.584 | 0.664 | **0.857** | 24.1% |

Calibration (test, 10 groups): predicted vs actual 0.002/0.001 · 0.010/0.015 · 0.028/0.025 · 0.057/0.049 · 0.115/0.141 · 0.238/0.251 · 0.419/0.428 · 0.632/0.655 · 0.874/0.848 · 0.990/0.989. Well calibrated.

Wired into PharmaWatch via pharmawatch/real_model.py. model.score() dispatches to the real model when the suspect drug is a GLP-1 (brand names mapped: Ozempic/Wegovy/Rybelsus, Mounjaro/Zepbound, Trulicity, Victoza/Saxenda, Byetta/Bydureon, Adlyxin) and falls back to the demo model otherwise. Training and serving share the same build_features (no train/serve skew). Smoke test: Ozempic + pancreatitis + vomiting → p=0.99 high (ae_pancreatic +2.87); Mounjaro + injection site pain → p=0.00 low. Agent triage of Ozempic + pancreatitis gives HIGH. 21 tests pass.

Defensible claim: "Reviewing only the 43% of GLP-1 reports flagged by the model catches 90.5% of serious cases (validation-chosen threshold, held-out test), with well-calibrated probabilities."

## Engineering notes
- Each openFDA call took ~44 s because of a failed IPv6 connect plus fallback. Forcing IPv4 and reusing a requests.Session brought it to ~1.7 s.
- On-disk cache of counts (benchmarks/data/openfda_count_cache.json) and per-month raw files make re-runs and experiments free and resumable.
- Windows SQLite 999-variable limit fixed in db.load_frames (executemany for SQLite).
- pandas groupby().apply returned a different shape across versions; replaced with named .agg().
- Notepad paste mishaps: use a new filename for big rewrites, then `python -m py_compile` to check before running.

## Next
- 5d: real GLP-1 drug labels (openFDA label endpoint) into the RAG index, so check_known_risk works on real drugs
- 5c: README / interview guide / resume bullets refreshed with real numbers