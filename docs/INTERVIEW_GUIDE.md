# PharmaWatch: interview guide

## 30-second pitch
> "PharmaWatch triages adverse-event complaints for a drug-safety team. A classical ML model scores how likely a report is to be serious. A RAG layer over drug labels checks whether each reaction is a known risk or a new one. SQL statistics (PRR and volume-spike detection) show whether this is part of an emerging signal. One agent calls all three as tools and writes a cited triage decision. Guardrails stop the LLM from ever downgrading a safety-critical case. On a synthetic world with a planted signal it catches the signal three independent ways and has zero under-triage on the golden set."

## Each piece in one breath (with the analogy)

| piece | what it does | analogy | why it's built this way |
|---|---|---|---|
| **Schema** | reports ↔ drugs ↔ reactions, many-to-many, like FAERS | an order with many line items | flat CSVs can't express "suspect vs concomitant drug" |
| **Window feature** | share of serious reports among this drug's *earlier* reports | a credit score built only from your past loans | `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` = no peeking at the future (leakage) |
| **Time split** | train on old reports, test on new ones | rehearse on last year's exam, sit this year's | random splits leak future drug history and inflate scores |
| **Recall-first threshold** | pick the cutoff that catches 90% of serious cases | a smoke alarm tuned to be over-sensitive | a missed serious case (FN) costs far more than an extra review (FP) |
| **Two thresholds** | the recall threshold → HIGH; the precision threshold → counts as "serious evidence" for URGENT | a triage nurse vs. a consultant sign-off | one number can't do both jobs without flooding the URGENT queue |
| **LogReg beat XGBoost** | simpler model won on validation PR-AUC | don't take a truck to the corner shop | always run a baseline first; choose on validation, never on test |
| **SHAP-style factors** | per-prediction feature contributions | an itemized bill | reviewers need to know *why* the model flagged a case |
| **PRR** | how much more often this reaction shows up with this drug than with all other drugs | "umbrellas sell 5x more in this shop than city-wide" | standard regulatory screening statistic; Evans criteria PRR≥2, χ²≥4, n≥3 |
| **Guard-banded spike detection** | this week vs a baseline that skips the last 3 weeks | don't compare a fever to yesterday's fever | a slowly ramping signal otherwise drags its own baseline up (I hit this bug; z went from 2.7 to 4.6 after the fix) |
| **RAG known-risk check** | retrieve label chunks for the drug, then confirm the term really appears | a librarian finds the page, you read the sentence yourself | dense similarity alone can't be trusted to say "yes, it's labeled"; hybrid = retrieve + verify |
| **Spelling normalization** | MedDRA "haemorrhage" = US label "hemorrhage" | | a real bug class in pharmacovigilance text matching |
| **Agent, 3 tools** | the LLM decides which tools to call and writes the rationale | a case officer with three reference desks | "ML model wrapped as an agent tool" pattern |
| **Priority floor** | deterministic rules set a minimum priority; the LLM can escalate, never downgrade | a junior can escalate but can't close a safety case alone | the LLM adds judgment on narratives without becoming a single point of failure |
| **Citation guardrail** | cited chunk_ids must come from this run's tool outputs | "show me the page" | catches hallucinated citations (tested with a fake LLM that invents one) |
| **Fallback** | any LLM error → deterministic policy | a generator when the power cuts | a safety system can't go down because of an API quota |
| **Golden-set eval** | 20 labeled cases, tracking under-triage separately | a mock exam marked on the dangerous mistakes first | accuracy hides the asymmetric cost of errors |

## Keyword coverage by job title

- **Data Scientist (Amgen-style)**: SQL (CTEs, window functions), feature engineering, logistic regression, XGBoost, time-based validation, precision/recall trade-off, PR-AUC, calibration (Brier), model explainability, statistical signal detection (PRR, chi-square, confidence intervals, z-scores), dashboards (Streamlit, Plotly), pharmacovigilance/FAERS domain.
- **ML Engineer**: train/serve parity (one shared `featurize()`), model artifacts, FastAPI serving, Docker/Compose, CI-ready pytest suite, Postgres + SQLite portability, structured request logging, health checks, deployment (Render).
- **LLM / GenAI Engineer**: RAG (chunking with overlap, embeddings, metadata filtering, hybrid verification), Gemini function calling, temperature 0, schema-constrained JSON output (Pydantic), grounding and citations, hallucination checks, offline evals.
- **Agentic AI Engineer**: tool-using agent loop, tool design, ML-model-as-tool, guardrails (priority floor, citation validation, schema validation), graceful degradation, LLM vs rule-based baseline evaluation, agent testing with a scripted fake client.

## Resume bullets (choose 3–4)

- Built **PharmaWatch**, a drug-safety triage system combining a classical ML severity model, RAG over drug labels and a Gemini tool-calling agent. Its guardrails held under-triage at **0%** with **100% URGENT recall** on a 20-case golden set.
- Engineered leakage-safe features in **Postgres SQL (CTEs, window functions)** over 23.8k FAERS-style reports. Benchmarked logistic regression vs **XGBoost** on a time-based split and tuned the decision threshold to reach **91% recall** on held-out future data.
- Implemented pharmacovigilance signal detection (**PRR/chi-square disproportionality** and guard-banded weekly spike detection). It surfaced a planted emerging safety signal about 5 weeks after onset with **zero false volume alarms** across 10 drugs.
- Designed an agent that wraps the ML model, RAG retriever and SQL statistics as **tools**. Deterministic guardrails (priority floor, citation verification, Pydantic schema validation, offline fallback) stop the LLM from downgrading safety-critical cases or citing evidence it never retrieved.
- Shipped as a **FastAPI** service and **Streamlit** dashboard with Docker Compose (Postgres) and 21 pytest tests, including agent-loop tests against a scripted fake LLM.

*(Once you run it on openFDA data and/or with Gemini, swap in those numbers. Real-data metrics carry more weight.)*

## Questions you will get

**Why is precision only ~46%?** I chose that operating point on purpose. At 90% recall, about half of the flagged cases are serious. The alternative is missing more serious cases. The PR curve is in `/model/metrics`, and the second, precision-tuned threshold decides what counts as URGENT.

**How do you know the LLM isn't making things up?** Three ways. Every label claim must cite a chunk_id that a tool returned in this run, and anything else is removed and logged. The label status itself comes from the RAG tool, not the LLM. The priority can't go below the deterministic floor.

**Why not just use the LLM for everything?** It's slower, it costs money per call, and you can't audit it statistically. The ML model gives a calibrated probability across thousands of reports in milliseconds. The LLM's job is narrative judgment and explanation, and it's evaluated against the rule baseline.

**PRR flags labeled reactions too. Isn't that noise?** Yes. Known reactions are disproportionate by definition. That's why the useful view is PRR signal **AND** not in the label. One chance hit (8 cases) remains, which is why signals go to medical review rather than straight to action.

**What would you do next?** Real openFDA run, deduplication of FAERS follow-ups, MedDRA coding with an LLM, LLM-as-judge on rationale quality, monitoring triage-priority drift, and Bayesian disproportionality (EBGM/BCPNN) in place of PRR for sparse pairs.
