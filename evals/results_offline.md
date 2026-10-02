### Eval: offline-policy (20 cases)

| metric | value |
|---|---|
| priority_accuracy | 0.95 |
| under_triage_rate | 0.0 |
| over_triage_rate | 0.05 |
| urgent_recall | 1.0 |
| label_status_accuracy | 1.0 |
| new_signal_accuracy | 1.0 |
| citation_validity | 1.0 |
| guardrail_interventions | 0 |
| mean_latency_ms | 77 |
| wall_time_s | 1.6 |

| case | drug | expected | predicted | ok |
|---|---|---|---|---|
| G01 | Zentravex | URGENT | URGENT | ✅ |
| G02 | Zentravex | URGENT | URGENT | ✅ |
| G03 | Zentravex | ROUTINE | ROUTINE | ✅ |
| G04 | Zentravex | HIGH | HIGH | ✅ |
| G05 | Neurisol | ROUTINE | ROUTINE | ✅ |
| G06 | Neurisol | URGENT | URGENT | ✅ |
| G07 | Velmora | HIGH | HIGH | ✅ |
| G08 | Velmora | ROUTINE | HIGH | over |
| G09 | Pressavil | HIGH | HIGH | ✅ |
| G10 | Pressavil | ROUTINE | ROUTINE | ✅ |
| G11 | Glucentra | ROUTINE | ROUTINE | ✅ |
| G12 | Glucentra | HIGH | HIGH | ✅ |
| G13 | Corvalin | URGENT | URGENT | ✅ |
| G14 | Corvalin | ROUTINE | ROUTINE | ✅ |
| G15 | Ostevia | HIGH | HIGH | ✅ |
| G16 | Ostevia | HIGH | HIGH | ✅ |
| G17 | Lipravex | URGENT | URGENT | ✅ |
| G18 | Thyrovance | ROUTINE | ROUTINE | ✅ |
| G19 | Respiqan | URGENT | URGENT | ✅ |
| G20 | Lipravex | HIGH | HIGH | ✅ |