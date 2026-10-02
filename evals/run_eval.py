"""Golden-set evaluation of the triage agent.

    python -m evals.run_eval                 # offline policy
    python -m evals.run_eval --backend gemini  # LLM agent (needs GEMINI_API_KEY)

Metrics (why each matters in drug safety):
* priority_accuracy      - exact match with the expert label
* under_triage_rate      - predicted LOWER than expected  (the dangerous error)
* over_triage_rate       - predicted HIGHER than expected (costs reviewer time)
* urgent_recall          - share of truly URGENT cases caught as URGENT
* label_status_accuracy  - RAG correctly said labeled/boxed/warning/not-in-label
* new_signal_accuracy    - agrees on "is this a suspected new signal?"
* citation_validity      - every citation is from the suspect drug's label
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pharmawatch.agent import get_agent
from pharmawatch.schemas import PRIORITY_ORDER, Complaint

HERE = Path(__file__).parent


def run(backend: str = "offline") -> dict:
    cases = json.loads((HERE / "golden_set.json").read_text())
    agent = get_agent(backend)
    rows, t0 = [], time.time()
    for case in cases:
        c = Complaint(**case["complaint"])
        rep = agent.triage(c)
        exp, got = case["expected_priority"], rep.priority
        status = {r.reaction: r.label_status for r in rep.reactions}
        st_checks = [status.get(k) == v for k, v in case.get("expected_status", {}).items()]
        drug = c.drugs[0].lower()
        cites_ok = all(ci["chunk_id"].lower().startswith(drug + ":") for ci in rep.citations)
        rows.append({
            "id": case["id"], "drug": c.drugs[0], "expected": exp, "predicted": got,
            "correct": exp == got, "under": PRIORITY_ORDER[got] < PRIORITY_ORDER[exp],
            "over": PRIORITY_ORDER[got] > PRIORITY_ORDER[exp],
            "status_correct": sum(st_checks), "status_total": len(st_checks),
            "new_signal_ok": rep.new_signal_suspected == case["expected_new_signal"],
            "citations_ok": cites_ok, "p_serious": rep.severity_probability,
            "guardrail_notes": rep.guardrail_notes, "latency_ms": rep.latency_ms, "backend": rep.backend,
        })
    n = len(rows)
    urgent = [r for r in rows if r["expected"] == "URGENT"]
    summary = {
        "backend": rows[0]["backend"] if rows else backend, "n_cases": n,
        "priority_accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "under_triage_rate": round(sum(r["under"] for r in rows) / n, 3),
        "over_triage_rate": round(sum(r["over"] for r in rows) / n, 3),
        "urgent_recall": round(sum(r["predicted"] == "URGENT" for r in urgent) / max(1, len(urgent)), 3),
        "label_status_accuracy": round(sum(r["status_correct"] for r in rows) / max(1, sum(r["status_total"] for r in rows)), 3),
        "new_signal_accuracy": round(sum(r["new_signal_ok"] for r in rows) / n, 3),
        "citation_validity": round(sum(r["citations_ok"] for r in rows) / n, 3),
        "guardrail_interventions": sum(bool(r["guardrail_notes"]) for r in rows),
        "mean_latency_ms": int(sum(r["latency_ms"] for r in rows) / n),
        "wall_time_s": round(time.time() - t0, 1),
    }
    out = {"summary": summary, "cases": rows}
    (HERE / f"results_{backend}.json").write_text(json.dumps(out, indent=2))
    return out


def to_markdown(out: dict) -> str:
    s = out["summary"]
    lines = [f"### Eval: {s['backend']} ({s['n_cases']} cases)", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s.items() if k not in ("backend", "n_cases")]
    lines += ["", "| case | drug | expected | predicted | ok |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['drug']} | {r['expected']} | {r['predicted']} | {'✅' if r['correct'] else ('⚠️ under' if r['under'] else 'over')} |"
              for r in out["cases"]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="offline", choices=["offline", "gemini"])
    args = ap.parse_args()
    res = run(args.backend)
    md = to_markdown(res)
    (HERE / f"results_{args.backend}.md").write_text(md)
    print(md)
