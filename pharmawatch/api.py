"""FastAPI service.

    GET  /health              liveness + which backends are active
    POST /score               ML severity score only (fast path, no LLM)
    POST /triage              full agent triage report (tools + reasoning + citations)
    GET  /signals             current PRR signals, optionally only unlabeled ones
    GET  /signals/spikes      weekly volume spikes
    GET  /model/metrics       offline evaluation metrics of the deployed model
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from . import __version__, config, model, rag, signals
from .agent import get_agent
from .schemas import Complaint, TriageReport

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pharmawatch")

app = FastAPI(title="PharmaWatch", version=__version__,
              description="Drug-safety complaint scoring, signal detection and agentic triage.")


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid, t0 = uuid.uuid4().hex[:12], time.time()
    response = await call_next(request)
    log.info(json.dumps({"request_id": rid, "path": request.url.path, "method": request.method,
                         "status": response.status_code, "ms": int((time.time() - t0) * 1000)}))
    response.headers["x-request-id"] = rid
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    ok = config.MODEL_PATH.exists() and config.INDEX_PATH.exists()
    return {"status": "ok" if ok else "artifacts_missing", "version": __version__,
            "llm_backend": config.LLM_BACKEND if config.GEMINI_API_KEY else "offline",
            "embedder": config.EMBEDDER, "data_mode": config.DATA_MODE}


@app.post("/score")
def score(c: Complaint):
    if not c.reactions:
        raise HTTPException(422, "reactions required for /score (use /triage to extract from narrative)")
    return model.score(c.model_dump()).__dict__


@app.post("/triage", response_model=TriageReport)
def triage(c: Complaint):
    try:
        report = get_agent().triage(c)
    except ValueError as e:
        raise HTTPException(422, str(e))
    log.info(json.dumps({"event": "triage", "drug": c.drugs[0], "priority": report.priority,
                         "backend": report.backend, "guardrails": len(report.guardrail_notes),
                         "latency_ms": report.latency_ms}))
    return report


@app.get("/signals")
def get_signals(unlabeled_only: bool = False, limit: int = 50):
    df = signals.prr_table()
    df = df[df["signal"]]
    rows = []
    for r in df.itertuples():
        status = None
        if unlabeled_only or len(rows) < limit:
            status = rag.check_known_risk(r.drug, [r.reaction_term])["results"][0]["status"]
        if unlabeled_only and status != "not_in_label":
            continue
        rows.append({"drug": r.drug, "reaction": r.reaction_term, "cases": int(r.a), "prr": round(r.prr, 2),
                     "prr_ci_low": round(r.prr_ci_low, 2), "chi2": round(r.chi2, 1), "label_status": status})
        if len(rows) >= limit:
            break
    return rows


@app.get("/signals/spikes")
def get_spikes():
    sp = signals.spikes()
    sp = sp[sp["spike"]]
    return [{"drug": r.drug, "week": str(r.week.date()), "reports": int(r.reports),
             "baseline": round(float(r.baseline), 1), "z": round(float(r.z), 2)} for r in sp.itertuples()]


@app.get("/model/metrics")
def metrics():
    if not config.METRICS_PATH.exists():
        raise HTTPException(404, "model not trained yet")
    return json.loads(config.METRICS_PATH.read_text())
