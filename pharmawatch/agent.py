"""Triage agent: one agent, three tools.

    score_severity      -> the trained ML model, wrapped as a tool
    check_known_risk    -> RAG over drug labels (known vs. NEW risk, with citations)
    get_signal_context  -> PRR disproportionality + trend statistics from SQL

Two backends share the same tools and the same output schema:
* GeminiTriageAgent - LLM decides which tools to call (function calling,
  temperature 0) and writes the rationale; final answer is schema-constrained JSON.
* OfflineTriageAgent - deterministic policy; used when no API key is set, as
  the LLM's fallback on any failure, and as a baseline in the evals.

Guardrails (applied to the LLM's answer, never optional):
1. Priority FLOOR: deterministic safety rules set a minimum priority.
   The LLM may escalate, but can never downgrade below the floor.
2. Citation check: every cited chunk_id must have been returned by a tool
   during this run; anything else is dropped and noted.
3. Schema validation (pydantic); on failure -> fall back to the offline policy.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from . import config, db, model, rag, signals
from .schemas import PRIORITY_ORDER, Complaint, LLMDecision, ReactionStatus, TriageReport

# "Important Medical Events" style list: always urgent regardless of model score.
IME_TERMS = {"death", "anaphylactic reaction", "stevens-johnson syndrome", "toxic epidermal necrolysis",
             "hepatic failure", "myocardial infarction", "pulmonary embolism", "seizure", "suicidal ideation",
             "suicide attempt", "agranulocytosis", "cardiac arrest", "respiratory failure"}

LAY_TERMS = {
    r"short(ness)? of breath|breathless|trouble breathing|difficulty breathing": "dyspnoea",
    r"swell(ing|ed) (of )?(the )?(face|lips|tongue|throat)": "angioedema",
    r"\bfaint(ed|ing)?\b|passed out|blacked out": "syncope",
    r"\bdied\b|\bdeath\b|passed away": "death",
    r"black stool|blood in (the )?stool|vomiting blood|stomach bleed": "gastrointestinal haemorrhage",
    r"throwing up|threw up": "vomiting",
    r"low blood sugar": "hypoglycaemia",
    r"racing heart|heart racing": "tachycardia",
    r"jaw (bone )?(pain|exposed bone).*|exposed jaw bone": "osteonecrosis of jaw",
    r"muscle (pain|ache)s?": "myalgia",
    r"(convulsion|fit)s?\b": "seizure",
    r"thoughts of (suicide|killing)|wants? to die": "suicidal ideation",
    r"itch(ing|y)": "pruritus",
    r"severe allergic reaction|anaphylaxis": "anaphylactic reaction",
}


# ------------------------------------------------------------------ helpers
@lru_cache(maxsize=1)
def _vocab() -> tuple[str, ...]:
    terms = db.query("SELECT DISTINCT reaction_term FROM report_reactions")["reaction_term"].tolist()
    return tuple(sorted(terms, key=len, reverse=True))


def extract_reactions(narrative: str) -> list[str]:
    """Deterministic narrative -> MedDRA-style terms (vocabulary + lay-language map)."""
    text = rag.normalize_spelling(narrative.lower())
    found = []
    for t in _vocab():
        if re.search(rf"\b{re.escape(rag.normalize_spelling(t))}\b", text):
            found.append(t)
    for pat, term in LAY_TERMS.items():
        if re.search(pat, text) and term not in found:
            found.append(term)
    return found


SERIOUSNESS_CRITERIA = re.compile(
    r"hospitali[sz](ed|ation)|admitted|life[- ]threatening|intensive care|\bicu\b|disabilit|"
    r"congenital|birth defect|\bdied\b|\bdeath\b|fatal", re.I)


def seriousness_criteria(narrative: str | None) -> list[str]:
    """Regulatory (ICH E2A) seriousness criteria stated in the narrative, e.g. 'hospitalised'."""
    return sorted({m.group(0).lower() for m in SERIOUSNESS_CRITERIA.finditer(narrative or "")})


def safety_floor(sev: dict, known: dict, sig: dict, narrative: str | None = None) -> tuple[str, list[str]]:
    """Deterministic MINIMUM priority + the reasons for it.

    Two thresholds, two jobs: the recall-tuned threshold (flag_for_review) only
    escalates to HIGH; 'serious' evidence for URGENT needs the precision-tuned
    band, an important medical event, or a stated regulatory seriousness criterion.
    """
    reasons = []
    statuses = {r["reaction"]: r["status"] for r in known["results"]}
    sig_by = {r["reaction"]: r for r in sig["reactions"]}
    unlabeled = [r for r, s in statuses.items() if s == "not_in_label"]
    ime = sorted(set(statuses) & IME_TERMS)
    criteria = seriousness_criteria(narrative)
    serious_evidence = bool(ime or criteria or sev["risk_band"] == "high")
    emerging = [r for r in unlabeled
                if sig_by.get(r, {}).get("disproportionality_signal")
                and (sig_by[r].get("recent_count") or 0) >= 3] or ([*unlabeled] if sig.get("volume_spike_last_4_weeks") else [])

    if ime:
        reasons.append(f"important medical event(s): {', '.join(ime)}")
    if unlabeled and serious_evidence:
        why = ", ".join(filter(None, [f"criteria: {', '.join(criteria)}" if criteria else "",
                                      "model band high" if sev["risk_band"] == "high" else "",
                                      "IME" if ime else ""]))
        reasons.append(f"serious ({why}) AND unexpected (not in label): {', '.join(unlabeled)} "
                       "-> candidate for 15-day expedited report")
    if emerging:
        reasons.append(f"unlabeled reaction(s) with an active safety signal: {', '.join(emerging)}")
    if reasons:
        return "URGENT", reasons
    if criteria:
        return "HIGH", [f"serious per narrative ({', '.join(criteria)}) but expected (labeled)"]
    if sev["risk_band"] == "high":
        return "HIGH", [f"model risk band HIGH (p={sev['probability']:.2f})"]
    if sev["flag_for_review"]:
        return "HIGH", [f"model flags for review (p={sev['probability']:.2f} >= threshold {sev['threshold']:.2f})"]
    if unlabeled:
        return "HIGH", [f"unexpected but non-serious reaction(s): {', '.join(unlabeled)}"]
    return "ROUTINE", ["low predicted severity and all reactions are labeled"]


def _run_tools(c: Complaint) -> dict:
    drug = c.drugs[0]
    sev = model.score(c.model_dump()).__dict__
    known = rag.check_known_risk(drug, c.reactions)
    sig = signals.signal_context(drug, c.reactions)
    return {"score_severity": sev, "check_known_risk": known, "get_signal_context": sig}


def _reaction_rows(known: dict, sig: dict) -> list[ReactionStatus]:
    sig_by = {r["reaction"]: r for r in sig["reactions"]}
    return [ReactionStatus(reaction=r["reaction"], label_status=r["status"],
                           disproportionality_signal=bool(sig_by.get(r["reaction"], {}).get("disproportionality_signal")),
                           recent_count=sig_by.get(r["reaction"], {}).get("recent_count"))
            for r in known["results"]]


def _citation_pool(known: dict) -> dict[str, dict]:
    return {e["chunk_id"]: {"chunk_id": e["chunk_id"], "section": e["section"], "quote": e["quote"]}
            for r in known["results"] for e in r["evidence"] if r["status"] != "not_in_label"}


def _prepare(c: Complaint) -> Complaint:
    if not c.reactions and c.narrative:
        c = c.model_copy(update={"reactions": extract_reactions(c.narrative)})
    if not c.reactions:
        raise ValueError("No reactions given and none could be extracted from the narrative.")
    return c


# ------------------------------------------------------------------ offline agent
class OfflineTriageAgent:
    backend = "offline-policy"

    def triage(self, complaint: Complaint) -> TriageReport:
        t0 = time.time()
        c = _prepare(complaint)
        out = _run_tools(c)
        sev, known, sig = out["score_severity"], out["check_known_risk"], out["get_signal_context"]
        prio, reasons = safety_floor(sev, known, sig, c.narrative)
        pool = _citation_pool(known)
        unlabeled = [r["reaction"] for r in known["results"] if r["status"] == "not_in_label"]
        labeled = [f"{r['reaction']} ({r['status'].replace('labeled_', '').replace('_', ' ')})"
                   for r in known["results"] if r["status"].startswith("labeled")]
        rationale = (f"Severity model ({sev['model']}) estimates P(serious)={sev['probability']:.2f} "
                     f"[{sev['risk_band']}]. "
                     + (f"Known/labeled: {', '.join(labeled)}. " if labeled else "")
                     + (f"NOT in label: {', '.join(unlabeled)}. " if unlabeled else "")
                     + f"Priority {prio} because: {'; '.join(reasons)}.")
        return TriageReport(
            priority=prio, severity_probability=sev["probability"], risk_band=sev["risk_band"],
            reactions=_reaction_rows(known, sig), new_signal_suspected=_emerging_signal(known, sig),
            rationale=rationale, recommended_actions=_actions(prio, unlabeled),
            citations=list(pool.values())[:6], guardrail_notes=[], backend=self.backend,
            tool_calls=list(out), latency_ms=int((time.time() - t0) * 1000))


def _emerging_signal(known: dict, sig: dict) -> bool:
    """A SIGNAL is aggregate evidence, not one case: an unlabeled reaction with an
    active disproportionality signal and recent cases (or a volume spike)."""
    sig_by = {r["reaction"]: r for r in sig["reactions"]}
    unlabeled = [r["reaction"] for r in known["results"] if r["status"] == "not_in_label"]
    return any(sig_by.get(r, {}).get("disproportionality_signal") and (sig_by[r].get("recent_count") or 0) >= 3
               for r in unlabeled) or bool(unlabeled and sig.get("volume_spike_last_4_weeks"))


def _actions(prio: str, unlabeled: list[str]) -> list[str]:
    acts = {
        "URGENT": ["Route to safety physician for medical review within 24h",
                   "Request follow-up information (onset, dechallenge/rechallenge, labs)"],
        "HIGH": ["Medical review within 3 business days", "Request follow-up information"],
        "ROUTINE": ["Process in routine case-processing queue", "Include in periodic aggregate report"],
    }[prio]
    if unlabeled:
        acts.append(f"Assess expectedness/causality for unlabeled event(s): {', '.join(unlabeled)}; "
                    "add to signal-evaluation tracker")
    return acts


# ------------------------------------------------------------------ Gemini agent
SYSTEM_PROMPT = """You are a pharmacovigilance triage assistant. For the complaint provided you MUST:
1. call score_severity, check_known_risk and get_signal_context (all three) for the suspect drug;
2. decide a triage priority: URGENT (possible serious + unexpected event, important medical event,
   or emerging signal), HIGH (likely serious but expected, or unexpected but not serious), ROUTINE;
3. ground every claim about the label in the tool evidence and cite it as [chunk_id];
   never claim a reaction is labeled unless check_known_risk says so.
When unsure between two priorities, choose the higher one. Patient safety over workload."""

TOOL_DECLS = [
    {"name": "score_severity",
     "description": "Trained ML model: probability the complaint is a SERIOUS adverse event, risk band, and top contributing features.",
     "parameters_json_schema": {"type": "object", "properties": {
         "drugs": {"type": "array", "items": {"type": "string"}},
         "reactions": {"type": "array", "items": {"type": "string"}},
         "age": {"type": "number"}, "sex": {"type": "string", "enum": ["F", "M", "U"]},
         "reporter": {"type": "string"}}, "required": ["drugs", "reactions"]}},
    {"name": "check_known_risk",
     "description": "Search the drug's official label (RAG) to see whether each reaction is already a labeled risk; returns status + quoted evidence with chunk_ids.",
     "parameters_json_schema": {"type": "object", "properties": {
         "drug": {"type": "string"}, "reactions": {"type": "array", "items": {"type": "string"}}},
         "required": ["drug", "reactions"]}},
    {"name": "get_signal_context",
     "description": "Database statistics for drug-reaction pairs: PRR disproportionality, chi-square, case counts, recent trend, volume spikes.",
     "parameters_json_schema": {"type": "object", "properties": {
         "drug": {"type": "string"}, "reactions": {"type": "array", "items": {"type": "string"}}},
         "required": ["drug", "reactions"]}},
]


def _dispatch(name: str, args: dict, c: Complaint) -> dict:
    if name == "score_severity":
        merged = {**c.model_dump(), **{k: v for k, v in args.items() if v is not None}}
        return model.score(merged).__dict__
    if name == "check_known_risk":
        return rag.check_known_risk(args.get("drug") or c.drugs[0], args.get("reactions") or c.reactions)
    if name == "get_signal_context":
        return signals.signal_context(args.get("drug") or c.drugs[0], args.get("reactions") or c.reactions)
    return {"error": f"unknown tool {name}"}


def _try_parse(resp) -> LLMDecision | None:
    try:
        txt = (resp.text or "").strip().removeprefix("```json").removesuffix("```").strip()
        return LLMDecision.model_validate_json(txt)
    except Exception:
        return None


class GeminiTriageAgent:
    backend = "gemini"

    def __init__(self, client=None, model_name: str | None = None, max_steps: int = 6):
        if client is None:
            from google import genai
            client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.client, self.model_name, self.max_steps = client, model_name or config.GEMINI_MODEL, max_steps

    def triage(self, complaint: Complaint) -> TriageReport:
        from google.genai import types
        t0 = time.time()
        c = _prepare(complaint)
        notes: list[str] = []
        try:
            tools = [types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in TOOL_DECLS])]
            cfg = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, tools=tools, temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            contents = [types.Content(role="user", parts=[types.Part.from_text(
                text="Complaint:\n" + c.model_dump_json(indent=1))])]
            called, cited_pool, decision = [], {}, None
            for _ in range(self.max_steps):
                resp = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
                calls = resp.function_calls or []
                if not calls:
                    decision = _try_parse(resp)       # model may already have answered in JSON
                    break
                contents.append(resp.candidates[0].content)
                parts = []
                for fc in calls:
                    result = _dispatch(fc.name, dict(fc.args or {}), c)
                    called.append(fc.name)
                    if fc.name == "check_known_risk":
                        cited_pool.update(_citation_pool(result))
                    parts.append(types.Part.from_function_response(
                        name=fc.name, response={"result": json.loads(json.dumps(result, default=str))}))
                contents.append(types.Content(role="user", parts=parts))

            if decision is None:                      # ask for schema-constrained final JSON
                contents.append(types.Content(role="user", parts=[types.Part.from_text(
                    text="Now give your final triage decision as JSON.")]))
                final = self.client.models.generate_content(
                    model=self.model_name, contents=contents,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0,
                                                       response_mime_type="application/json",
                                                       response_schema=LLMDecision))
                decision = final.parsed if isinstance(final.parsed, LLMDecision) else LLMDecision.model_validate_json(final.text)
        except Exception as e:  # any LLM/transport/schema failure -> safe deterministic fallback
            rep = OfflineTriageAgent().triage(c)
            rep.guardrail_notes.append(f"LLM failed ({type(e).__name__}: {str(e)[:120]}); used offline policy")
            rep.backend = "gemini->offline-fallback"
            return rep

        # ---- guardrails -------------------------------------------------
        truth = _run_tools(c)        # authoritative tool outputs on the TRUE complaint (cached model/index)
        floor, reasons = safety_floor(truth["score_severity"], truth["check_known_risk"], truth["get_signal_context"], c.narrative)
        priority = decision.priority
        if PRIORITY_ORDER[priority] < PRIORITY_ORDER[floor]:
            notes.append(f"priority raised {priority}->{floor} by safety floor: {'; '.join(reasons)}")
            priority = floor
        for missing in {"score_severity", "check_known_risk", "get_signal_context"} - set(called):
            notes.append(f"LLM skipped tool {missing}; result injected by guardrail")
        pool = {**_citation_pool(truth["check_known_risk"]), **cited_pool}
        good = [pool[cid] for cid in decision.citations if cid in pool]
        bad = [cid for cid in decision.citations if cid not in pool]
        if bad:
            notes.append(f"dropped {len(bad)} citation(s) not returned by tools: {bad}")
        rationale = decision.rationale
        for cid in bad:
            rationale = rationale.replace(f"[{cid}]", "[citation removed]")

        sev = truth["score_severity"]
        return TriageReport(
            priority=priority, severity_probability=sev["probability"], risk_band=sev["risk_band"],
            reactions=_reaction_rows(truth["check_known_risk"], truth["get_signal_context"]),
            new_signal_suspected=decision.new_signal_suspected, rationale=rationale,
            recommended_actions=decision.recommended_actions, citations=good, guardrail_notes=notes,
            backend=f"gemini:{self.model_name}", tool_calls=called, latency_ms=int((time.time() - t0) * 1000))


def get_agent(backend: str | None = None):
    backend = backend or config.LLM_BACKEND
    return GeminiTriageAgent() if backend == "gemini" and config.GEMINI_API_KEY else OfflineTriageAgent()
