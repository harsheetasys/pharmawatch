import json

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from pharmawatch.agent import GeminiTriageAgent, OfflineTriageAgent, extract_reactions, seriousness_criteria
from pharmawatch.api import app
from pharmawatch.schemas import Complaint

SIGNAL_CASE = Complaint(drugs=["Zentravex"], reactions=["interstitial lung disease"], age=60, sex="F",
                        reporter="physician", narrative="Admitted with breathlessness.")


def test_extract_reactions_from_lay_language(world):
    terms = extract_reactions("Noticed black stool, felt faint and had trouble breathing.")
    assert {"gastrointestinal haemorrhage", "syncope", "dyspnoea"} <= set(terms)


def test_seriousness_criteria():
    assert seriousness_criteria("was hospitalised overnight") == ["hospitalised"]
    assert seriousness_criteria("mild rash") == []


def test_offline_agent_priorities(world):
    a = OfflineTriageAgent()
    assert a.triage(SIGNAL_CASE).priority == "URGENT"
    assert a.triage(Complaint(drugs=["Neurisol"], reactions=["nausea"], age=29, sex="F")).priority == "ROUTINE"
    r = a.triage(SIGNAL_CASE)
    assert r.new_signal_suspected and set(r.tool_calls) == {"score_severity", "check_known_risk", "get_signal_context"}


# ---- Gemini agent with a scripted fake client (no network) ------------------------------
def _fc_response(*calls):
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def _text_response(obj):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=json.dumps(obj))]))])


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        self.calls.append(config)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


ARGS = {"drug": "Zentravex", "reactions": ["interstitial lung disease"]}


def test_gemini_agent_tool_loop_and_priority_floor(world):
    script = [
        _fc_response(("score_severity", {"drugs": ["Zentravex"], "reactions": ["interstitial lung disease"]}),
                     ("check_known_risk", ARGS)),
        _fc_response(("get_signal_context", ARGS)),
        _text_response({"priority": "ROUTINE", "new_signal_suspected": True,
                        "rationale": "Looks mild [Zentravex:boxed_warning:0] [made-up:chunk:9].",
                        "recommended_actions": ["review"], "citations": ["made-up:chunk:9"]}),
    ]
    rep = GeminiTriageAgent(client=FakeClient(script), model_name="fake").triage(SIGNAL_CASE)
    assert rep.tool_calls == ["score_severity", "check_known_risk", "get_signal_context"]
    assert rep.priority == "URGENT"                                     # LLM said ROUTINE -> floor wins
    assert any("safety floor" in n for n in rep.guardrail_notes)
    assert any("dropped 1 citation" in n for n in rep.guardrail_notes)  # hallucinated chunk removed
    assert "[made-up:chunk:9]" not in rep.rationale and rep.citations == []


def test_gemini_agent_falls_back_on_error(world):
    rep = GeminiTriageAgent(client=FakeClient([RuntimeError("quota")]), model_name="fake").triage(SIGNAL_CASE)
    assert rep.backend == "gemini->offline-fallback" and rep.priority == "URGENT"
    assert "quota" in rep.guardrail_notes[0]


def test_gemini_agent_can_escalate(world):
    c = Complaint(drugs=["Neurisol"], reactions=["nausea"], age=29, sex="F")
    script = [_text_response({"priority": "HIGH", "new_signal_suspected": False, "rationale": "Escalating.",
                              "recommended_actions": ["follow up"], "citations": []})]
    rep = GeminiTriageAgent(client=FakeClient(script), model_name="fake").triage(c)
    assert rep.priority == "HIGH"
    assert any("skipped tool" in n for n in rep.guardrail_notes)


# ---- API ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(world):
    return TestClient(app)


def test_api_health_and_score(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/score", json={"drugs": ["Velmora"], "reactions": ["gastrointestinal haemorrhage"], "age": 80})
    assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1


def test_api_triage_and_validation(client):
    r = client.post("/triage", json=SIGNAL_CASE.model_dump())
    assert r.status_code == 200 and r.json()["priority"] == "URGENT"
    assert client.post("/triage", json={"drugs": []}).status_code == 422
    assert client.post("/triage", json={"drugs": ["Neurisol"], "narrative": "no symptoms"}).status_code == 422


def test_api_signals(client):
    rows = client.get("/signals", params={"unlabeled_only": True}).json()
    assert any(r["drug"] == "Zentravex" and r["reaction"] == "interstitial lung disease" for r in rows)
    assert client.get("/model/metrics").json()["selected_model"] in {"xgboost", "logistic_regression"}
