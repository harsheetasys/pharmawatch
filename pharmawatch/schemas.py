from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["URGENT", "HIGH", "ROUTINE"]
PRIORITY_ORDER = {"ROUTINE": 0, "HIGH": 1, "URGENT": 2}


class Complaint(BaseModel):
    drugs: list[str] = Field(..., min_length=1, description="Suspect drug first, then concomitant drugs")
    reactions: list[str] = Field(default_factory=list, description="MedDRA-style terms; extracted from narrative if empty")
    narrative: str | None = None
    age: float | None = Field(None, ge=0, le=120)
    sex: Literal["F", "M", "U"] | None = "U"
    weight: float | None = Field(None, gt=0, le=400)
    reporter: Literal["consumer", "physician", "pharmacist", "other_hp", "lawyer"] | None = "consumer"
    report_id: str | None = None

    model_config = {"json_schema_extra": {"example": {
        "drugs": ["Zentravex"], "reactions": ["interstitial lung disease", "dyspnoea"],
        "narrative": "Physician reports a 58-year-old woman on Zentravex with new dry cough and breathlessness; CT shows interstitial lung disease.",
        "age": 58, "sex": "F", "reporter": "physician"}}}


class ReactionStatus(BaseModel):
    reaction: str
    label_status: str
    disproportionality_signal: bool = False
    recent_count: int | None = None


class LLMDecision(BaseModel):
    """What the LLM is allowed to decide. Everything else is computed by tools."""
    priority: Priority
    new_signal_suspected: bool
    rationale: str = Field(..., description="3-5 sentences, cite evidence as [chunk_id]")
    recommended_actions: list[str]
    citations: list[str] = Field(default_factory=list, description="chunk_ids from check_known_risk evidence")


class TriageReport(BaseModel):
    priority: Priority
    severity_probability: float
    risk_band: str
    reactions: list[ReactionStatus]
    new_signal_suspected: bool
    rationale: str
    recommended_actions: list[str]
    citations: list[dict]
    guardrail_notes: list[str] = Field(default_factory=list)
    backend: str
    tool_calls: list[str]
    latency_ms: int
