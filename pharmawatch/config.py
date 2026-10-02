"""Central configuration. Everything is driven by environment variables so the
same code runs on a laptop (SQLite, no API key) and in Docker (Postgres + Gemini)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.getenv("PW_ARTIFACTS", ROOT / "artifacts"))
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Database: Postgres in Docker/production, SQLite fallback for quick local runs.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ARTIFACTS / 'pharmawatch.db'}")

# Data source: "synthetic" (fictional drugs, planted signal, fully offline)
# or "openfda" (real FDA FAERS adverse-event reports + real drug labels).
DATA_MODE = os.getenv("PW_DATA_MODE", "synthetic")

# LLM: Gemini if a key is present, otherwise the deterministic offline policy.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("PW_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("PW_GEMINI_EMBED_MODEL", "gemini-embedding-001")
LLM_BACKEND = os.getenv("PW_LLM", "gemini" if GEMINI_API_KEY else "offline")
EMBEDDER = os.getenv("PW_EMBEDDER", "tfidf")  # "tfidf" (offline) or "gemini"

# Safety: the operating threshold is chosen on validation data to hit this recall.
TARGET_RECALL = float(os.getenv("PW_TARGET_RECALL", "0.90"))

MODEL_PATH = ARTIFACTS / "severity_model.joblib"
METRICS_PATH = ARTIFACTS / "model_metrics.json"
INDEX_PATH = ARTIFACTS / "label_index.joblib"
