"""Builds a small synthetic world ONCE per test session in a temp dir (SQLite, offline)."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="pw_test_")
os.environ["PW_ARTIFACTS"] = _TMP
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ["PW_LLM"] = "offline"
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def world():
    from pharmawatch import db, model, rag
    from pharmawatch.ingest import synthetic
    frames = synthetic.generate(12000, seed=11)
    engine = db.get_engine()
    db.reset_schema(engine)
    db.load_frames(engine, frames)
    summary = model.train(engine, verbose=False)
    rag.build_index(engine)
    return {"frames": frames, "summary": summary, "engine": engine}
