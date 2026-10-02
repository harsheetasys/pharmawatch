"""Database schema + helpers.

The schema mirrors how FDA FAERS reports are structured: one report can name
many drugs (suspect vs concomitant) and many reactions (MedDRA preferred terms).
That many-to-many shape is exactly why the feature engineering needs real SQL
joins instead of a flat CSV.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS drugs (
        drug_id      INTEGER PRIMARY KEY,
        name         VARCHAR(200) UNIQUE NOT NULL,
        drug_class   VARCHAR(200)
    )""",
    """
    CREATE TABLE IF NOT EXISTS reports (
        report_id               VARCHAR(64) PRIMARY KEY,
        received_date           DATE NOT NULL,
        serious                 INTEGER NOT NULL,      -- target label (1 = serious)
        patient_age             REAL,
        patient_sex             VARCHAR(8),            -- M / F / U
        patient_weight          REAL,
        reporter_qualification  VARCHAR(32),           -- physician / pharmacist / other_hp / consumer / lawyer
        country                 VARCHAR(8),
        narrative               TEXT
    )""",
    """
    CREATE TABLE IF NOT EXISTS report_drugs (
        report_id  VARCHAR(64) NOT NULL REFERENCES reports(report_id),
        drug_id    INTEGER NOT NULL REFERENCES drugs(drug_id),
        role       VARCHAR(16) NOT NULL,               -- suspect / concomitant / interacting
        PRIMARY KEY (report_id, drug_id)
    )""",
    """
    CREATE TABLE IF NOT EXISTS report_reactions (
        report_id      VARCHAR(64) NOT NULL REFERENCES reports(report_id),
        reaction_term  VARCHAR(200) NOT NULL,          -- MedDRA preferred term, lower-case
        PRIMARY KEY (report_id, reaction_term)
    )""",
    """
    CREATE TABLE IF NOT EXISTS label_sections (
        drug_id   INTEGER NOT NULL REFERENCES drugs(drug_id),
        section   VARCHAR(64) NOT NULL,                -- boxed_warning / warnings_and_cautions / adverse_reactions ...
        body      TEXT NOT NULL,
        PRIMARY KEY (drug_id, section)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_reports_date ON reports(received_date)",
    "CREATE INDEX IF NOT EXISTS ix_rd_drug ON report_drugs(drug_id)",
    "CREATE INDEX IF NOT EXISTS ix_rr_term ON report_reactions(reaction_term)",
]

TABLES_IN_DROP_ORDER = ["label_sections", "report_reactions", "report_drugs", "reports", "drugs"]

_engine: Engine | None = None


def get_engine(url: str | None = None) -> Engine:
    global _engine
    if url is not None:
        return create_engine(url, future=True)
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL, future=True)
    return _engine


def is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def reset_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for t in TABLES_IN_DROP_ORDER:
            conn.execute(text(f"DROP TABLE IF EXISTS {t}" + (" CASCADE" if is_postgres(engine) else "")))
        for stmt in SCHEMA:
            conn.execute(text(stmt))


def load_frames(engine: Engine, frames: dict[str, pd.DataFrame]) -> None:
    """Bulk insert dataframes in FK-safe order."""
    order = ["drugs", "reports", "report_drugs", "report_reactions", "label_sections"]
    # Postgres: one big multi-row INSERT is fastest.
    # SQLite: older builds allow only 999 "?" placeholders per statement, so use
    # executemany (method=None), which sends one row's worth of placeholders at a time.
    method = "multi" if is_postgres(engine) else None
    with engine.begin() as conn:
        for name in order:
            if name in frames and len(frames[name]):
                frames[name].to_sql(name, conn, if_exists="append", index=False, chunksize=5000, method=method)

def query(sql: str, params: dict | None = None, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def week_expr(engine: Engine, col: str) -> str:
    """Portable 'start of ISO week' expression (Postgres vs SQLite)."""
    if is_postgres(engine):
        return f"CAST(date_trunc('week', {col}) AS DATE)"
    return f"date({col}, '-' || ((CAST(strftime('%w', {col}) AS INTEGER) + 6) % 7) || ' days')"
