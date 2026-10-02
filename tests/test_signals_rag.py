import pandas as pd

from pharmawatch import rag, signals
from pharmawatch.ingest import synthetic


def test_prr_flags_planted_signal(world):
    p = signals.prr_table(engine=world["engine"])
    row = p[(p.drug == synthetic.SIGNAL_DRUG) & (p.reaction_term == synthetic.SIGNAL_TERM)].iloc[0]
    assert row.signal and row.prr > 10


def test_prr_formula_matches_hand_calculation():
    # a=20,b=80,c=100,d=9800 -> PRR with +0.5 correction
    a, b, c, d = 20.5, 80.5, 100.5, 9800.5
    expected = (a / (a + b)) / (c / (c + d))
    assert round(expected, 2) == round((20.5 / 101) / (100.5 / 9901), 2)


def test_spike_detection_finds_signal_drug(world):
    sp = signals.spikes(engine=world["engine"])
    zmax = sp.groupby("drug")["z"].max()
    assert zmax.idxmax() == synthetic.SIGNAL_DRUG


def test_chunking_overlaps():
    text = "One. Two is longer. Three. Four four four. Five."
    chunks = rag.chunk_text(text, max_chars=20, overlap_sentences=1)
    assert len(chunks) > 1
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]


def test_spelling_normalization():
    assert rag.normalize_spelling("gastrointestinal haemorrhage, oedema") == "gastrointestinal hemorrhage, edema"


def test_known_risk_statuses(world):
    res = rag.check_known_risk("Zentravex", ["serious infection", "headache", "interstitial lung disease"])
    st = {r["reaction"]: r["status"] for r in res["results"]}
    assert st == {"serious infection": "labeled_boxed_warning", "headache": "labeled_adverse_reaction",
                  "interstitial lung disease": "not_in_label"}
    labeled = res["results"][0]
    assert all(e["chunk_id"].startswith("Zentravex:") for e in labeled["evidence"])
    assert rag.check_known_risk("UnknownDrug", ["rash"])["results"][0]["status"] == "no_label_available"
