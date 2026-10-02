import numpy as np
import pandas as pd

from pharmawatch import features, model
from pharmawatch.ingest import openfda, synthetic


def test_synthetic_world_has_planted_signal(world):
    f = world["frames"]
    assert set(f) == {"drugs", "reports", "report_drugs", "report_reactions", "label_sections"}
    ild = f["report_reactions"].query("reaction_term == @synthetic.SIGNAL_TERM")
    assert len(ild) > 20
    # the planted term must NOT be in the signal drug's label
    zid = f["drugs"].set_index("name").loc[synthetic.SIGNAL_DRUG, "drug_id"]
    label = " ".join(f["label_sections"].query("drug_id == @zid").body).lower()
    assert synthetic.SIGNAL_TERM not in label


def test_window_feature_is_leakage_safe(world):
    df, _ = features.load_training_frame(world["engine"])
    for _, g in df.groupby("drug_id"):
        g = g.sort_values(["received_date", "report_id"])
        assert g["drug_prior_reports"].iloc[0] == 0                       # first report sees no history
        expected = g["serious"].shift(1).fillna(0).cumsum().values           # only strictly-earlier rows
        assert np.allclose(g["drug_prior_serious"].values, expected)


def test_time_split_is_chronological(world):
    df, _ = features.load_training_frame(world["engine"])
    tr, va, te = model.time_split(df)
    assert tr.received_date.max() <= va.received_date.min() <= te.received_date.min()


def test_model_meets_recall_target_on_test(world):
    s = world["summary"]
    test = s["results"][s["selected_model"]]["test"]
    assert test["recall"] >= 0.85            # target 0.90 on validation; allow drift on future data
    assert test["roc_auc"] > 0.70
    assert s["threshold_high"] >= s["threshold"]


def test_score_is_monotone_in_severity(world):
    mild = model.score({"drugs": ["Neurisol"], "reactions": ["nausea"], "age": 30, "sex": "F"})
    severe = model.score({"drugs": ["Velmora"], "reactions": ["gastrointestinal haemorrhage", "anaemia"],
                          "age": 82, "sex": "M", "reporter": "physician"})
    assert severe.probability > mild.probability
    assert severe.flag_for_review and not mild.flag_for_review
    assert severe.top_factors and "contribution" in severe.top_factors[0]


def test_openfda_parser_maps_fields():
    ev = {"safetyreportid": "10001", "receivedate": "20240315", "serious": "1", "occurcountry": "US",
          "primarysource": {"qualification": "1"},
          "patient": {"patientonsetage": "67", "patientonsetageunit": "801", "patientsex": "2", "patientweight": "70.5",
                      "drug": [{"medicinalproduct": "PROLIA", "drugcharacterization": "1", "openfda": {"generic_name": ["DENOSUMAB"]}},
                               {"medicinalproduct": "ASPIRIN", "drugcharacterization": "2"}],
                      "reaction": [{"reactionmeddrapt": "Hypocalcaemia"}, {"reactionmeddrapt": "Fatigue"}]}}
    report, drugs, terms = openfda.parse_event(ev)
    assert report["serious"] == 1 and report["patient_age"] == 67 and report["patient_sex"] == "F"
    assert report["reporter_qualification"] == "physician"
    assert drugs[0] == {"name": "denosumab", "role": "suspect"} and drugs[1]["role"] == "concomitant"
    assert terms == ["fatigue", "hypocalcaemia"]
    assert openfda.parse_event({"receivedate": "bad"}) is None
