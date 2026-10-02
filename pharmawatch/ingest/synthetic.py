"""Synthetic pharmacovigilance world (fully offline, fully FICTIONAL).

Every drug name and label here is invented. The point is a controlled world
where we KNOW the ground truth, so we can measure whether the system works:

* seriousness is generated from a known logistic process (so the ML model has
  real signal to learn, plus noise so it is not trivial)
* one drug (Zentravex) gets a PLANTED emerging safety signal: from a start
  date onward, "interstitial lung disease" (NOT in its label) appears in a
  growing share of reports and report volume spikes. A working system must
  (a) flag the spike, (b) flag the drug-event pair via PRR, and (c) tell the
  reviewer the event is NOT a known/labeled risk.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

SIGNAL_DRUG = "Zentravex"
SIGNAL_TERM = "interstitial lung disease"
START = date(2024, 10, 7)          # a Monday
WEEKS = 104
SIGNAL_START_WEEK = 82             # emerging signal begins ~5 months before the end

# MedDRA-style preferred terms -> contribution to seriousness log-odds
REACTION_SEVERITY: dict[str, float] = {
    # mild
    "nausea": 0.0, "headache": 0.0, "fatigue": 0.0, "insomnia": 0.0, "cough": 0.0,
    "constipation": 0.0, "injection site reaction": 0.0, "dizziness": 0.2, "rash": 0.3,
    "diarrhoea": 0.1, "arthralgia": 0.1, "pruritus": 0.1, "myalgia": 0.3, "vomiting": 0.4,
    "palpitations": 0.5, "dry mouth": 0.0,
    # moderate
    "hypotension": 1.0, "dyspnoea": 1.2, "hypoglycaemia": 1.2, "tachycardia": 1.0,
    "syncope": 1.5, "hepatic enzyme increased": 1.0, "oedema peripheral": 0.6,
    "hypocalcaemia": 1.4, "hyperkalaemia": 1.3, "anaemia": 1.2, "neutropenia": 1.8,
    # severe
    "anaphylactic reaction": 3.0, "angioedema": 2.2, "pancreatitis": 2.5,
    "acute kidney injury": 2.6, "stevens-johnson syndrome": 3.2,
    "gastrointestinal haemorrhage": 2.8, "myocardial infarction": 3.0,
    "pulmonary embolism": 3.0, "rhabdomyolysis": 2.8, "hepatic failure": 3.3,
    "osteonecrosis of jaw": 2.0, "seizure": 2.4, "suicidal ideation": 2.5,
    "serious infection": 2.2, "interstitial lung disease": 2.6, "death": 7.0,
}
BACKGROUND = list(REACTION_SEVERITY)

# name -> (class, popularity, baseline risk, mean age, labeled reactions, boxed warning terms)
DRUGS: dict[str, dict] = {
    "Corvalin":   dict(cls="lipid-lowering agent (fictional)", pop=1.4, risk=0.0, age=62,
                       labeled=["myalgia", "hepatic enzyme increased", "rhabdomyolysis", "headache", "nausea", "arthralgia"], boxed=[]),
    "Glucentra":  dict(cls="incretin mimetic (fictional)", pop=1.6, risk=0.1, age=55,
                       labeled=["nausea", "vomiting", "diarrhoea", "constipation", "hypoglycaemia", "pancreatitis"], boxed=[]),
    "Velmora":    dict(cls="oral anticoagulant (fictional)", pop=1.1, risk=0.4, age=70,
                       labeled=["gastrointestinal haemorrhage", "anaemia", "dizziness", "rash", "pruritus"], boxed=["gastrointestinal haemorrhage"]),
    "Zentravex":  dict(cls="immunomodulatory biologic (fictional)", pop=0.9, risk=0.3, age=48,
                       labeled=["injection site reaction", "serious infection", "headache", "rash", "neutropenia"], boxed=["serious infection"]),
    "Ostevia":    dict(cls="bone resorption inhibitor (fictional)", pop=0.8, risk=0.1, age=68,
                       labeled=["hypocalcaemia", "osteonecrosis of jaw", "arthralgia", "fatigue", "myalgia"], boxed=["hypocalcaemia"]),
    "Neurisol":   dict(cls="selective serotonin reuptake inhibitor (fictional)", pop=1.2, risk=0.0, age=42,
                       labeled=["nausea", "insomnia", "dizziness", "headache", "dry mouth", "suicidal ideation"], boxed=["suicidal ideation"]),
    "Pressavil":  dict(cls="ACE inhibitor (fictional)", pop=1.3, risk=0.0, age=64,
                       labeled=["cough", "dizziness", "hypotension", "angioedema", "hyperkalaemia"], boxed=[]),
    "Respiqan":   dict(cls="leukotriene receptor antagonist (fictional)", pop=0.7, risk=-0.2, age=35,
                       labeled=["headache", "insomnia", "pruritus", "suicidal ideation"], boxed=["suicidal ideation"]),
    "Lipravex":   dict(cls="PCSK9 inhibitor (fictional)", pop=0.6, risk=-0.1, age=60,
                       labeled=["injection site reaction", "myalgia", "cough", "fatigue"], boxed=[]),
    "Thyrovance": dict(cls="thyroid hormone (fictional)", pop=0.9, risk=-0.2, age=50,
                       labeled=["tachycardia", "palpitations", "insomnia", "fatigue"], boxed=[]),
}

REPORTERS = ["consumer", "physician", "pharmacist", "other_hp", "lawyer"]
REPORTER_P = [0.45, 0.25, 0.10, 0.18, 0.02]
REPORTER_EFFECT = {"consumer": -0.3, "physician": 0.4, "pharmacist": 0.0, "other_hp": 0.2, "lawyer": 1.0}
COUNTRIES = ["US", "GB", "DE", "IN", "JP", "CA", "FR"]
COUNTRY_P = [0.6, 0.08, 0.08, 0.08, 0.06, 0.05, 0.05]


def _label_text(name: str, d: dict) -> dict[str, str]:
    hdr = f"FICTIONAL LABEL for {name} ({d['cls']}). Synthetic text generated for the PharmaWatch demo; not a real product."
    common = [r for r in d["labeled"] if REACTION_SEVERITY[r] < 1.0]
    serious = [r for r in d["labeled"] if REACTION_SEVERITY[r] >= 1.0]
    sections = {
        "adverse_reactions": (
            f"{hdr}\n6 ADVERSE REACTIONS. The most common adverse reactions reported in clinical trials of {name} "
            f"(incidence of 5% or greater and more frequent than placebo) were: {', '.join(common) or 'none above threshold'}. "
            f"Less common but clinically important adverse reactions included {', '.join(serious) or 'none'}. "
            f"Postmarketing experience: because these reactions are reported voluntarily from a population of uncertain size, "
            f"it is not always possible to estimate their frequency. Reports have included {', '.join(d['labeled'])}."
        ),
        "warnings_and_cautions": (
            f"{hdr}\n5 WARNINGS AND PRECAUTIONS. "
            + " ".join(
                f"5.{i+1} {r.title()}: cases of {r} have been reported in patients treated with {name}. "
                f"Monitor patients for signs and symptoms of {r} and discontinue {name} if {r} is suspected."
                for i, r in enumerate(serious)
            )
            if serious else f"{hdr}\n5 WARNINGS AND PRECAUTIONS. No specific warnings beyond class labeling."
        ),
    }
    if d["boxed"]:
        sections["boxed_warning"] = (
            f"{hdr}\nWARNING: {' AND '.join(t.upper() for t in d['boxed'])}. "
            f"Patients treated with {name} are at increased risk of {', '.join(d['boxed'])}. "
            f"Evaluate the benefits and risks before initiating therapy."
        )
    return sections


def generate(n_reports: int = 24000, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = list(DRUGS)
    drugs = pd.DataFrame({"drug_id": range(1, len(names) + 1), "name": names,
                          "drug_class": [DRUGS[n]["cls"] for n in names]})
    did = dict(zip(names, drugs.drug_id))

    labels = [
        {"drug_id": did[n], "section": s, "body": body}
        for n in names for s, body in _label_text(n, DRUGS[n]).items()
    ]

    # weekly volume; signal drug ramps up after the signal start
    pop = np.array([DRUGS[n]["pop"] for n in names])
    weekly_base = n_reports / WEEKS
    reports, rdrugs, rreacts = [], [], []
    rid = 0
    for w in range(WEEKS):
        wk_start = START + timedelta(weeks=w)
        p = pop.copy()
        ramp = 0.0
        if w >= SIGNAL_START_WEEK:
            ramp = min(1.0, (w - SIGNAL_START_WEEK + 1) / 6)       # ramps over ~6 weeks
            p[names.index(SIGNAL_DRUG)] *= 1 + 1.8 * ramp
        p = p / p.sum()
        n_w = rng.poisson(weekly_base * (1 + 0.1 * np.sin(w / 8)))
        for _ in range(n_w):
            rid += 1
            name = names[rng.choice(len(names), p=p)]
            d = DRUGS[name]
            # reactions
            k = rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.2, 0.1])
            terms: set[str] = set()
            for _ in range(k):
                if rng.random() < 0.75:
                    terms.add(d["labeled"][rng.integers(len(d["labeled"]))])
                else:
                    bg = BACKGROUND[rng.integers(len(BACKGROUND))]
                    if REACTION_SEVERITY[bg] >= 2.0 and rng.random() < 0.8:   # severe off-label events are rare
                        bg = "fatigue"
                    if bg != SIGNAL_TERM:
                        terms.add(bg)
            if not terms:
                terms.add(d["labeled"][0])
            if name == SIGNAL_DRUG and ramp > 0 and rng.random() < 0.35 * ramp:
                terms.add(SIGNAL_TERM)
                if rng.random() < 0.6:
                    terms.add("dyspnoea")
                if rng.random() < 0.4:
                    terms.add("cough")
            # patient + context
            age = float(np.clip(rng.normal(d["age"], 15), 18, 95)) if rng.random() > 0.08 else np.nan
            sex = rng.choice(["F", "M", "U"], p=[0.55, 0.41, 0.04])
            weight = float(np.clip(rng.normal(78 if sex == "M" else 68, 14), 40, 160)) if rng.random() > 0.15 else np.nan
            reporter = REPORTERS[rng.choice(5, p=REPORTER_P)]
            n_con = rng.choice([0, 1, 2, 3], p=[0.45, 0.3, 0.15, 0.1])
            cons = [c for c in rng.choice(names, size=n_con, replace=False) if c != name]
            # seriousness: known logistic process + noise
            sev = sorted((REACTION_SEVERITY[t] for t in terms), reverse=True)
            logit = (-2.3 + sev[0] + 0.3 * sum(sev[1:]) + d["risk"]
                     + (0.03 * (age - 55) if not np.isnan(age) else 0.2)
                     + 0.25 * len(cons) + REPORTER_EFFECT[reporter] + rng.normal(0, 0.8))
            serious = int(rng.random() < 1 / (1 + np.exp(-logit)))
            day = wk_start + timedelta(days=int(rng.integers(7)))
            rep_id = f"SYN{rid:07d}"
            sexw = {"F": "female", "M": "male", "U": "patient"}[sex]
            agew = f"{int(age)}-year-old " if not np.isnan(age) else ""
            narrative = (f"{reporter.replace('_', ' ').title()} reports a {agew}{sexw} taking {name}"
                         + (f" (with {', '.join(cons)})" if cons else "")
                         + f" who experienced {', '.join(sorted(terms))}.")
            reports.append(dict(report_id=rep_id, received_date=day, serious=serious,
                                patient_age=None if np.isnan(age) else round(age, 1), patient_sex=sex,
                                patient_weight=None if np.isnan(weight) else round(weight, 1),
                                reporter_qualification=reporter,
                                country=COUNTRIES[rng.choice(len(COUNTRIES), p=COUNTRY_P)],
                                narrative=narrative))
            rdrugs.append(dict(report_id=rep_id, drug_id=did[name], role="suspect"))
            rdrugs.extend(dict(report_id=rep_id, drug_id=did[c], role="concomitant") for c in cons)
            rreacts.extend(dict(report_id=rep_id, reaction_term=t) for t in sorted(terms))

    return {
        "drugs": drugs,
        "reports": pd.DataFrame(reports),
        "report_drugs": pd.DataFrame(rdrugs),
        "report_reactions": pd.DataFrame(rreacts),
        "label_sections": pd.DataFrame(labels),
    }
