"""PharmaWatch dashboard (Streamlit).   streamlit run dashboard/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from pharmawatch import config, db, rag, signals
from pharmawatch.agent import get_agent
from pharmawatch.schemas import Complaint

st.set_page_config(page_title="PharmaWatch", layout="wide")
st.title("PharmaWatch: drug safety signal detection & triage")
st.caption(f"Data: {config.DATA_MODE} · DB: {db.get_engine().dialect.name} · "
           f"LLM: {'gemini' if config.GEMINI_API_KEY and config.LLM_BACKEND == 'gemini' else 'offline policy'}"
           + (" · all drug names in synthetic mode are FICTIONAL" if config.DATA_MODE == "synthetic" else ""))


@st.cache_data(ttl=600)
def load():
    weekly = signals.weekly_counts()
    sp = signals.spikes()
    prr = signals.prr_table()
    metrics = json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else None
    return weekly, sp, prr, metrics


weekly, sp, prr, metrics = load()
drugs = sorted(weekly["drug"].unique())

tab_overview, tab_signals, tab_triage, tab_model = st.tabs(["Overview", "Signals", "Triage a complaint", "Model"])

with tab_overview:
    tot = weekly.groupby("week")[["reports", "serious"]].sum().reset_index()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports", f"{int(tot.reports.sum()):,}")
    c2.metric("Serious share", f"{tot.serious.sum() / tot.reports.sum():.0%}")
    c3.metric("PRR signals", int(prr["signal"].sum()))
    c4.metric("Volume spikes", int(sp["spike"].sum()))
    st.plotly_chart(px.line(weekly, x="week", y="reports", color="drug", title="Weekly report volume by suspect drug"),
                    use_container_width=True)
    by_drug = weekly.groupby("drug")[["reports", "serious"]].sum().reset_index()
    by_drug["serious_share"] = by_drug.serious / by_drug.reports
    st.plotly_chart(px.bar(by_drug.sort_values("serious_share"), x="serious_share", y="drug", orientation="h",
                           title="Share of serious reports by drug"), use_container_width=True)

with tab_signals:
    drug = st.selectbox("Drug", drugs, index=drugs.index("Zentravex") if "Zentravex" in drugs else 0)
    d = sp[sp.drug == drug]
    fig = px.line(d, x="week", y=["reports", "baseline"], title=f"{drug}: weekly reports vs baseline (guard-banded)")
    for w in d[d.spike].week:
        fig.add_vline(x=w, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Disproportionality (PRR) signals")
    sig = prr[prr.signal].copy()
    only_new = st.checkbox("Only reactions NOT in the drug's label (potential new signals)", value=True)
    sig["label_status"] = [rag.check_known_risk(r.drug, [r.reaction_term])["results"][0]["status"] for r in sig.itertuples()]
    if only_new:
        sig = sig[sig.label_status == "not_in_label"]
    st.dataframe(sig[["drug", "reaction_term", "a", "prr", "prr_ci_low", "chi2", "label_status"]]
                 .rename(columns={"a": "cases"}).round(2), use_container_width=True)

with tab_triage:
    with st.form("triage"):
        c1, c2 = st.columns(2)
        drug_in = c1.text_input("Suspect drug (then concomitant, comma-separated)", "Zentravex")
        reactions_in = c2.text_input("Reactions (comma-separated; leave blank to extract from narrative)", "")
        narrative = st.text_area("Narrative", "Physician reports a 58-year-old woman on Zentravex admitted with dry "
                                               "cough and shortness of breath; CT shows interstitial lung disease.")
        c3, c4, c5 = st.columns(3)
        age = c3.number_input("Age", 0, 120, 58)
        sex = c4.selectbox("Sex", ["F", "M", "U"])
        reporter = c5.selectbox("Reporter", ["physician", "consumer", "pharmacist", "other_hp", "lawyer"])
        go = st.form_submit_button("Triage")
    if go:
        c = Complaint(drugs=[x.strip() for x in drug_in.split(",") if x.strip()],
                      reactions=[x.strip().lower() for x in reactions_in.split(",") if x.strip()],
                      narrative=narrative, age=age, sex=sex, reporter=reporter)
        try:
            rep = get_agent().triage(c)
        except ValueError as e:
            st.error(str(e))
        else:
            color = {"URGENT": "red", "HIGH": "orange", "ROUTINE": "green"}[rep.priority]
            st.markdown(f"## :{color}[{rep.priority}]  ·  P(serious) = {rep.severity_probability:.2f} ({rep.risk_band})")
            st.write(rep.rationale)
            st.dataframe(pd.DataFrame([r.model_dump() for r in rep.reactions]), use_container_width=True)
            st.markdown("**Recommended actions**\n" + "\n".join(f"- {a}" for a in rep.recommended_actions))
            with st.expander("Label citations"):
                for ci in rep.citations:
                    st.markdown(f"`{ci['chunk_id']}` ({ci['section']}): {ci['quote']}")
            if rep.guardrail_notes:
                st.warning("Guardrails: " + " | ".join(rep.guardrail_notes))
            st.caption(f"backend={rep.backend} · tools={rep.tool_calls} · {rep.latency_ms} ms")

with tab_model:
    if metrics:
        best = metrics["selected_model"]
        st.write(f"Selected model: **{best}** (chosen on validation PR-AUC). Operating threshold "
                 f"{metrics['threshold']} targets recall ≥ {metrics['target_recall']}.")
        rows = [{"model": m, "split": s, **v[s]} for m, v in metrics["results"].items() for s in ("validation", "test")]
        st.dataframe(pd.DataFrame(rows)[["model", "split", "roc_auc", "pr_auc", "recall", "precision",
                                         "false_negatives", "brier"]], use_container_width=True)
        st.plotly_chart(px.bar(pd.DataFrame(metrics["top_features"]), x="mean_abs_contribution", y="feature",
                               orientation="h", title="Top features (mean |contribution|)"), use_container_width=True)
