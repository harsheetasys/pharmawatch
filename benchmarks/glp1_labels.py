"""Lesson 5d: load the REAL FDA drug labels for GLP-1 drugs into PharmaWatch's RAG index.

After this, the agent's check_known_risk tool can answer "is pancreatitis a known risk of
Ozempic?" from the actual label text (with quotes), not from the demo's fictional labels.
Re-run this after `python -m scripts.build_all`, because build_all resets the database.
"""
import pandas as pd
from sqlalchemy import text

from benchmarks.openfda_counts import API_KEY, SESSION
from pharmawatch import db, rag

LABEL_URL = "https://api.fda.gov/drug/label.json"
MOLECULES = ["semaglutide", "tirzepatide", "dulaglutide", "liraglutide", "exenatide", "lixisenatide"]
SECTIONS = ["boxed_warning", "warnings_and_cautions", "adverse_reactions", "contraindications"]


def fetch_label(molecule: str) -> dict[str, str]:
    params = {"search": f'openfda.generic_name:"{molecule}"', "limit": 1}
    if API_KEY:
        params["api_key"] = API_KEY
    r = SESSION.get(LABEL_URL, params=params, timeout=(10, 60))
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    label = r.json()["results"][0]
    brand = ", ".join((label.get("openfda") or {}).get("brand_name", [])) or "?"
    print(f"   {molecule:13s} label found (brand: {brand})")
    return {s: "\n".join(label[s]) for s in SECTIONS if label.get(s)}


if __name__ == "__main__":
    engine = db.get_engine()
    with engine.begin() as conn:                                   # remove old copies -> safe to re-run
        for m in MOLECULES:
            conn.execute(text("DELETE FROM label_sections WHERE drug_id IN (SELECT drug_id FROM drugs WHERE name = :m)"), {"m": m})
            conn.execute(text("DELETE FROM drugs WHERE name = :m AND drug_id NOT IN (SELECT drug_id FROM report_drugs)"), {"m": m})
    next_id = int(db.query("SELECT COALESCE(MAX(drug_id), 0) AS m FROM drugs")["m"].iloc[0]) + 1
    existing = set(db.query("SELECT name FROM drugs")["name"])

    drugs, sections = [], []
    for m in MOLECULES:
        label = fetch_label(m)
        if not label:
            print(f"   {m:13s} no label found")
            continue
        if m not in existing:
            drugs.append({"drug_id": next_id, "name": m, "drug_class": "GLP-1 receptor agonist"})
            drug_id = next_id
            next_id += 1
        else:
            drug_id = int(db.query("SELECT drug_id FROM drugs WHERE name = :m", {"m": m})["drug_id"].iloc[0])
        sections += [{"drug_id": drug_id, "section": s, "body": body} for s, body in label.items()]

    db.load_frames(engine, {"drugs": pd.DataFrame(drugs), "label_sections": pd.DataFrame(sections)})
    n = rag.build_index(engine)
    print(f"\nAdded {len(sections)} label sections for {len(drugs)} GLP-1 drugs | RAG index now has {n} chunks")

    for drug, reactions in [("semaglutide", ["pancreatitis", "thyroid cancer", "hair loss"]),
                            ("tirzepatide", ["injection site reaction", "suicidal ideation"])]:
        print(f"\n{drug}:")
        for r in rag.check_known_risk(drug, reactions)["results"]:
            quote = r["evidence"][0]["quote"][:110] if r["status"] != "not_in_label" and r["evidence"] else ""
            print(f"   {r['reaction']:24s} {r['status']:26s} {quote}")