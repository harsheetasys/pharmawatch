from pharmawatch import rag

for drug, reaction in [("Ozempic", "hair loss"), ("Ozempic", "pancreatitis")]:
    res = rag.check_known_risk(drug, [reaction])["results"][0]
    print(f"\n{drug} + {reaction}: {res['status']}")
    for e in res["evidence"][:2]:
        print(f"   [{e['chunk_id']}] {e['quote'][:300]}")