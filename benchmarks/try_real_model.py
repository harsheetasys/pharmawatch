from pharmawatch import model
from pharmawatch.agent import OfflineTriageAgent
from pharmawatch.schemas import Complaint

for c in [
    {"drugs": ["Ozempic"], "reactions": ["pancreatitis", "vomiting"], "age": 52, "sex": "F",
     "narrative": "on it for weight loss"},
    {"drugs": ["Mounjaro"], "reactions": ["injection site pain"], "age": 40, "sex": "F"},
]:
    r = model.score(c)
    print(f"{c['drugs'][0]:9s} {c['reactions']}  ->  p={r.probability:.2f}  band={r.risk_band}  model={r.model}")
    for f in r.top_factors[:3]:
        print(f"      {f['feature']:35s} {f['contribution']:+.2f}")

rep = OfflineTriageAgent().triage(Complaint(drugs=["Ozempic"], reactions=["pancreatitis"], age=60, sex="F"))
print(f"\nAgent triage for Ozempic + pancreatitis: {rep.priority} (p={rep.severity_probability:.2f})")