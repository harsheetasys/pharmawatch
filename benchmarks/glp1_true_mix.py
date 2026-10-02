"""Lesson 4a check: the TRUE mix of GLP-1 reports in FAERS 2015-2025 (from FDA's own counts)."""
from benchmarks.openfda_counts import total

DRUGS = ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide", "lixisenatide", "albiglutide"]
DATE = "receivedate:[20150101 TO 20251231]"

counts = {}
for d in DRUGS:
    counts[d] = total(f'{DATE} AND patient.drug.openfda.generic_name:"{d}"')
    print(f"   counted {d}")

all_reports = sum(counts.values())
print(f"\n{'molecule':13s} {'reports':>10s}  share")
for d, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{d:13s} {n:10,d}  {n / all_reports:.1%}")