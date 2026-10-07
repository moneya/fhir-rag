"""How much of the comparison is signal?

The macro numbers favour the local model, but per-query recall moved in both
directions and some queries have very few patients with ground truth. A query
backed by 2 patients cannot support a claim: one failure reads as -0.500.

So count the pairs behind every query before believing any per-query delta, and
check whether the headline win survives dropping the thinnest cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries

report = json.loads((ROOT / "data" / "model_comparison_full.json").read_text())
index = VectorIndex.load("data/index_30.json")

by_patient: dict[str, list] = {}
for chunk in index.chunks:
    by_patient.setdefault(chunk.patient_id, []).append(chunk)

counts: dict[str, int] = {}
for pid, chunks in by_patient.items():
    for query in build_queries(pid):
        if any(query.relevant(c) for c in chunks):
            counts[query.id] = counts.get(query.id, 0) + 1

base = report["baseline"]["per_query"]
cand = report["candidate"]["per_query"]

print(f"{'query':26s} {'pairs':>5s} {'nemotron':>9s} {'gemma':>7s} {'delta':>8s}")
thin = []
for qid in sorted(counts, key=lambda q: -counts[q]):
    n = counts[qid]
    b, c = base.get(qid, 0.0), cand.get(qid, 0.0)
    mark = ""
    if n <= 3:
        mark = "  <- too thin to read"
        thin.append(qid)
    print(f"{qid:26s} {n:5d} {b:9.3f} {c:7.3f} {c - b:+8.3f}{mark}")

print(f"\ntotal pairs: {sum(counts.values())}")
print(f"queries with <=3 patients: {thin}")

# Does the headline win survive removing the thin cells? Weight by pair count,
# since a macro mean over queries over-weights rare ones.
def weighted(per_query, exclude=()):
    num = den = 0
    for qid, value in per_query.items():
        if qid in exclude:
            continue
        num += value * counts.get(qid, 0)
        den += counts.get(qid, 0)
    return num / den if den else 0.0

print(f"\npair-weighted recall@10")
print(f"  all queries      nemotron {weighted(base):.4f}  gemma {weighted(cand):.4f}")
print(f"  thin removed     nemotron {weighted(base, thin):.4f}  gemma {weighted(cand, thin):.4f}")

b_all, c_all = weighted(base), weighted(cand)
b_thin, c_thin = weighted(base, thin), weighted(cand, thin)
print()
if (c_all > b_all) == (c_thin > b_thin):
    print("Direction is stable with and without the thin cells.")
else:
    print("Direction FLIPS when thin cells are removed - the result is not robust.")
