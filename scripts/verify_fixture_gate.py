"""Does the fixture gate actually catch the regression it exists to catch?

A fixture scoring 1.000 everywhere is suspicious: a gate with no headroom cannot
show degradation. So this disables the shape-aware policy — the exact regression
the gate guards — and checks the numbers move.

If they do not, the fixture is decorative and should not be committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhir_rag.diversify import cap_per_resource_type
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries

index = VectorIndex.load(ROOT / "tests" / "fixtures" / "index_fixture.json")
qvecs = json.loads((ROOT / "tests" / "fixtures" / "query_vectors.json").read_text())
pid = index.chunks[0].patient_id
K = 10

print(f"{len(index)} chunks, patient {pid[:8]}\n")
print(f"{'query':24s} {'shape-aware':>11s} {'flat':>8s} {'cap2':>8s}")

rows = []
for query in build_queries(pid):
    relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
    if not relevant or query.id not in qvecs:
        continue
    vector = qvecs[query.id]

    # Policy 1: shape-aware (what the gate protects).
    from fhir_rag.retrieve import retrieve
    shaped = retrieve(
        index, vector, k=K, patient_id=pid,
        resource_types=query.resource_types, shape=query.shape,
    )

    # Policy 2: flat top-k, no shape awareness.
    flat = index.search(
        vector, k=K, patient_id=pid, resource_types=query.resource_types
    )

    # Policy 3: global cap, the heuristic that rescues sparse and ruins dense.
    wide = index.search(
        vector, k=K * 6, patient_id=pid, resource_types=query.resource_types
    )
    capped = cap_per_resource_type(wide, cap=2, k=K)

    def recall(hits):
        found = [h for h in hits if h.chunk.chunk_id in relevant]
        return len(found) / len(relevant)

    r = (recall(shaped), recall(flat), recall(capped))
    rows.append((query.id, *r))
    print(f"{query.id:24s} {r[0]:11.3f} {r[1]:8.3f} {r[2]:8.3f}")

for i, label in ((1, "shape-aware"), (2, "flat"), (3, "cap2")):
    macro = sum(row[i] for row in rows) / len(rows)
    print(f"\nmacro recall@{K} ({label:11s}) {macro:.4f}")

shaped_macro = sum(row[1] for row in rows) / len(rows)
flat_macro = sum(row[2] for row in rows) / len(rows)
cap_macro = sum(row[3] for row in rows) / len(rows)
print()
if shaped_macro > max(flat_macro, cap_macro) + 1e-9:
    print("GOOD: the fixture distinguishes shape-aware from both alternatives, "
          "so a policy regression fails the gate.")
else:
    print("PROBLEM: the fixture does NOT separate the policies. A gate built on "
          "it would pass a real regression — do not commit it as-is.")
