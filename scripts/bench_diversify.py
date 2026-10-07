"""Measure whether diversification actually improves recall, or just moves it.

Run before/after on the same index and cached vectors so the comparison is
free and exact: same chunks, same embeddings, only the ranking policy changes.
"""

import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "/Users/whitefang/lab/evalkit/src")

from evalkit.env import load_env

load_env()

from fhir_rag.cache import EmbeddingCache
from fhir_rag.diversify import cap_per_resource_type, interleave_by_type, mmr
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries

K = 10
FETCH = 60  # over-fetch, then re-rank into K

idx = VectorIndex.load("data/index_30.json")
embedder = get_embedder("nvidia", cache=EmbeddingCache(idx.model))
by_id = {c.chunk_id: v for c, v in zip(idx.chunks, idx.vectors)}
patients = sorted({c.patient_id for c in idx.chunks})

print(f"{len(idx)} chunks, {len(patients)} patients, k={K}, fetch={FETCH}\n")

strategies = {
    "flat": lambda hits: hits[:K],
    "cap2": lambda hits: cap_per_resource_type(hits, k=K, cap=2),
    "cap3": lambda hits: cap_per_resource_type(hits, k=K, cap=3),
    "mmr0.7": lambda hits: mmr(hits, [by_id[h.chunk.chunk_id] for h in hits], k=K, lambda_=0.7),
    "interleave": lambda hits: interleave_by_type(hits, k=K),
    "shape-aware": None,  # handled inline: needs the query's declared shape
}

# question -> cached query vector, so each strategy reuses one embedding
qvecs: dict[str, list[float]] = {}
agg: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in strategies}

for pid in patients:
    for query in build_queries(pid):
        candidates = [c for c in idx.chunks if c.patient_id == pid]
        relevant = {c.chunk_id for c in candidates if query.relevant(c)}
        if not relevant:
            continue
        if query.question not in qvecs:
            qvecs[query.question] = embedder.embed_query(query.question)
        hits = idx.search(
            qvecs[query.question], k=FETCH, patient_id=pid,
            resource_types=query.resource_types,
        )
        for name, apply in strategies.items():
            if name == "shape-aware":
                ranked = (
                    cap_per_resource_type(list(hits), k=K, cap=2)
                    if query.shape == "single" else list(hits)[:K]
                )
            else:
                ranked = apply(list(hits))
            ranks = [h.rank for h in ranked if h.chunk.chunk_id in relevant]
            recall = sum(1 for r in ranks if r <= K) / len(relevant)
            agg[name][query.id].append(recall)
            agg[name]["_mrr"].append(1.0 / min(ranks) if ranks else 0.0)
            agg[name]["_hit"].append(1.0 if ranks else 0.0)

query_ids = sorted(k for k in agg["flat"] if not k.startswith("_"))
print(f"{'query':24s} " + "  ".join(f"{s:>8s}" for s in strategies))
for qid in query_ids:
    row = "  ".join(
        f"{sum(agg[s][qid]) / len(agg[s][qid]):8.3f}" for s in strategies
    )
    print(f"{qid:24s} {row}")

print()
for label, key in (("MACRO recall@10", None), ("hit@10", "_hit"), ("MRR", "_mrr")):
    cells = []
    for s in strategies:
        if key:
            values = agg[s][key]
        else:
            values = [v for qid in query_ids for v in agg[s][qid]]
        cells.append(f"{sum(values) / len(values):8.3f}")
    print(f"{label:24s} " + "  ".join(cells))

json.dump(
    {
        s: {
            "recall@10": sum(v for qid in query_ids for v in agg[s][qid])
            / sum(len(agg[s][qid]) for qid in query_ids),
            "hit@10": sum(agg[s]["_hit"]) / len(agg[s]["_hit"]),
            "mrr": sum(agg[s]["_mrr"]) / len(agg[s]["_mrr"]),
            "per_query": {qid: sum(agg[s][qid]) / len(agg[s][qid]) for qid in query_ids},
        }
        for s in strategies
    },
    open("data/bench_diversify.json", "w"),
    indent=2,
)
print("\nwrote data/bench_diversify.json")
