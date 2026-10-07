"""Precision-at-small-k: does reranking put the right answer FIRST?

Recall@10 barely moved (0.908 -> 0.909) because shape-aware retrieval already
pulls the right chunks into the top ten. The question this answers is whether
reranking improves the ORDER within those ten, which is what a user or a
generation step actually consumes — nobody reads to rank 10.
"""

import collections
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "/Users/whitefang/lab/evalkit/src")

from evalkit.env import load_env

load_env()

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.rerank import GroqReranker
from fhir_rag.retrieve import retrieve

PATIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

index = VectorIndex.load("data/index_30.json")
embedder = get_embedder("nvidia", cache=EmbeddingCache(index.model))
reranker = GroqReranker(cache=EmbeddingCache("rerank-qwen3.8-27b"))
patients = sorted({c.patient_id for c in index.chunks})[:PATIENTS]

agg: dict[str, dict[str, list[float]]] = {
    "plain": collections.defaultdict(list),
    "rerank": collections.defaultdict(list),
}
fallbacks = 0
calls = 0

for pid in patients:
    for query in build_queries(pid):
        relevant = {
            c.chunk_id for c in index.chunks
            if c.patient_id == pid and query.relevant(c)
        }
        if not relevant:
            continue
        qv = embedder.embed_query(query.question)
        for label, extra in (
            ("plain", {}),
            ("rerank", {"reranker": reranker, "question": query.question}),
        ):
            hits = retrieve(
                index, qv, k=10, patient_id=pid,
                resource_types=query.resource_types, shape=query.shape, **extra
            )
            ranks = [h.rank for h in hits if h.chunk.chunk_id in relevant]
            for k in (1, 3, 5):
                # precision@k: of the k shown, how many were relevant
                agg[label][f"p@{k}"].append(
                    sum(1 for r in ranks if r <= k) / k
                )
            agg[label]["top1_relevant"].append(1.0 if 1 in ranks else 0.0)

print(f"{'metric':18s} {'plain':>8s} {'rerank':>8s}   delta")
for metric in ("p@1", "p@3", "p@5", "top1_relevant"):
    a = sum(agg["plain"][metric]) / len(agg["plain"][metric])
    b = sum(agg["rerank"][metric]) / len(agg["rerank"][metric])
    mark = "" if abs(b - a) < 0.005 else ("  <-- better" if b > a else "  <-- WORSE")
    print(f"{metric:18s} {a:8.3f} {b:8.3f}   {b - a:+.3f}{mark}")
