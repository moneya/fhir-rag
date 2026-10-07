"""Does LLM reranking actually improve recall, or just cost latency?

Compares shape-aware retrieval with and without a Groq listwise reranker on the
same index and the same cached query embeddings, so only the ranking policy
differs. Reranker replies are cached too, making re-runs free.
"""

import json
import sys
import time
from collections import defaultdict

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

K = 10
PATIENT_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10

index = VectorIndex.load("data/index_30.json")
embedder = get_embedder("nvidia", cache=EmbeddingCache(index.model))
reranker = GroqReranker(cache=EmbeddingCache("rerank-qwen3.8-27b"))

patients = sorted({c.patient_id for c in index.chunks})[:PATIENT_LIMIT]
print(f"{len(index)} chunks, {len(patients)} patients, k={K}, model={reranker.model}\n")

qvecs: dict[str, list[float]] = {}
results: dict[str, dict[str, list[float]]] = {
    "shape-aware": defaultdict(list),
    "shape+rerank": defaultdict(list),
}
latencies: list[float] = []
fallbacks: list[str] = []

for pid in patients:
    for query in build_queries(pid):
        candidates = [c for c in index.chunks if c.patient_id == pid]
        relevant = {c.chunk_id for c in candidates if query.relevant(c)}
        if not relevant:
            continue
        if query.question not in qvecs:
            qvecs[query.question] = embedder.embed_query(query.question)
        qv = qvecs[query.question]

        plain = retrieve(
            index, qv, k=K, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
        )

        started = time.time()
        reranked = retrieve(
            index, qv, k=K, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
            reranker=reranker, question=query.question,
        )
        latencies.append(time.time() - started)

        for label, hits in (("shape-aware", plain), ("shape+rerank", reranked)):
            ranks = [h.rank for h in hits if h.chunk.chunk_id in relevant]
            results[label][query.id].append(len([r for r in ranks if r <= K]) / len(relevant))
            results[label]["_mrr"].append(1.0 / min(ranks) if ranks else 0.0)
            results[label]["_hit"].append(1.0 if ranks else 0.0)

query_ids = sorted(q for q in results["shape-aware"] if not q.startswith("_"))
print(f"{'query':24s} {'shape-aware':>12s} {'+rerank':>10s}   delta")
for qid in query_ids:
    a = sum(results["shape-aware"][qid]) / len(results["shape-aware"][qid])
    b = sum(results["shape+rerank"][qid]) / len(results["shape+rerank"][qid])
    arrow = "  " if abs(b - a) < 0.005 else ("UP" if b > a else "DOWN")
    print(f"{qid:24s} {a:12.3f} {b:10.3f}   {b - a:+.3f} {arrow}")

print()
for label in ("shape-aware", "shape+rerank"):
    flat = [v for qid in query_ids for v in results[label][qid]]
    print(
        f"{label:14s} recall@{K}={sum(flat) / len(flat):.3f}  "
        f"hit@{K}={sum(results[label]['_hit']) / len(results[label]['_hit']):.3f}  "
        f"MRR={sum(results[label]['_mrr']) / len(results[label]['_mrr']):.3f}"
    )

if latencies:
    ordered = sorted(latencies)
    print(
        f"\nrerank latency: mean {sum(latencies) / len(latencies):.2f}s  "
        f"p50 {ordered[len(ordered) // 2]:.2f}s  max {ordered[-1]:.2f}s  "
        f"({len(latencies)} calls)"
    )

json.dump(
    {
        label: {
            "recall@10": sum(v for qid in query_ids for v in results[label][qid])
            / sum(len(results[label][qid]) for qid in query_ids),
            "hit@10": sum(results[label]["_hit"]) / len(results[label]["_hit"]),
            "mrr": sum(results[label]["_mrr"]) / len(results[label]["_mrr"]),
            "per_query": {
                qid: sum(results[label][qid]) / len(results[label][qid])
                for qid in query_ids
            },
        }
        for label in results
    },
    open("data/bench_rerank.json", "w"),
    indent=2,
)
print("\nwrote data/bench_rerank.json")
