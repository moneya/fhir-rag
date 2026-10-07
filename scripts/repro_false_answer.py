"""Reproduce the single unsafe generation: answered with zero relevant evidence.

The benchmark flagged one FALSE answer — retrieval returned no relevant chunk for
the prediabetes query, and the model answered anyway, citing two real but
irrelevant records. That is the most important row in the whole run, so it is
reproduced in full rather than summarised, and kept as a script so the fix can be
re-verified.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "/Users/whitefang/lab/evalkit/src")

from evalkit.env import load_env

load_env()

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.generate import GroqGenerator
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.rerank import GroqReranker
from fhir_rag.retrieve import retrieve

index = VectorIndex.load("data/index_30.json")
embedder = get_embedder("nvidia", cache=EmbeddingCache(index.model))
reranker = GroqReranker(cache=EmbeddingCache("rerank-qwen3.8-27b"))
generator = GroqGenerator()

patients = sorted({c.patient_id for c in index.chunks})[:5]

for pid in patients:
    query = next(q for q in build_queries(pid) if q.id == "prediabetes")
    relevant = {
        c.chunk_id for c in index.chunks
        if c.patient_id == pid and query.relevant(c)
    }
    if not relevant:
        continue

    hits = retrieve(
        index, embedder.embed_query(query.question), k=5, patient_id=pid,
        resource_types=query.resource_types, shape=query.shape,
        reranker=reranker, question=query.question,
    )
    retrieved_relevant = [h for h in hits if h.chunk.chunk_id in relevant]
    if retrieved_relevant:
        continue  # not the failing case

    answer = generator.answer(query.question, hits)
    print("=" * 72)
    print(f"patient {pid[:8]}  question: {query.question}")
    print(f"ground truth ({len(relevant)} chunk(s)) NOT retrieved:")
    for cid in relevant:
        text = next(c.text for c in index.chunks if c.chunk_id == cid)
        print(f"   {cid}  {text[:80]}")
    print("\ncontext actually sent:")
    for i, hit in enumerate(hits, start=1):
        print(f"  [{i}] {hit.chunk.text[:76]}")
    print(f"\nanswer (abstained={answer.abstained}, grounded={answer.grounded}):")
    print(f"  {answer.text}")
    print(f"  citations {answer.citations} -> {[c for c in answer.cited_chunks]}")
    print(f"  any cited chunk relevant? {any(c in relevant for c in answer.cited_chunks)}")
    break
