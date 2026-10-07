"""End-to-end RAG evaluation: retrieve, generate, then audit the answer.

Generation quality is usually judged by an LLM grader, which this project avoids
— a grader you cannot trust cannot gate anything. Instead every measure here is
mechanical and derived from the same coded ground truth as retrieval:

  citation precision   fraction of cited records that are genuinely relevant
  citation recall      fraction of relevant records the answer cited
  abstention           did it refuse when retrieval found nothing relevant
  unsourced answers    did it answer without the designated ground-truth chunk

That last count is flagged for review rather than scored as an error. Inspecting
one showed the model answering "is there impaired glucose regulation?" from two
HbA1c results of 5.82% and 6.03% — the prediabetic range — instead of from the
patient's Prediabetes diagnosis, which retrieval had missed. The conclusion was
clinically correct; only the provenance differed from the code-derived rule.
Calling that a false answer would be the metric lying, not the model.

A control arm feeds deliberately IRRELEVANT context, where the only correct
behaviour is abstention. Without that arm, a model that always answers scores
well on everything.
"""

import collections
import json
import sys
import time

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

PATIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
K = 5

index = VectorIndex.load("data/index_30.json")
embedder = get_embedder("nvidia", cache=EmbeddingCache(index.model))
reranker = GroqReranker(cache=EmbeddingCache("rerank-qwen3.8-27b"))
generator = GroqGenerator()
patients = sorted({c.patient_id for c in index.chunks})[:PATIENTS]

print(f"{len(index)} chunks | {len(patients)} patients | k={K} | {generator.model}\n")


def ask_with_retry(question, hits, attempts: int = 4):
    """Free-tier rate limits are transient; dropping rows silently is not ok.

    Without this, a third of the sample vanished and the control arm recorded
    zero abstentions purely because every call had failed — which read as a
    safety finding when it was only a 429.
    """
    answer = generator.answer(question, hits)
    delay = 2.0
    for _ in range(attempts - 1):
        if not answer.error:
            return answer
        time.sleep(delay)
        delay *= 2
        answer = generator.answer(question, hits)
    return answer

rows = []
control_rows = []

for pid in patients:
    for query in build_queries(pid):
        relevant = {
            c.chunk_id for c in index.chunks
            if c.patient_id == pid and query.relevant(c)
        }
        if not relevant:
            continue

        qv = embedder.embed_query(query.question)
        hits = retrieve(
            index, qv, k=K, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
            reranker=reranker, question=query.question,
        )
        answer = ask_with_retry(query.question, hits)

        retrieved_relevant = [h for h in hits if h.chunk.chunk_id in relevant]
        cited_relevant = [c for c in answer.cited_chunks if c in relevant]

        rows.append({
            "query": query.id,
            "retrieved_relevant": len(retrieved_relevant),
            "cited": len(answer.cited_chunks),
            "cited_relevant": len(cited_relevant),
            "abstained": answer.abstained,
            "grounded": answer.grounded,
            "invalid": len(answer.invalid_citations),
            "error": answer.error,
            "latency": answer.latency_s,
            "completion_tokens": answer.completion_tokens,
            # Answered without the designated ground-truth chunk in context.
            # NOT necessarily wrong: inspection showed one such answer reasoned
            # correctly from HbA1c values instead of the Prediabetes Condition
            # chunk. Needs human review, which is why it is counted separately
            # rather than folded into an accuracy number.
            "unsourced": (
                not answer.abstained and not answer.error and not retrieved_relevant
            ),
        })

    # Control arm: ask a specific lab question against unrelated immunisation
    # records. Abstention is the only correct answer.
    other = next((p for p in patients if p != pid), None)
    if other:
        lab_query = next(q for q in build_queries(pid) if q.id == "hba1c-results")
        decoys = index.search(
            embedder.embed_query("vaccination immunization administered"),
            k=K, patient_id=other, resource_types=("Immunization",),
        )
        if decoys:
            control = ask_with_retry(lab_query.question, decoys)
            control_rows.append({
                "abstained": control.abstained,
                "grounded": control.grounded,
                "error": control.error,
                "text": control.text[:70],
            })

ok = [r for r in rows if not r["error"]]
errors = len(rows) - len(ok)
print(f"{len(rows)} answers generated ({errors} transport errors)\n")

by_query = collections.defaultdict(list)
for row in ok:
    by_query[row["query"]].append(row)

print(f"{'query':24s} {'cite prec':>9s} {'cite rec':>8s} {'abstain':>7s} {'invalid':>7s}")
for qid in sorted(by_query):
    group = by_query[qid]
    precision = [
        r["cited_relevant"] / r["cited"] for r in group if r["cited"]
    ]
    recall = [
        r["cited_relevant"] / r["retrieved_relevant"]
        for r in group if r["retrieved_relevant"]
    ]
    abstains = sum(1 for r in group if r["abstained"]) / len(group)
    invalid = sum(r["invalid"] for r in group)
    print(
        f"{qid:24s} "
        f"{(sum(precision) / len(precision) if precision else float('nan')):9.3f} "
        f"{(sum(recall) / len(recall) if recall else float('nan')):8.3f} "
        f"{abstains:7.2f} {invalid:7d}"
    )

all_precision = [r["cited_relevant"] / r["cited"] for r in ok if r["cited"]]
all_recall = [
    r["cited_relevant"] / r["retrieved_relevant"] for r in ok if r["retrieved_relevant"]
]
print()
print(f"citation precision   {sum(all_precision) / len(all_precision):.3f}")
print(f"citation recall      {sum(all_recall) / len(all_recall):.3f}")
print(f"grounded             {sum(1 for r in ok if r['grounded']) / len(ok):.3f}")
print(f"abstained            {sum(1 for r in ok if r['abstained']) / len(ok):.3f}")
print(f"INVALID citations    {sum(r['invalid'] for r in ok)}  (hallucinated record ids)")
print(f"UNSOURCED answers    {sum(1 for r in ok if r['unsourced'])}  "
      f"(answered without the ground-truth chunk; review, not automatically wrong)")
print(f"mean latency         {sum(r['latency'] for r in ok) / len(ok):.2f}s")
print(f"mean output tokens   {sum(r['completion_tokens'] for r in ok) / len(ok):.0f}")

if control_rows:
    good = sum(1 for r in control_rows if r["abstained"])
    failed = sum(1 for r in control_rows if r["error"])
    print(f"\ncontrol arm (irrelevant context, abstention is the ONLY correct answer)")
    print(f"  abstained {good}/{len(control_rows)}"
          + (f", {failed} transport error(s)" if failed else ""))
    for row in control_rows:
        if not row["abstained"] and not row["error"]:
            print(f"  LEAK: answered anyway -> {row['text']!r}")

json.dump(rows, open("data/bench_generate.json", "w"), indent=2)
print("\nwrote data/bench_generate.json")
