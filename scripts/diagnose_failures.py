"""Diagnose why a query fails: is the ground truth absent, or just out-ranked?

A zero recall score has two very different causes. Either the relevant chunk is
not in the corpus (the benchmark is wrong) or it is present and something else
beat it (the retriever is wrong). Only the second is a real finding, so this
script checks which one applies before anything gets written down as a result.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "/Users/whitefang/lab/evalkit/src")

from evalkit.env import load_env

load_env()

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import SNOMED_HYPERTENSION, SNOMED_PREDIABETES

idx = VectorIndex.load("data/index_30.json")
print(f"index: {len(idx)} chunks, {idx.dims} dims, model={idx.model}")

cases = [
    ("hypertension", SNOMED_HYPERTENSION, "Does this patient have high blood pressure?"),
    ("prediabetes", SNOMED_PREDIABETES, "Is there any indication of impaired glucose regulation?"),
]


def truth_chunks(codes):
    return [
        c for c in idx.chunks
        if c.resource_type == "Condition" and any(e.split("|")[-1] in codes for e in c.codes)
    ]


embedder = get_embedder("nvidia", cache=EmbeddingCache(idx.model))

for label, codes, question in cases:
    found = truth_chunks(codes)
    print(f"\n{'=' * 70}\n{label}: {len(found)} ground-truth chunk(s) in corpus")
    if not found:
        print("  -> benchmark problem: nothing to retrieve")
        continue
    print("  example:", found[0].text[:95])

    pid = found[0].patient_id
    qv = embedder.embed_query(question)
    hits = idx.search(qv, k=8, patient_id=pid)
    truth_ids = {c.chunk_id for c in found if c.patient_id == pid}

    print(f"\n  top-8 for {question!r}")
    for h in hits:
        mark = "  <== GROUND TRUTH" if h.chunk.chunk_id in truth_ids else ""
        print(f"   {h.rank}. {h.score:.4f} [{h.chunk.resource_type:12s}] {h.chunk.text[:58]}{mark}")

    # Where does the ground truth actually rank?
    all_hits = idx.search(qv, k=len(idx), patient_id=pid)
    for h in all_hits:
        if h.chunk.chunk_id in truth_ids:
            print(f"\n  ground truth ranked {h.rank} of {len(all_hits)} (score {h.score:.4f})")
            break

    # Does restricting to Conditions recover it? If so, the content is findable
    # and the problem is cross-resource ranking, not the embedding itself.
    cond = idx.search(qv, k=3, patient_id=pid, resource_types=["Condition"])
    print("  within Conditions only:")
    for h in cond:
        mark = "  <== GROUND TRUTH" if h.chunk.chunk_id in truth_ids else ""
        print(f"   {h.rank}. {h.score:.4f} {h.chunk.text[:58]}{mark}")
