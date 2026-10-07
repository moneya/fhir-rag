"""Is the difference between the two models statistically real?

The pair-weighted macro recall differs by +0.0008 — a number small enough that
calling it a "win" without a test would be dishonest. These are PAIRED
measurements (same 92 patient/query pairs, two models), so the right test is a
paired one on the per-pair differences.

No scipy in this project, so: an exact-ish permutation test by sign-flipping,
which needs no distributional assumption and is a few lines of stdlib.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/whitefang/lab/evalkit/src")

from evalkit.env import load_env

load_env()

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.retrieve import retrieve

K = 10
LOCAL_MODEL = "embeddinggemma:300m"


def per_pair_recall(index: VectorIndex, embedder) -> dict[tuple[str, str], float]:
    by_patient: dict[str, list] = {}
    for chunk in index.chunks:
        by_patient.setdefault(chunk.patient_id, []).append(chunk)

    out: dict[tuple[str, str], float] = {}
    for pid, chunks in sorted(by_patient.items()):
        for query in build_queries(pid):
            relevant = {c.chunk_id for c in chunks if query.relevant(c)}
            if not relevant:
                continue
            found = retrieve(
                index, embedder.embed_query(query.question), k=K, patient_id=pid,
                resource_types=query.resource_types, shape=query.shape,
            )
            hit = len([h for h in found if h.chunk.chunk_id in relevant])
            out[(pid, query.id)] = hit / len(relevant)
    return out


def main() -> int:
    base_index = VectorIndex.load("data/index_30.json")
    nemotron = get_embedder("nvidia", cache=EmbeddingCache(base_index.model))
    a = per_pair_recall(base_index, nemotron)

    local = get_embedder(
        "ollama", model=LOCAL_MODEL, timeout=300.0,
        cache=EmbeddingCache(LOCAL_MODEL), concurrency=4,
    )
    texts = [c.text for c in base_index.chunks]
    vectors = local.embed_passages(texts).vectors  # fully cached by now
    local_index = VectorIndex(model=LOCAL_MODEL, asymmetric=False)
    local_index.add(list(base_index.chunks), vectors)
    b = per_pair_recall(local_index, local)

    keys = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in keys]
    nonzero = [d for d in diffs if abs(d) > 1e-12]

    print(f"paired on {len(keys)} (patient, query) pairs")
    print(f"mean difference (gemma - nemotron): {statistics.fmean(diffs):+.4f}")
    print(f"pairs where they differ at all:     {len(nonzero)} of {len(keys)}")
    print(f"  gemma better: {sum(1 for d in nonzero if d > 0)}")
    print(f"  gemma worse:  {sum(1 for d in nonzero if d < 0)}")

    if not nonzero:
        print("\nThe models are identical on every pair. No test needed.")
        return 0

    observed = statistics.fmean(diffs)
    rng = random.Random(0)
    trials = 20000
    extreme = 0
    for _ in range(trials):
        flipped = statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(flipped) >= abs(observed):
            extreme += 1
    p = (extreme + 1) / (trials + 1)

    print(f"\npermutation test ({trials} sign-flips, seed 0)")
    print(f"  p = {p:.3f}")
    print()
    if p < 0.05:
        print(f"Difference is unlikely to be noise (p={p:.3f}).")
    else:
        print(
            f"NOT significant (p={p:.3f}). On this corpus the two models are "
            f"indistinguishable on recall@{K}; the +{observed:.4f} macro gap is noise.\n"
            f"The defensible claim is PARITY, not a win."
        )

    Path("results/comparison_significance.json").write_text(
        json.dumps(
            {
                "pairs": len(keys),
                "mean_difference": round(statistics.fmean(diffs), 4),
                "pairs_differing": len(nonzero),
                "gemma_better": sum(1 for d in nonzero if d > 0),
                "gemma_worse": sum(1 for d in nonzero if d < 0),
                "permutation_p": round(p, 4),
                "trials": trials,
                "conclusion": "parity" if p >= 0.05 else "difference",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
