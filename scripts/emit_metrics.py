"""Emit retrieval metrics as JSON for an evalkit gate.

Separating measurement from gating is deliberate. This script knows how to
compute recall; it does not know what counts as acceptable. The thresholds live
in `evals/retrieval_gate.yaml`, so tightening a bar is a config change reviewed
in a PR rather than an edit buried in Python.

Runs entirely from the on-disk embedding cache by default, so CI needs no API
key and costs nothing: query embeddings for the fixed benchmark questions are
already cached, and a cache miss is reported rather than silently hitting the
network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.retrieve import retrieve


def measure(index_path: str, *, k: int, patients: int | None) -> dict:
    index = VectorIndex.load(index_path)
    cache = EmbeddingCache(index.model)
    embedder = get_embedder("nvidia", cache=cache)

    ids = sorted({c.patient_id for c in index.chunks})
    if patients:
        ids = ids[:patients]

    per_query: dict[str, list[float]] = defaultdict(list)
    mrr: list[float] = []
    hits_at_k: list[float] = []
    precision_at_1: list[float] = []
    pairs = 0

    for pid in ids:
        for query in build_queries(pid):
            relevant = {
                c.chunk_id for c in index.chunks
                if c.patient_id == pid and query.relevant(c)
            }
            if not relevant:
                continue
            pairs += 1
            vector = embedder.embed_query(query.question)
            found = retrieve(
                index, vector, k=k, patient_id=pid,
                resource_types=query.resource_types, shape=query.shape,
            )
            ranks = [h.rank for h in found if h.chunk.chunk_id in relevant]
            per_query[query.id].append(len(ranks) / len(relevant))
            mrr.append(1.0 / min(ranks) if ranks else 0.0)
            hits_at_k.append(1.0 if ranks else 0.0)
            precision_at_1.append(1.0 if 1 in ranks else 0.0)

    if not pairs:
        raise SystemExit("no (patient, query) pairs had ground truth — nothing to gate")

    flat = [v for values in per_query.values() for v in values]
    return {
        "model": index.model,
        "k": k,
        "index_size": len(index),
        "patients": len(ids),
        "pairs": pairs,
        "retrieval": {
            f"recall@{k}": round(sum(flat) / len(flat), 4),
            f"hit@{k}": round(sum(hits_at_k) / len(hits_at_k), 4),
            "mrr": round(sum(mrr) / len(mrr), 4),
            "precision@1": round(sum(precision_at_1) / len(precision_at_1), 4),
        },
        "per_query": {
            qid: round(sum(values) / len(values), 4)
            for qid, values in sorted(per_query.items())
        },
        "cache": {"hits": cache.hits, "misses": cache.misses},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/index_30.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--patients", type=int, default=None)
    parser.add_argument("--out", default="-", help="file path, or - for stdout")
    args = parser.parse_args()

    metrics = measure(args.index, k=args.k, patients=args.patients)
    text = json.dumps(metrics, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
