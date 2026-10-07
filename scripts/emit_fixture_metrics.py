"""Measure the fixture index and emit gate metrics.

Separate from `emit_metrics.py` on purpose: that script measures the real
30-patient index with live query embeddings, this one measures the committed
fixture with query vectors that also ship in the repo. Same code path through
`retrieve()`, different inputs, and no network either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.retrieve import retrieve

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(FIXTURE_DIR / "index_fixture.json"))
    parser.add_argument("--queries", default=str(FIXTURE_DIR / "query_vectors.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    index = VectorIndex.load(args.index)
    query_vectors = json.loads(Path(args.queries).read_text(encoding="utf-8"))

    pid = index.chunks[0].patient_id
    per_query: dict[str, float] = {}
    mrr, hits, p1 = [], [], []

    for query in build_queries(pid):
        relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
        if not relevant:
            continue
        vector = query_vectors.get(query.id)
        if vector is None:
            raise SystemExit(
                f"no stored query vector for {query.id!r} — "
                f"rebuild with scripts/build_fixture_queries.py"
            )
        found = retrieve(
            index, vector, k=args.k, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
        )
        ranks = [h.rank for h in found if h.chunk.chunk_id in relevant]
        per_query[query.id] = round(len(ranks) / len(relevant), 4)
        mrr.append(1.0 / min(ranks) if ranks else 0.0)
        hits.append(1.0 if ranks else 0.0)
        p1.append(1.0 if 1 in ranks else 0.0)

    if not per_query:
        raise SystemExit("fixture has no query with ground truth — it gates nothing")

    values = list(per_query.values())
    metrics = {
        "model": index.model,
        "encoding": "int8-base64",
        "k": args.k,
        "index_size": len(index),
        "patients": 1,
        "pairs": len(values),
        "retrieval": {
            f"recall@{args.k}": round(sum(values) / len(values), 4),
            f"hit@{args.k}": round(sum(hits) / len(hits), 4),
            "mrr": round(sum(mrr) / len(mrr), 4),
            "precision@1": round(sum(p1) / len(p1), 4),
        },
        "per_query": dict(sorted(per_query.items())),
    }

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
