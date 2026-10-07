"""Bits vs dimensions at a fixed memory budget.

Qdrant's TurboQuant post reports that at ~72 bytes/vector, 512 dims at 1 bit kept
97.3% of nDCG@10 while 128 dims at 4 bits kept 84.0% — i.e. when memory is tight,
cut precision before dimensions. That is counter-intuitive and directly relevant:
the committed fixture is 1.0 MB of int8, and the obvious way to shrink it further
would have been to truncate dimensions.

So test it on this corpus rather than trusting a blog post. Three encodings of the
SAME 2048-dim nemotron vectors, compared on retrieval recall against the project's
own code-derived ground truth:

  int8      full dims, 1 byte/dim      (what the fixture ships)
  1-bit     full dims, sign only       (1/8 the bytes)
  truncate  first N dims, float        (matched to the 1-bit budget)

Matryoshka caveat: nemotron is not documented as Matryoshka-trained, so naive
truncation may be unfairly weak for it. That is stated, not hidden — and it is
exactly why EmbeddingGemma (which IS Matryoshka-trained) is the more interesting
model for this test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.retrieve import retrieve

K = 10


def to_int8(vector):
    return [max(-127, min(127, round(x * 127.0))) / 127.0 for x in vector]


def to_one_bit(vector):
    """Sign only. 1 bit/dim, so 2048 dims costs 256 bytes."""
    return [1.0 if x >= 0 else -1.0 for x in vector]


def truncate(vector, dims):
    return list(vector[:dims])


def score(index: VectorIndex, query_vectors: dict, pid: str) -> dict:
    recalls, mrr, p1 = [], [], []
    for query in build_queries(pid):
        relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
        if not relevant or query.id not in query_vectors:
            continue
        found = retrieve(
            index, query_vectors[query.id], k=K, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
        )
        ranks = [h.rank for h in found if h.chunk.chunk_id in relevant]
        recalls.append(len(ranks) / len(relevant))
        mrr.append(1.0 / min(ranks) if ranks else 0.0)
        p1.append(1.0 if 1 in ranks else 0.0)
    n = len(recalls) or 1
    return {
        f"recall@{K}": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrr) / n, 4),
        "precision@1": round(sum(p1) / n, 4),
        "pairs": len(recalls),
    }


def rebuild(source: VectorIndex, transform, qtransform, query_vectors):
    index = VectorIndex(model=source.model, asymmetric=source.asymmetric)
    index.add(list(source.chunks), [transform(v) for v in source.vectors])
    queries = {qid: qtransform(v) for qid, v in query_vectors.items()}
    return index, queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(ROOT / "tests/fixtures/index_fixture.json"))
    parser.add_argument("--queries", default=str(ROOT / "tests/fixtures/query_vectors.json"))
    parser.add_argument("--out", default="results/bits_vs_dims.json")
    args = parser.parse_args()

    source = VectorIndex.load(args.index)
    qvecs = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    pid = source.chunks[0].patient_id
    dims = source.dims

    # 1 bit/dim over `dims` dims == dims/8 bytes. A float32 index matching that
    # budget can afford dims/32 dimensions.
    one_bit_bytes = dims / 8
    matched_dims = max(1, int(one_bit_bytes / 4))

    print(f"{len(source)} chunks, {dims} dims")
    print(f"1-bit budget: {one_bit_bytes:.0f} bytes/vector")
    print(f"float32 at the same budget: {matched_dims} dims\n")

    rows = []

    base = score(source, qvecs, pid)
    rows.append(("int8 full dims", dims, dims * 1.0, base))

    bit_index, bit_q = rebuild(source, to_one_bit, to_one_bit, qvecs)
    rows.append(("1-bit full dims", dims, one_bit_bytes, score(bit_index, bit_q, pid)))

    trunc_index, trunc_q = rebuild(
        source,
        lambda v: truncate(v, matched_dims),
        lambda v: truncate(v, matched_dims),
        qvecs,
    )
    rows.append((
        f"float32 {matched_dims} dims", matched_dims, matched_dims * 4.0,
        score(trunc_index, trunc_q, pid),
    ))

    # A middle point: int8 truncated to half the dims, a common "just use fewer
    # dimensions" instinct.
    half = dims // 2
    half_index, half_q = rebuild(
        source,
        lambda v: to_int8(truncate(v, half)),
        lambda v: truncate(v, half),
        qvecs,
    )
    rows.append((f"int8 {half} dims", half, half * 1.0, score(half_index, half_q, pid)))

    print(f"{'encoding':22s} {'dims':>6s} {'bytes/vec':>10s} {'recall@10':>10s} {'mrr':>7s} {'p@1':>6s}")
    for label, d, nbytes, metrics in rows:
        print(
            f"{label:22s} {d:6d} {nbytes:10.0f} "
            f"{metrics[f'recall@{K}']:10.3f} {metrics['mrr']:7.3f} {metrics['precision@1']:6.3f}"
        )

    bit = rows[1][3][f"recall@{K}"]
    trunc = rows[2][3][f"recall@{K}"]
    print()
    if bit > trunc:
        print(f"CONFIRMED on this corpus: at ~{one_bit_bytes:.0f} bytes/vector, "
              f"1-bit full dims ({bit:.3f}) beats float32 {matched_dims} dims ({trunc:.3f}).")
        print("Cut bits before dimensions.")
    elif bit < trunc:
        print(f"NOT confirmed here: truncation ({trunc:.3f}) beat 1-bit ({bit:.3f}).")
    else:
        print(f"Tie at {bit:.3f} — this corpus cannot separate them.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "chunks": len(source),
                "dims": dims,
                "one_bit_bytes_per_vector": one_bit_bytes,
                "results": [
                    {"encoding": l, "dims": d, "bytes_per_vector": n, **m}
                    for l, d, n, m in rows
                ],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
