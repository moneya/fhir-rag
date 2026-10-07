"""Full-corpus comparison: local open model vs hosted proprietary model.

The fixture comparison uses 5 queries on 1 patient — far too small to support a
claim. This runs the same comparison over all 30 patients and 92 (patient, query)
pairs, which is the sample the project's headline numbers come from.

The nemotron side is read from the existing index and embedding cache, so it costs
nothing and reproduces PR #1's numbers exactly. The candidate side re-embeds all
17,472 chunks locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def evaluate_all(index: VectorIndex, embedder, label: str) -> dict:
    """Every (patient, query) pair with ground truth, as in PR #1."""
    pids = sorted({c.patient_id for c in index.chunks})
    by_patient: dict[str, list] = {}
    for chunk in index.chunks:
        by_patient.setdefault(chunk.patient_id, []).append(chunk)

    recalls, mrr, hits, p1 = [], [], [], []
    per_query: dict[str, list[float]] = {}

    for pid in pids:
        chunks = by_patient[pid]
        for query in build_queries(pid):
            relevant = {c.chunk_id for c in chunks if query.relevant(c)}
            if not relevant:
                continue  # unmatched queries are skipped, never scored zero
            vector = embedder.embed_query(query.question)
            found = retrieve(
                index, vector, k=K, patient_id=pid,
                resource_types=query.resource_types, shape=query.shape,
            )
            ranks = [h.rank for h in found if h.chunk.chunk_id in relevant]
            recall = len(ranks) / len(relevant)
            recalls.append(recall)
            mrr.append(1.0 / min(ranks) if ranks else 0.0)
            hits.append(1.0 if ranks else 0.0)
            p1.append(1.0 if 1 in ranks else 0.0)
            per_query.setdefault(query.id, []).append(recall)

    n = len(recalls) or 1
    return {
        "label": label,
        "model": index.model,
        "dims": index.dims,
        "patients": len(pids),
        "pairs": len(recalls),
        f"recall@{K}": round(sum(recalls) / n, 4),
        f"hit@{K}": round(sum(hits) / n, 4),
        "mrr": round(sum(mrr) / n, 4),
        "precision@1": round(sum(p1) / n, 4),
        "per_query": {
            qid: round(sum(vals) / len(vals), 4)
            for qid, vals in sorted(per_query.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--index", default="data/index_30.json")
    parser.add_argument("--out", default="results/model_comparison_full.json")
    parser.add_argument("--limit", type=int, default=0, help="cap chunks for a smoke run")
    args = parser.parse_args()

    base_index = VectorIndex.load(args.index)
    print(f"loaded {len(base_index)} chunks, {base_index.dims} dims", file=sys.stderr)

    if args.limit:
        keep = list(zip(base_index.chunks, base_index.vectors))[: args.limit]
        trimmed = VectorIndex(model=base_index.model, asymmetric=base_index.asymmetric)
        trimmed.add([c for c, _ in keep], [list(v) for _, v in keep])
        base_index = trimmed

    # Baseline: nemotron query vectors come from the cache, so no key is needed.
    nemotron = get_embedder("nvidia", cache=EmbeddingCache(base_index.model))
    baseline = evaluate_all(base_index, nemotron, "nemotron-3-embed-1b (hosted, asymmetric)")
    print(
        f"baseline: {baseline['pairs']} pairs, "
        f"recall@{K}={baseline[f'recall@{K}']}",
        file=sys.stderr,
    )

    # Candidate: re-embed everything locally.
    texts = [c.text for c in base_index.chunks]
    local = get_embedder(
        "ollama", model=args.model, timeout=300.0,
        cache=EmbeddingCache(args.model), concurrency=4,
    )
    t0 = time.time()
    result = local.embed_passages(
        texts,
        on_progress=lambda done, total: (
            print(f"  embedded {done}/{total}", file=sys.stderr)
            if done % 2000 == 0 else None
        ),
    )
    elapsed = time.time() - t0
    print(
        f"embedded {len(texts)} chunks in {elapsed:.0f}s "
        f"({len(texts) / elapsed:.1f} chunks/s, {result.dims} dims)",
        file=sys.stderr,
    )

    local_index = VectorIndex(model=args.model, asymmetric=False)
    local_index.add(list(base_index.chunks), result.vectors)
    candidate = evaluate_all(local_index, local, f"{args.model} (local, symmetric)")
    candidate["embed_seconds"] = round(elapsed, 1)
    candidate["chunks_per_second"] = round(len(texts) / elapsed, 1)

    report = {
        "corpus": {"chunks": len(base_index), "k": K},
        "baseline": baseline,
        "candidate": candidate,
        "confound": (
            "nemotron is asymmetric (separate query/passage encoding), "
            "EmbeddingGemma is symmetric. Not a single-variable comparison."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'metric':14s} {'nemotron(2048)':>15s} {'gemma(768)':>12s} {'delta':>9s}")
    for metric in (f"recall@{K}", f"hit@{K}", "mrr", "precision@1"):
        b, c = baseline[metric], candidate[metric]
        print(f"{metric:14s} {b:15.4f} {c:12.4f} {c - b:+9.4f}")

    print(f"\nper-query recall@{K} (mean over patients):")
    for qid in sorted(baseline["per_query"]):
        b = baseline["per_query"][qid]
        c = candidate["per_query"].get(qid, 0.0)
        flag = "  WORSE" if c < b - 0.01 else ("  better" if c > b + 0.01 else "")
        print(f"  {qid:26s} {b:6.3f} -> {c:6.3f}{flag}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
