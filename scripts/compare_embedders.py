"""Compare a local open embedding model against the hosted proprietary one.

Same corpus, same ground truth, same retrieval code, same k. The only variable is
which model produced the vectors.

Why this is worth measuring rather than assuming: the project's numbers were all
produced by `nvidia/nemotron-3-embed-1b`, a 2048-dim hosted model requiring a key,
which is also why the full CI gate cannot run without one. If a 768-dim model that
runs on a laptop is close, the honest conclusion is that the dependency buys less
than it costs.

CONFOUND, stated up front: nemotron is asymmetric (separate `input_type` for
queries and passages), EmbeddingGemma is symmetric. That asymmetry was the
original reason for choosing nemotron. So this is not a clean single-variable
comparison, and a loss for the local model is not by itself a verdict on the
model — it may be a verdict on symmetric encoding for this task. Reported either
way, not explained away.
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

from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries
from fhir_rag.retrieve import retrieve

K = 10


def evaluate(index: VectorIndex, embedder, pid: str, label: str) -> dict:
    """Run every query with ground truth and return retrieval metrics."""
    recalls, mrr, hits, p1 = [], [], [], []
    per_query = {}

    for query in build_queries(pid):
        relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
        if not relevant:
            continue
        vector = embedder.embed_query(query.question)
        found = retrieve(
            index, vector, k=K, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
        )
        ranks = [h.rank for h in found if h.chunk.chunk_id in relevant]
        recall = len(ranks) / len(relevant)
        per_query[query.id] = round(recall, 4)
        recalls.append(recall)
        mrr.append(1.0 / min(ranks) if ranks else 0.0)
        hits.append(1.0 if ranks else 0.0)
        p1.append(1.0 if 1 in ranks else 0.0)

    n = len(recalls) or 1
    return {
        "label": label,
        "model": index.model,
        "dims": index.dims,
        "pairs": len(recalls),
        f"recall@{K}": round(sum(recalls) / n, 4),
        f"hit@{K}": round(sum(hits) / n, 4),
        "mrr": round(sum(mrr) / n, 4),
        "precision@1": round(sum(p1) / n, 4),
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--baseline", default=str(ROOT / "tests/fixtures/index_fixture.json"))
    parser.add_argument("--baseline-queries", default=str(ROOT / "tests/fixtures/query_vectors.json"))
    parser.add_argument("--out", default="results/model_comparison.json")
    args = parser.parse_args()

    # --- baseline: the committed fixture, nemotron vectors already in it ------
    base_index = VectorIndex.load(args.baseline)
    pid = base_index.chunks[0].patient_id
    qvecs = json.loads(Path(args.baseline_queries).read_text(encoding="utf-8"))

    class Frozen:
        """Replays the stored query vectors so the baseline needs no API key."""
        def embed_query(self, text: str) -> list[float]:
            for query in build_queries(pid):
                if query.question == text:
                    return qvecs[query.id]
            raise KeyError(text)

    baseline = evaluate(base_index, Frozen(), pid, "nemotron (hosted, asymmetric)")

    # --- candidate: re-embed the same chunks locally --------------------------
    texts = [c.text for c in base_index.chunks]
    embedder = get_embedder("ollama", model=args.model, timeout=180.0)

    t0 = time.time()
    result = embedder.embed_passages(texts)
    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed else 0.0
    print(
        f"embedded {len(texts)} chunks with {args.model} in {elapsed:.1f}s "
        f"({rate:.1f} chunks/s, {result.dims} dims)",
        file=sys.stderr,
    )

    local_index = VectorIndex(model=args.model, asymmetric=False)
    local_index.add(list(base_index.chunks), result.vectors)
    candidate = evaluate(local_index, embedder, pid, f"{args.model} (local, symmetric)")
    candidate["embed_seconds"] = round(elapsed, 1)
    candidate["chunks_per_second"] = round(rate, 1)

    report = {
        "corpus": {"chunks": len(base_index), "patient": pid, "k": K},
        "baseline": baseline,
        "candidate": candidate,
        "note": (
            "nemotron is asymmetric, EmbeddingGemma is symmetric. Not a clean "
            "single-variable comparison; a loss may reflect symmetric encoding "
            "rather than model capacity."
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'metric':14s} {'nemotron-2048':>14s} {args.model[:16]:>16s}")
    for metric in (f"recall@{K}", f"hit@{K}", "mrr", "precision@1"):
        print(f"{metric:14s} {baseline[metric]:14.3f} {candidate[metric]:16.3f}")
    print(f"\nper-query recall@{K}:")
    for qid in sorted(baseline["per_query"]):
        b, c = baseline["per_query"][qid], candidate["per_query"].get(qid, 0.0)
        flag = "  <-- WORSE" if c < b - 1e-9 else ("  <-- better" if c > b + 1e-9 else "")
        print(f"  {qid:24s} {b:6.3f} {c:8.3f}{flag}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
