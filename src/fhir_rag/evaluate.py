"""Retrieval evaluation with ground truth taken from the data, not from opinion.

The honest difficulty in RAG is knowing whether retrieval actually worked.
Hand-labelled relevance is small, subjective and unreproducible. Synthea gives
something better: every clinical fact carries a coded identity (SNOMED, RxNorm,
LOINC, CVX), so for a question like "which diabetes medication is this patient
taking?" the set of correct chunks is *derivable* — it is exactly the
MedicationRequest resources whose RxNorm code is in the diabetes drug set.

That makes recall@k a computed number rather than a claim, and it is why this
project uses Synthea instead of real de-identified records.

Metrics:

  recall@k      fraction of relevant chunks appearing in the top k
  precision@k   fraction of the top k that are relevant
  hit@k         did at least one relevant chunk appear (the "did RAG work" number)
  MRR           1/rank of the first relevant chunk, averaged
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .index import Hit, VectorIndex
from .ingest import Chunk


@dataclass
class Query:
    """A question plus the rule that decides which chunks answer it."""

    id: str
    question: str
    # A chunk is relevant if this returns True. Derived from codes where possible.
    relevant: Callable[[Chunk], bool]
    patient_id: str | None = None
    resource_types: tuple[str, ...] | None = None
    note: str = ""
    # Expected answer shape. Measured on this corpus, diversification helps
    # "single" queries (one diagnosis buried under 100 measurements) and hurts
    # "list" queries (every vaccination, every weight), so the ranking policy is
    # chosen per query rather than globally. See docs/diversification.md.
    shape: str = "list"  # "single" | "list"


@dataclass
class QueryResult:
    query_id: str
    question: str
    relevant_total: int
    retrieved: list[Hit]
    relevant_ranks: list[int] = field(default_factory=list)

    def recall_at(self, k: int) -> float | None:
        if self.relevant_total == 0:
            return None
        found = sum(1 for r in self.relevant_ranks if r <= k)
        return found / self.relevant_total

    def precision_at(self, k: int) -> float | None:
        if not self.retrieved:
            return None
        denominator = min(k, len(self.retrieved))
        if denominator == 0:
            return None
        return sum(1 for r in self.relevant_ranks if r <= k) / denominator

    def hit_at(self, k: int) -> bool | None:
        if self.relevant_total == 0:
            return None
        return any(r <= k for r in self.relevant_ranks)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / min(self.relevant_ranks) if self.relevant_ranks else 0.0


@dataclass
class EvalReport:
    model: str
    asymmetric: bool
    k: int
    index_size: int
    results: list[QueryResult]

    @property
    def scored(self) -> list[QueryResult]:
        """Queries that have at least one relevant chunk.

        A question with no relevant chunk in the corpus measures nothing, so it
        is excluded rather than silently counted as a failure — which would make
        a sparser dataset look like a worse retriever.
        """
        return [r for r in self.results if r.relevant_total > 0]

    def mean(self, fn: Callable[[QueryResult], float | None]) -> float | None:
        values = [v for v in (fn(r) for r in self.scored) if v is not None]
        return sum(values) / len(values) if values else None

    def summary(self) -> dict[str, Any]:
        ks = sorted({1, 3, 5, self.k})
        out: dict[str, Any] = {
            "model": self.model,
            "asymmetric": self.asymmetric,
            "index_size": self.index_size,
            "queries_total": len(self.results),
            "queries_scored": len(self.scored),
            "queries_skipped_no_ground_truth": len(self.results) - len(self.scored),
            "mrr": self.mean(lambda r: r.reciprocal_rank),
        }
        for k in ks:
            if k > self.k:
                continue
            out[f"recall@{k}"] = self.mean(lambda r, k=k: r.recall_at(k))
            out[f"precision@{k}"] = self.mean(lambda r, k=k: r.precision_at(k))
            out[f"hit@{k}"] = self.mean(
                lambda r, k=k: None if r.hit_at(k) is None else float(r.hit_at(k))
            )
        return out

    def to_json(self) -> str:
        return json.dumps(
            {
                "summary": self.summary(),
                "queries": [
                    {
                        "id": r.query_id,
                        "question": r.question,
                        "relevant_total": r.relevant_total,
                        "relevant_ranks": r.relevant_ranks,
                        "recall@k": r.recall_at(self.k),
                        "top_hits": [
                            {
                                "rank": h.rank,
                                "score": round(h.score, 4),
                                "chunk_id": h.chunk.chunk_id,
                                "text": h.chunk.text,
                                "relevant": h.rank in r.relevant_ranks,
                            }
                            for h in r.retrieved[:5]
                        ],
                    }
                    for r in self.results
                ],
            },
            indent=2,
        )


def evaluate(
    index: VectorIndex,
    queries: Sequence[Query],
    embed_query: Callable[[str], list[float]],
    *,
    k: int = 10,
) -> EvalReport:
    results: list[QueryResult] = []
    for query in queries:
        candidates = [
            c for c in index.chunks
            if (query.patient_id is None or c.patient_id == query.patient_id)
        ]
        relevant_ids = {c.chunk_id for c in candidates if query.relevant(c)}

        hits = index.search(
            embed_query(query.question),
            k=k,
            patient_id=query.patient_id,
            resource_types=query.resource_types,
        )
        ranks = [h.rank for h in hits if h.chunk.chunk_id in relevant_ids]
        results.append(QueryResult(
            query_id=query.id,
            question=query.question,
            relevant_total=len(relevant_ids),
            retrieved=hits,
            relevant_ranks=ranks,
        ))

    return EvalReport(
        model=index.model,
        asymmetric=index.asymmetric,
        k=k,
        index_size=len(index),
        results=results,
    )


def format_report(report: EvalReport) -> str:
    s = report.summary()
    lines = [
        "",
        f"  retrieval eval — {s['model']}"
        f"{'  (asymmetric)' if s['asymmetric'] else '  (symmetric)'}",
        f"  {s['index_size']} chunks indexed, {s['queries_scored']}/{s['queries_total']} queries scored",
        "",
    ]
    if s["queries_skipped_no_ground_truth"]:
        lines.append(
            f"  {s['queries_skipped_no_ground_truth']} query(ies) skipped: no relevant chunk in corpus"
        )
        lines.append("")
    for k in (1, 3, 5, report.k):
        key = f"recall@{k}"
        if key not in s or s[key] is None:
            continue
        lines.append(
            f"  k={k:<3} recall {s[key]:.3f}   precision {s[f'precision@{k}']:.3f}"
            f"   hit-rate {s[f'hit@{k}']:.3f}"
        )
    if s["mrr"] is not None:
        lines += ["", f"  MRR {s['mrr']:.3f}"]
    return "\n".join(lines) + "\n"


__all__ = ["EvalReport", "Query", "QueryResult", "evaluate", "format_report"]
