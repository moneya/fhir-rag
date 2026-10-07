"""Retrieval policy: how many candidates to fetch and how to re-rank them.

Measured finding that drives this module: diversification is not globally good
or bad, it depends on how many chunks legitimately answer the question.

  "single" answers (one hypertension diagnosis among ~100 blood-pressure
  readings) are invisible in a flat top-10 — recall 0.125 — and jump to 0.875
  when each resource type is capped at 2.

  "list" answers (every influenza vaccination, every recorded weight) are
  damaged by the same cap: flu-vaccination falls from 1.000 to 0.332, because
  the cap truncates an answer that genuinely is seven rows of one type.

A global heuristic cannot win both: capping moved macro recall@10 from 0.767
down to 0.523 while raising hit@10 from 0.826 to 0.957. So the policy is chosen
per query from its declared shape, and the trade is explicit rather than hidden
inside an average.
"""

from __future__ import annotations

from typing import Any

from .diversify import cap_per_resource_type
from .index import Hit, VectorIndex

# Over-fetch before re-ranking, so capping has material to work with.
OVERFETCH = 6


def retrieve(
    index: VectorIndex,
    query_vector: list[float],
    *,
    k: int = 10,
    patient_id: str | None = None,
    resource_types: tuple[str, ...] | None = None,
    shape: str = "list",
    cap: int = 2,
    reranker: Any | None = None,
    question: str | None = None,
) -> list[Hit]:
    """Top-k hits, re-ranked according to the expected answer shape.

    With a `reranker`, candidates are over-fetched for BOTH shapes and reordered
    by the model before truncation — reranking only helps if it can see more
    candidates than the final k. `question` is required then, since the reranker
    scores text against the question rather than against the query vector.
    """
    if shape not in ("single", "list"):
        raise ValueError(f"shape must be 'single' or 'list', got {shape!r}")
    if reranker is not None and not question:
        raise ValueError("reranker needs the question text, not just its vector")

    overfetch = shape == "single" or reranker is not None
    fetch = k * OVERFETCH if overfetch else k
    hits = index.search(
        query_vector, k=fetch, patient_id=patient_id, resource_types=resource_types
    )

    if reranker is not None:
        outcome = reranker.rerank(question, hits, k=k)
        hits = outcome.hits
        # A fallback is not an error: embedding order is still a valid ranking.
        if shape == "single":
            return cap_per_resource_type(hits, k=k, cap=cap)
        return hits[:k]

    if shape == "single":
        return cap_per_resource_type(hits, k=k, cap=cap)
    return hits[:k]


__all__ = ["OVERFETCH", "retrieve"]
