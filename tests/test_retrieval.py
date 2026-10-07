"""Tests for diversification and shape-aware retrieval.

The measured finding these encode: capping per resource type rescues queries
whose answer is a single chunk and damages queries whose answer is legitimately
a list. The policy therefore depends on the query, and these tests pin that
behaviour so a future "simplification" to one global strategy fails loudly.
"""

from __future__ import annotations

import pytest

from fhir_rag.diversify import cap_per_resource_type, interleave_by_type, mmr
from fhir_rag.index import Hit, VectorIndex
from fhir_rag.ingest import Chunk
from fhir_rag.retrieve import retrieve


def chunk(kind: str, n: int, patient: str = "p1") -> Chunk:
    return Chunk(
        patient_id=patient,
        patient_name="Test Patient",
        resource_type=kind,
        resource_id=f"{kind.lower()}-{n}",
        text=f"{kind} number {n}",
    )


def hits(spec: list[tuple[str, float]]) -> list[Hit]:
    return [
        Hit(chunk=chunk(kind, i), score=score, rank=i + 1)
        for i, (kind, score) in enumerate(spec)
    ]


# -- cap_per_resource_type ------------------------------------------------

def test_cap_surfaces_a_buried_minority_type():
    """The real failure: 8 blood-pressure readings bury one diagnosis."""
    ranked = hits([("Observation", 0.9 - i * 0.01) for i in range(8)] + [("Condition", 0.5)])
    out = cap_per_resource_type(ranked, k=3, cap=2)
    kinds = [h.chunk.resource_type for h in out]
    assert kinds.count("Observation") == 2
    assert "Condition" in kinds, "the lone diagnosis must reach the top-3"


def test_cap_refill_can_exceed_the_cap_to_fill_k():
    """Documented trade-off: k wins over cap when types run out.

    With 2 types, cap=2 and k=4 only 3 slots can honour the cap. Returning 3
    results when 4 were asked for is worse than relaxing the cap, so overflow
    refills the tail — the minority type is already in by then.
    """
    ranked = hits([("Observation", 0.9 - i * 0.01) for i in range(8)] + [("Condition", 0.5)])
    out = cap_per_resource_type(ranked, k=4, cap=2)
    assert len(out) == 4
    kinds = [h.chunk.resource_type for h in out]
    assert "Condition" in kinds
    assert kinds.count("Observation") == 3  # the 3rd arrives via refill


def test_cap_preserves_relative_order_within_a_type():
    ranked = hits([("Observation", 0.9), ("Observation", 0.8), ("Observation", 0.7)])
    out = cap_per_resource_type(ranked, k=2, cap=2)
    assert [h.score for h in out] == [0.9, 0.8]


def test_cap_refills_rather_than_returning_fewer_than_k():
    """One type only: capping must not shrink the result set."""
    ranked = hits([("Observation", 0.9 - i * 0.1) for i in range(6)])
    out = cap_per_resource_type(ranked, k=5, cap=2)
    assert len(out) == 5


def test_cap_reassigns_ranks_contiguously():
    ranked = hits([("Observation", 0.9), ("Observation", 0.8), ("Condition", 0.7)])
    out = cap_per_resource_type(ranked, k=3, cap=1)
    assert [h.rank for h in out] == [1, 2, 3]


def test_cap_rejects_zero():
    with pytest.raises(ValueError):
        cap_per_resource_type(hits([("Observation", 0.5)]), k=1, cap=0)


# -- interleave -----------------------------------------------------------

def test_interleave_gives_every_type_its_best_hit_first():
    ranked = hits([("Observation", 0.9), ("Observation", 0.85), ("Condition", 0.4)])
    out = interleave_by_type(ranked, k=2)
    assert [h.chunk.resource_type for h in out] == ["Observation", "Condition"]


def test_interleave_strongest_type_leads():
    ranked = hits([("Condition", 0.95), ("Observation", 0.9)])
    out = interleave_by_type(ranked, k=2)
    assert out[0].chunk.resource_type == "Condition"


# -- mmr ------------------------------------------------------------------

def test_mmr_at_lambda_one_is_the_flat_ranking():
    ranked = hits([("Observation", 0.9), ("Observation", 0.8), ("Condition", 0.7)])
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    out = mmr(ranked, vectors, k=3, lambda_=1.0)
    assert [h.score for h in out] == [0.9, 0.8, 0.7]


def test_mmr_demotes_a_near_duplicate():
    """Second pick should be the dissimilar chunk, not the identical vector."""
    ranked = hits([("Observation", 0.90), ("Observation", 0.89), ("Condition", 0.60)])
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    out = mmr(ranked, vectors, k=2, lambda_=0.5)
    assert out[1].chunk.resource_type == "Condition"


def test_mmr_validates_inputs():
    with pytest.raises(ValueError):
        mmr(hits([("Observation", 0.5)]), [[1.0]], k=1, lambda_=1.5)
    with pytest.raises(ValueError, match="2 hits but 1 vectors"):
        mmr(hits([("Observation", 0.5), ("Observation", 0.4)]), [[1.0]], k=1)


# -- shape-aware retrieval ------------------------------------------------

def build_index() -> VectorIndex:
    """100 near-identical observations plus one distinct condition.

    Mirrors the real corpus shape that caused recall 0.125 for hypertension: the
    condition scores lower than every observation, so a flat top-10 never shows
    it, but it is within the over-fetched window.
    """
    index = VectorIndex(model="test", asymmetric=False)
    chunks, vectors = [], []
    # 30 observations that all match the query strongly.
    for i in range(30):
        chunks.append(chunk("Observation", i))
        vectors.append([1.0, 0.001 * i, 0.0])
    # One condition that matches less well — below every observation, but well
    # inside the over-fetched window, exactly like the real corpus. The vector
    # must point in a DIFFERENT direction, not merely be shorter: the index
    # normalises, so a scaled copy of the same direction scores identically.
    chunks.append(chunk("Condition", 0))
    vectors.append([0.7, 0.7, 0.0])
    index.add(chunks, vectors)
    return index


def test_single_shape_surfaces_the_condition():
    """The headline fix: hypertension went 0.125 -> 0.875 recall this way."""
    index = build_index()
    flat = retrieve(index, [1.0, 0.0, 0.0], k=10, shape="list")
    assert "Condition" not in [h.chunk.resource_type for h in flat], (
        "fixture is wrong: the condition must be buried under a flat ranking"
    )

    out = retrieve(index, [1.0, 0.0, 0.0], k=10, shape="single", cap=2)
    kinds = [h.chunk.resource_type for h in out]
    assert "Condition" in kinds, "a lone diagnosis must not stay buried"
    # The condition is reached early, not pushed to the tail by refill.
    assert kinds.index("Condition") <= 2


def test_list_shape_keeps_the_dense_answer_intact():
    """The counter-case: capping a genuine list answer destroys recall."""
    index = build_index()
    out = retrieve(index, [1.0, 0.0, 0.0], k=10, shape="list")
    kinds = [h.chunk.resource_type for h in out]
    assert kinds.count("Observation") == 10


def test_shape_must_be_valid():
    with pytest.raises(ValueError, match="single"):
        retrieve(build_index(), [1.0, 0.0, 0.0], k=5, shape="everything")


def test_queries_declare_a_valid_shape():
    from fhir_rag.queries import build_queries

    for query in build_queries():
        assert query.shape in ("single", "list"), f"{query.id}: {query.shape}"


def test_sparse_answer_queries_are_marked_single():
    """Shapes were measured, not guessed: these have ~1 relevant chunk each."""
    from fhir_rag.queries import build_queries

    by_id = {q.id: q for q in build_queries()}
    for qid in ("hypertension-diagnosis", "prediabetes", "kidney-problems"):
        assert by_id[qid].shape == "single", qid
    for qid in ("flu-vaccination", "body-weight", "hba1c-results"):
        assert by_id[qid].shape == "list", qid
