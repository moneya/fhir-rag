"""Tests for LLM listwise reranking.

Two properties matter more than ranking quality, because violating either is
silent and corrupts results rather than failing:

  * the output is a PERMUTATION of the input — never invents, never drops
  * any failure returns embedding order, never an empty or partial list

No API key is needed: a fake reranker returns canned replies, including the
malformed ones a real model produces on a bad day.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from fhir_rag.index import Hit
from fhir_rag.ingest import Chunk
from fhir_rag.rerank import (
    GroqReranker,
    RerankError,
    complete_ranking,
    parse_ranking,
)
from fhir_rag.retrieve import retrieve


def chunk(kind: str, n: int, text: str | None = None) -> Chunk:
    return Chunk(
        patient_id="p1",
        patient_name="Test Patient",
        resource_type=kind,
        resource_id=f"{kind}-{n}",
        text=text or f"{kind} number {n}",
    )


def hits(n: int, kind: str = "Observation") -> list[Hit]:
    return [
        Hit(chunk=chunk(kind, i), score=0.9 - i * 0.01, rank=i + 1)
        for i in range(n)
    ]


class StubReranker(GroqReranker):
    """GroqReranker with the HTTP call replaced by a canned reply."""

    def __init__(self, reply: str | Exception, **kw):
        kw.setdefault("api_key", "test-key-not-real")
        super().__init__(**kw)
        self.reply = reply
        self.prompts: list[str] = []

    def _chat(self, prompt: str) -> tuple[str, int]:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, 7


# -- parsing --------------------------------------------------------------

def test_parses_a_plain_comma_list():
    assert parse_ranking("2,0,1", 3) == [2, 0, 1]


def test_parses_through_prose_and_brackets():
    """Models wrap answers in text; the ids still have to come out."""
    assert parse_ranking("Sure! The ranking is [3], [1], then [0].", 4) == [3, 1, 0]


def test_drops_out_of_range_ids():
    """A hallucinated index must not raise IndexError downstream."""
    assert parse_ranking("0, 99, 1", 3) == [0, 1]


def test_drops_duplicates_keeping_first_position():
    assert parse_ranking("2, 2, 0, 2", 3) == [2, 0]


def test_empty_and_prose_only_replies_yield_nothing():
    assert parse_ranking("", 5) == []
    assert parse_ranking("I cannot help with that.", 5) == []


def test_complete_ranking_appends_omitted_candidates():
    """The model ranked 2 of 5; the other 3 keep embedding order at the tail."""
    assert complete_ranking([3, 0], 5) == [3, 0, 1, 2, 4]


def test_complete_ranking_is_identity_when_already_full():
    assert complete_ranking([1, 0, 2], 3) == [1, 0, 2]


# -- permutation guarantee ------------------------------------------------

def test_output_is_a_permutation_of_input():
    reranker = StubReranker("4,3,2,1,0")
    out = reranker.rerank("q", hits(5), k=5)
    assert out.reranked
    assert sorted(h.chunk.chunk_id for h in out.hits) == sorted(
        h.chunk.chunk_id for h in hits(5)
    )


def test_partial_reply_still_returns_every_candidate():
    """Model named 2 of 6 — the rest must survive, in embedding order."""
    reranker = StubReranker("5,4")
    out = reranker.rerank("q", hits(6), k=6)
    assert len(out.hits) == 6
    ids = [h.chunk.resource_id for h in out.hits]
    assert ids[:2] == ["Observation-5", "Observation-4"]
    assert ids[2:] == [f"Observation-{i}" for i in (0, 1, 2, 3)]


def test_hallucinated_ids_do_not_lose_candidates():
    reranker = StubReranker("42, 1, 99")
    out = reranker.rerank("q", hits(3), k=3)
    assert len(out.hits) == 3
    assert out.hits[0].chunk.resource_id == "Observation-1"


def test_ranks_are_renumbered_contiguously():
    out = StubReranker("2,1,0").rerank("q", hits(3), k=3)
    assert [h.rank for h in out.hits] == [1, 2, 3]


def test_scores_are_preserved_not_invented():
    """Reranking changes order, not similarity scores."""
    original = {h.chunk.chunk_id: h.score for h in hits(4)}
    out = StubReranker("3,2,1,0").rerank("q", hits(4), k=4)
    for hit in out.hits:
        assert hit.score == original[hit.chunk.chunk_id]


def test_candidates_beyond_the_window_are_kept():
    reranker = StubReranker("1,0", max_candidates=2)
    out = reranker.rerank("q", hits(5), k=5)
    assert len(out.hits) == 5
    assert [h.chunk.resource_id for h in out.hits[:2]] == ["Observation-1", "Observation-0"]
    # The tail keeps embedding order.
    assert [h.chunk.resource_id for h in out.hits[2:]] == [
        "Observation-2", "Observation-3", "Observation-4"
    ]


# -- fail closed ----------------------------------------------------------

def test_network_error_falls_back_to_embedding_order():
    reranker = StubReranker(urllib.error.URLError("no route to host"))
    out = reranker.rerank("q", hits(4), k=4)
    assert out.reranked is False
    assert "URLError" in out.reason
    assert [h.chunk.resource_id for h in out.hits] == [f"Observation-{i}" for i in range(4)]


def test_http_error_falls_back():
    error = urllib.error.HTTPError("u", 429, "rate limited", {}, None)
    out = StubReranker(error).rerank("q", hits(3), k=3)
    assert out.reranked is False
    assert len(out.hits) == 3


def test_unusable_reply_falls_back_with_a_reason():
    out = StubReranker("I'm sorry, I can't rank medical records.").rerank("q", hits(3), k=3)
    assert out.reranked is False
    assert "no usable ids" in out.reason
    assert [h.chunk.resource_id for h in out.hits] == [f"Observation-{i}" for i in range(3)]


def test_malformed_json_response_falls_back():
    out = StubReranker(KeyError("choices")).rerank("q", hits(3), k=3)
    assert out.reranked is False
    assert len(out.hits) == 3


def test_single_candidate_is_not_sent_to_the_model():
    reranker = StubReranker("0")
    out = reranker.rerank("q", hits(1), k=1)
    assert out.reranked is False
    assert reranker.prompts == [], "no API call for a 1-item list"


def test_missing_key_is_an_explicit_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RerankError, match="GROQ_API_KEY"):
        _ = GroqReranker().api_key


# -- prompt shape ---------------------------------------------------------

def test_prompt_asks_to_rank_all_not_to_filter():
    """Measured: 'return only relevant' kept 1 of 7 vaccinations, 'rank all' kept 7.

    This test exists so nobody 'simplifies' the instruction back to filtering.
    """
    reranker = StubReranker("0,1")
    reranker.rerank("When was the patient vaccinated?", hits(3), k=3)
    prompt = reranker.prompts[0]
    assert "Rank ALL" in prompt
    assert "only the passage numbers that answer" not in prompt.lower()


def test_prompt_numbers_every_candidate():
    reranker = StubReranker("0")
    reranker.rerank("q", hits(3), k=3)
    prompt = reranker.prompts[0]
    for i in range(3):
        assert f"[{i}]" in prompt


# -- integration with retrieve() -----------------------------------------

def build_index():
    from fhir_rag.index import VectorIndex

    index = VectorIndex(model="test", asymmetric=False)
    chunks, vectors = [], []
    # 20 observations so a k=4 retrieve (over-fetching 24) still sees the
    # Condition, which scores below every one of them.
    for i in range(20):
        chunks.append(chunk("Observation", i))
        vectors.append([1.0, 0.001 * i, 0.0])
    chunks.append(chunk("Condition", 0))
    vectors.append([0.7, 0.7, 0.0])
    index.add(chunks, vectors)
    return index


def test_retrieve_requires_the_question_when_reranking():
    with pytest.raises(ValueError, match="question text"):
        retrieve(build_index(), [1.0, 0.0, 0.0], k=5, reranker=StubReranker("0"))


def test_retrieve_overfetches_for_list_shape_when_reranking():
    """Reranking only helps if it sees more candidates than the final k."""
    reranker = StubReranker("0,1,2")
    retrieve(
        build_index(), [1.0, 0.0, 0.0], k=5, shape="list",
        reranker=reranker, question="q",
    )
    numbered = reranker.prompts[0].count("[")
    assert numbered > 5, "list shape must over-fetch for the reranker"


def test_retrieve_applies_cap_after_reranking():
    """Order comes from the model, then the cap runs on that new order.

    The stub promotes the Condition (last candidate in embedding order) to the
    front, mimicking a reranker that correctly spots the diagnosis. The cap must
    then still limit the Observations that follow.
    """
    index = build_index()
    # Locate the Condition inside the window retrieve() will actually fetch.
    window = index.search([1.0, 0.0, 0.0], k=4 * 6, patient_id="p1")
    condition_at = next(
        (i for i, h in enumerate(window) if h.chunk.resource_type == "Condition"),
        None,
    )
    assert condition_at is not None, (
        "fixture must place the Condition inside the over-fetched window"
    )
    # Promote it to the front, as a correct reranker would.
    reranker = StubReranker(",".join(str(i) for i in [condition_at, 0, 1, 2, 3]))
    out = retrieve(
        index, [1.0, 0.0, 0.0], k=4, shape="single", cap=2,
        reranker=reranker, question="q",
    )
    kinds = [h.chunk.resource_type for h in out]
    assert kinds[0] == "Condition", "reranked order must drive the cap"
    assert kinds.count("Observation") == 3  # cap 2, then refill fills slot 4


def test_retrieve_returns_k_results_when_reranker_fails():
    out = retrieve(
        build_index(), [1.0, 0.0, 0.0], k=5, shape="list",
        reranker=StubReranker(urllib.error.URLError("down")), question="q",
    )
    assert len(out) == 5
