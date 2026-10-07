"""Tests for batching, concurrency and vector ordering.

The failure these guard against is silent: if a concurrent batch's results are
reassembled in completion order rather than input order, every vector attaches
to the wrong chunk. Retrieval still returns plausible-looking results, so
nothing crashes and the recall number is quietly meaningless.
"""

from __future__ import annotations

import threading
import time

import pytest

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import (
    EmbeddingBackend,
    EmbeddingError,
    EmbeddingResult,
    available,
    get_embedder,
)


class FakeBackend(EmbeddingBackend):
    """Returns a vector encoding each text's identity, so misordering is visible."""

    name = "fake"
    model = "fake-model"
    requires_key = False
    batch_limit = 4

    @property
    def default_base_url(self) -> str:
        return "http://fake"

    def __init__(self, *, delay_first_batch: float = 0.0, delay_every: float = 0.0, **kw):
        super().__init__(**kw)
        self.calls: list[list[str]] = []
        self.delay_first_batch = delay_first_batch
        self.delay_every = delay_every
        self._lock = threading.Lock()

    def _embed(self, texts, *, is_query: bool) -> EmbeddingResult:
        with self._lock:
            self.calls.append(list(texts))
            first = len(self.calls) == 1
        # Make the FIRST batch slowest, so completion order != input order.
        if first and self.delay_first_batch:
            time.sleep(self.delay_first_batch)
        elif self.delay_every:
            time.sleep(self.delay_every)
        role = 1.0 if is_query else 0.0
        return EmbeddingResult(
            vectors=[[float(len(t)), float(ord(t[0])), role] for t in texts],
            model=self.model,
            tokens=len(texts),
        )


def expected(text: str, *, is_query: bool = False) -> list[float]:
    return [float(len(text)), float(ord(text[0])), 1.0 if is_query else 0.0]


def test_single_batch():
    backend = FakeBackend()
    result = backend.embed_passages(["aa", "bb"])
    assert result.vectors == [expected("aa"), expected("bb")]


def test_batches_split_at_limit():
    backend = FakeBackend()
    texts = [f"{chr(97 + i)}x" for i in range(10)]
    backend.embed_passages(texts)
    assert [len(c) for c in backend.calls] == [4, 4, 2]


def test_order_preserved_sequentially():
    backend = FakeBackend()
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(9)]
    result = backend.embed_passages(texts)
    assert result.vectors == [expected(t) for t in texts]


def test_order_preserved_when_batches_finish_out_of_order():
    """The real concurrency hazard, forced: batch 1 returns last."""
    backend = FakeBackend(delay_first_batch=0.25, concurrency=4)
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(12)]
    result = backend.embed_passages(texts)
    assert result.vectors == [expected(t) for t in texts], "vectors misaligned"


def test_concurrency_is_actually_faster():
    """Every batch is slow, so sequential cost scales with batch count.

    An earlier version of this test delayed only the FIRST batch, which made
    both paths take the same ~0.3s and passed for the wrong reason.
    """
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(12)]  # 3 batches of 4
    delay = 0.2

    t0 = time.time()
    FakeBackend(delay_every=delay, concurrency=1).embed_passages(texts)
    sequential = time.time() - t0

    t0 = time.time()
    FakeBackend(delay_every=delay, concurrency=4).embed_passages(texts)
    parallel = time.time() - t0

    assert sequential > delay * 2.5, f"sequential should be ~3x delay, got {sequential:.2f}s"
    assert parallel < sequential * 0.6, f"parallel {parallel:.2f}s vs sequential {sequential:.2f}s"


def test_progress_callback_reports_monotonic_completion():
    seen: list[tuple[int, int]] = []
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(10)]
    FakeBackend(concurrency=3).embed_passages(texts, on_progress=lambda d, t: seen.append((d, t)))
    assert seen, "no progress reported"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert seen[-1] == (10, 10)
    assert all(total == 10 for _, total in seen)


def test_cache_is_written_incrementally_not_only_at_the_end(tmp_path):
    """A crash mid-run must not lose vectors already paid for.

    The backend flushes each batch as it lands; this asserts entries exist on
    disk before the final batch completes.
    """
    from fhir_rag.cache import EmbeddingCache

    cache = EmbeddingCache("fake-model", tmp_path)
    backend = FakeBackend(cache=cache, concurrency=1, delay_every=0.05)
    texts = [f"{chr(97 + i)}{'x' * i}" for i in range(12)]  # 3 batches

    on_disk: list[int] = []

    def probe(done: int, total: int) -> None:
        lines = cache.path.read_text().strip().splitlines() if cache.path.is_file() else []
        on_disk.append(len(lines))

    backend.embed_passages(texts, on_progress=probe)
    assert on_disk[0] > 0, "nothing on disk after the first batch"
    assert on_disk[0] < 12, "whole corpus written at once, not incrementally"
    assert on_disk[-1] == 12


def test_query_role_differs_from_passage():
    backend = FakeBackend()
    assert backend.embed_query("aa") == expected("aa", is_query=True)
    assert backend.embed_passages(["aa"]).vectors[0] == expected("aa")


def test_empty_input_makes_no_calls():
    backend = FakeBackend()
    assert backend.embed_passages([]).vectors == []
    assert backend.calls == []


def test_vector_count_mismatch_raises_rather_than_misaligning():
    class Broken(FakeBackend):
        def _embed(self, texts, *, is_query: bool):
            return EmbeddingResult(vectors=[[1.0]], model=self.model)  # one, not len(texts)

    with pytest.raises(EmbeddingError, match="misalign"):
        Broken().embed_passages(["a", "b", "c"])


def test_response_sorted_by_index_not_position():
    """OpenAI-shaped responses carry `index`; order is not guaranteed."""
    from fhir_rag.embeddings import _parse_openai_shape

    payload = {"data": [
        {"index": 2, "embedding": [3.0]},
        {"index": 0, "embedding": [1.0]},
        {"index": 1, "embedding": [2.0]},
    ]}
    result = _parse_openai_shape(payload, "m")
    assert result.vectors == [[1.0], [2.0], [3.0]]


# -- caching --------------------------------------------------------------

def test_cache_prevents_second_api_call(tmp_path):
    cache = EmbeddingCache("fake-model", tmp_path)
    first = FakeBackend(cache=cache)
    first.embed_passages(["aa", "bb"])
    assert len(first.calls) == 1

    second = FakeBackend(cache=EmbeddingCache("fake-model", tmp_path))
    result = second.embed_passages(["aa", "bb"])
    assert second.calls == [], "cache hit should issue no request"
    assert result.vectors == [expected("aa"), expected("bb")]


def test_partial_cache_only_fetches_the_gap(tmp_path):
    cache = EmbeddingCache("fake-model", tmp_path)
    FakeBackend(cache=cache).embed_passages(["aa", "bb"])

    backend = FakeBackend(cache=EmbeddingCache("fake-model", tmp_path))
    result = backend.embed_passages(["aa", "bb", "cc", "dd"])
    fetched = [t for call in backend.calls for t in call]
    assert sorted(fetched) == ["cc", "dd"]
    assert result.vectors == [expected(t) for t in ["aa", "bb", "cc", "dd"]]


def test_cached_and_fresh_vectors_stay_in_input_order(tmp_path):
    """A gap in the middle must not shuffle the result."""
    cache = EmbeddingCache("fake-model", tmp_path)
    FakeBackend(cache=cache).embed_passages(["bb"])

    backend = FakeBackend(cache=EmbeddingCache("fake-model", tmp_path), concurrency=4)
    texts = ["aa", "bb", "cc", "dd", "ee"]
    result = backend.embed_passages(texts)
    assert result.vectors == [expected(t) for t in texts]


def test_cache_does_not_confuse_query_and_passage(tmp_path):
    cache = EmbeddingCache("fake-model", tmp_path)
    backend = FakeBackend(cache=cache)
    backend.embed_passages(["aa"])
    # Same text as a query must still hit the API, with the query role.
    assert backend.embed_query("aa") == expected("aa", is_query=True)


# -- registry -------------------------------------------------------------

def test_registry_lists_three_backends():
    assert available() == ["nvidia", "ollama", "openai"]


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(EmbeddingError, match="nvidia, ollama, openai"):
        get_embedder("not-a-backend")


@pytest.mark.parametrize("name,asym", [("nvidia", True), ("ollama", False), ("openai", False)])
def test_asymmetry_is_declared(name, asym):
    assert get_embedder(name).asymmetric is asym


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_embedder("ollama").api_key == ""


def test_missing_key_names_the_offline_alternative(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="ollama"):
        _ = get_embedder("nvidia").api_key
