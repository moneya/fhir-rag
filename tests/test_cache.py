"""Tests for the embedding cache.

The cache's job is to be invisible: identical inputs must yield identical
vectors, and a cache hit must never attach the wrong vector to a text. Those
failures are silent and poison retrieval, so they are tested hardest.
"""

from __future__ import annotations

import json

import pytest

from fhir_rag.cache import EmbeddingCache, cache_key


def test_key_depends_on_text():
    a = cache_key("m", "one", is_query=False)
    b = cache_key("m", "two", is_query=False)
    assert a != b


def test_key_depends_on_model():
    a = cache_key("model-a", "text", is_query=False)
    b = cache_key("model-b", "text", is_query=False)
    assert a != b, "two models must not share cache entries"


def test_query_and_passage_keys_differ():
    """Asymmetric models return different vectors for the same text.

    Sharing one entry between query and passage roles would silently corrupt
    retrieval — the hardest kind of bug to notice, because results still look
    plausible.
    """
    q = cache_key("m", "same text", is_query=True)
    p = cache_key("m", "same text", is_query=False)
    assert q != p


def test_key_is_stable_across_calls():
    assert cache_key("m", "t", is_query=False) == cache_key("m", "t", is_query=False)


def test_roundtrip(tmp_path):
    cache = EmbeddingCache("test-model", tmp_path)
    cache.put_many([("hello", [0.1, 0.2]), ("world", [0.3, 0.4])], is_query=False)

    reopened = EmbeddingCache("test-model", tmp_path)
    assert reopened.get("hello", is_query=False) == [0.1, 0.2]
    assert reopened.get("world", is_query=False) == [0.3, 0.4]


def test_miss_returns_none(tmp_path):
    cache = EmbeddingCache("test-model", tmp_path)
    assert cache.get("never stored", is_query=False) is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_role_isolation_in_practice(tmp_path):
    """A passage entry must not satisfy a query lookup for the same text."""
    cache = EmbeddingCache("m", tmp_path)
    cache.put_many([("text", [1.0, 0.0])], is_query=False)
    assert cache.get("text", is_query=False) == [1.0, 0.0]
    assert cache.get("text", is_query=True) is None


def test_model_isolation_in_practice(tmp_path):
    """Two models write separate files, so vectors of different dims cannot mix."""
    a = EmbeddingCache("model-a", tmp_path)
    b = EmbeddingCache("model-b", tmp_path)
    a.put_many([("text", [1.0, 2.0])], is_query=False)
    assert a.path != b.path
    assert b.get("text", is_query=False) is None


def test_duplicate_put_does_not_duplicate_lines(tmp_path):
    cache = EmbeddingCache("m", tmp_path)
    cache.put_many([("same", [1.0])], is_query=False)
    cache.put_many([("same", [1.0])], is_query=False)
    lines = cache.path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_corrupt_trailing_line_is_skipped_not_fatal(tmp_path):
    """A process killed mid-write leaves one bad line.

    Thousands of good vectors above it are expensive to recompute, so the bad
    line is skipped rather than discarding the file.
    """
    cache = EmbeddingCache("m", tmp_path)
    cache.put_many([("good one", [1.0]), ("good two", [2.0])], is_query=False)
    with open(cache.path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "truncated", "vec')  # killed mid-write

    reopened = EmbeddingCache("m", tmp_path)
    assert reopened.load() == 2
    assert reopened.get("good one", is_query=False) == [1.0]


def test_missing_file_is_empty_not_an_error(tmp_path):
    cache = EmbeddingCache("never-written", tmp_path)
    assert cache.load() == 0
    assert cache.get("anything", is_query=False) is None


def test_stats_reports_counts(tmp_path):
    cache = EmbeddingCache("m", tmp_path)
    cache.put_many([("a", [1.0])], is_query=False)
    cache.get("a", is_query=False)
    cache.get("b", is_query=False)
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_model_name_with_slashes_is_a_safe_filename(tmp_path):
    """Model ids contain '/' — the cache path must not become a subdirectory."""
    cache = EmbeddingCache("nvidia/nemotron-3-embed-1b", tmp_path)
    cache.put_many([("a", [1.0])], is_query=False)
    assert cache.path.parent == tmp_path
    assert "/" not in cache.path.name


def test_file_is_valid_jsonl(tmp_path):
    cache = EmbeddingCache("m", tmp_path)
    cache.put_many([("a", [1.0]), ("b", [2.0])], is_query=False)
    for line in cache.path.read_text().strip().splitlines():
        record = json.loads(line)
        assert set(record) == {"key", "vector"}


def test_concurrent_writes_produce_valid_jsonl(tmp_path):
    """Batches are embedded in parallel, so put_many is called from N threads.

    Interleaved appends without a lock can split a line mid-write, which shows
    up later as a silently shorter cache.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache = EmbeddingCache("m", tmp_path)
    vectors = [(f"text-{i}", [float(i)] * 32) for i in range(200)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda pair: cache.put_many([pair], is_query=False),
            vectors,
        ))

    lines = cache.path.read_text().strip().splitlines()
    assert len(lines) == 200
    for line in lines:
        json.loads(line)  # raises if a write was torn

    reopened = EmbeddingCache("m", tmp_path)
    assert reopened.load() == 200
    assert reopened.get("text-137", is_query=False) == [137.0] * 32
