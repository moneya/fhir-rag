"""Disk cache for embeddings.

An embedding is a pure function of (model, input_type, text), so it never needs
computing twice. That matters more here than it sounds: the first full-corpus
run took ~4 hours of wall clock against a free-tier endpoint, almost all of it
network latency. With a cache, a re-run after changing retrieval or the query
set costs nothing, which is what makes iterating on recall practical.

Stored as one JSONL file per model: append-only, survives interruption, and
readable with `head`. A corrupt trailing line (killed mid-write) is skipped
rather than discarding the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_DIR = Path(os.environ.get("FHIR_RAG_CACHE", ".cache/embeddings"))


def cache_key(model: str, text: str, *, is_query: bool) -> str:
    """Stable id for one embedding.

    `is_query` is part of the key because asymmetric models return a different
    vector for the same text depending on it — sharing a cache entry between
    the two would silently corrupt retrieval.
    """
    role = "q" if is_query else "p"
    digest = hashlib.sha256(f"{model}\x00{role}\x00{text}".encode()).hexdigest()
    return f"{role}:{digest[:40]}"


def _safe_name(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in model)


class EmbeddingCache:
    """Append-only JSONL cache, loaded into memory on open."""

    def __init__(self, model: str, directory: Path | str | None = None) -> None:
        self.model = model
        self.directory = Path(directory) if directory else DEFAULT_DIR
        self.path = self.directory / f"{_safe_name(model)}.jsonl"
        self._entries: dict[str, list[float]] = {}
        self._loaded = False
        self.hits = 0
        self.misses = 0
        # Batches are embedded concurrently, so writes arrive from several
        # threads. Without this, interleaved appends can corrupt a JSONL line.
        self._lock = threading.Lock()

    def load(self) -> int:
        """Read the cache file. Returns how many vectors were loaded."""
        if self._loaded:
            return len(self._entries)
        self._loaded = True
        if not self.path.is_file():
            return 0
        skipped = 0
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._entries[record["key"]] = record["vector"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A process killed mid-write leaves one bad line; the
                    # thousands of good ones above it are still worth keeping.
                    skipped += 1
        if skipped:
            self._truncated_lines = skipped
        return len(self._entries)

    def get(self, text: str, *, is_query: bool) -> list[float] | None:
        self.load()
        key = cache_key(self.model, text, is_query=is_query)
        vector = self._entries.get(key)
        if vector is None:
            self.misses += 1
        else:
            self.hits += 1
        return vector

    def put_many(self, items: Sequence[tuple[str, list[float]]], *, is_query: bool) -> None:
        """Append new vectors. Thread-safe and durable (fsync before returning)."""
        self.load()
        with self._lock:
            fresh = []
            for text, vector in items:
                key = cache_key(self.model, text, is_query=is_query)
                if key in self._entries:
                    continue
                self._entries[key] = vector
                fresh.append({"key": key, "vector": vector})
            if not fresh:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                for record in fresh:
                    fh.write(json.dumps(record) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def stats(self) -> dict[str, int | str]:
        self.load()
        return {
            "model": self.model,
            "path": str(self.path),
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
        }


__all__ = ["DEFAULT_DIR", "EmbeddingCache", "cache_key"]
