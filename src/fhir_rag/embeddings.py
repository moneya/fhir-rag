"""Embedding backends.

Three providers behind one interface, chosen by config rather than by edit:

  nvidia    nemotron-3-embed-1b via integrate.api.nvidia.com. Asymmetric:
            queries and passages are encoded differently (`input_type`).
  ollama    nomic-embed-text on localhost. No key, fully offline.
  openai    text-embedding-3-small. Symmetric.

The abstraction exists because `input_type` is NOT part of the OpenAI embeddings
spec. "OpenAI-compatible" is only half-true for embeddings, so a single hardcoded
client would either drop the parameter (losing asymmetric retrieval) or send it
to providers that reject it. Each backend declares whether it is asymmetric and
the caller always states intent — `embed_query` vs `embed_passages`.

Why asymmetric matters: retrieval is a relevance problem, not a similarity one.
"Which diabetes medication is the patient taking?" should match "Patient was
prescribed metformin", two sentences that are not similar as text. Models trained
with separate query and passage encoders handle that directly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# Some providers sit behind a CDN that rejects urllib's default agent.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


class EmbeddingError(RuntimeError):
    """A backend could not produce embeddings."""


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    tokens: int | None = None

    @property
    def dims(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


class EmbeddingBackend(ABC):
    name: str = ""
    model: str = ""
    asymmetric: bool = False
    requires_key: bool = True
    env_key: str = ""
    batch_limit: int = 64

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache: Any | None = None,
        concurrency: int = 1,
    ) -> None:
        self.model = model or self.model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = cache
        self.concurrency = max(1, int(concurrency))

    @property
    def default_base_url(self) -> str:
        raise NotImplementedError

    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get(self.env_key, "").strip() if self.env_key else ""
        if not key and self.requires_key:
            raise EmbeddingError(
                f"{self.env_key} is not set. Put it in .env, or use "
                f"`--embedder ollama` to run locally with no key."
            )
        return key

    @abstractmethod
    def _embed(self, texts: Sequence[str], *, is_query: bool) -> EmbeddingResult: ...

    def embed_passages(
        self,
        texts: Sequence[str],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """Encode documents for storage."""
        return self._batched(texts, is_query=False, on_progress=on_progress)

    def embed_query(self, text: str) -> list[float]:
        """Encode a question for search."""
        return self._batched([text], is_query=True).vectors[0]

    def _batched(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """Embed many texts: cache first, then remaining batches in parallel.

        Most of the wall clock on a hosted endpoint is round-trip latency, not
        compute, so batches are issued concurrently. Results are reassembled by
        original position — a reordered batch would attach vectors to the wrong
        chunks, which is silent and ruins retrieval rather than failing loudly.
        """
        if not texts:
            return EmbeddingResult(vectors=[], model=self.model, tokens=0)

        slots: list[list[float] | None] = [None] * len(texts)

        # 1. cache
        pending: list[int] = []
        if self.cache is not None:
            for i, text in enumerate(texts):
                hit = self.cache.get(text, is_query=is_query)
                if hit is None:
                    pending.append(i)
                else:
                    slots[i] = hit
        else:
            pending = list(range(len(texts)))

        tokens = 0
        if pending:
            batches = [
                pending[s:s + self.batch_limit]
                for s in range(0, len(pending), self.batch_limit)
            ]

            def run(indices: list[int]) -> tuple[list[int], EmbeddingResult]:
                payload = [texts[i] for i in indices]
                return indices, self._embed(payload, is_query=is_query)

            def absorb(indices: list[int], result: EmbeddingResult) -> int:
                if len(result.vectors) != len(indices):
                    raise EmbeddingError(
                        f"{self.name} returned {len(result.vectors)} vectors for "
                        f"{len(indices)} inputs — refusing to misalign embeddings"
                    )
                pairs = []
                for i, vector in zip(indices, result.vectors):
                    slots[i] = vector
                    pairs.append((texts[i], vector))
                # Flush per batch as it lands, not once at the end: a corpus-sized
                # run takes minutes, and a crash or Ctrl-C partway through should
                # cost the in-flight batch, not every vector bought so far.
                if self.cache is not None:
                    self.cache.put_many(pairs, is_query=is_query)
                return result.tokens or 0

            if self.concurrency > 1 and len(batches) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = [pool.submit(run, b) for b in batches]
                    for future in as_completed(futures):
                        indices, result = future.result()
                        tokens += absorb(indices, result)
                        if on_progress is not None:
                            done = sum(1 for s in slots if s is not None)
                            on_progress(done, len(texts))
            else:
                for batch in batches:
                    indices, result = run(batch)
                    tokens += absorb(indices, result)
                    if on_progress is not None:
                        done = sum(1 for s in slots if s is not None)
                        on_progress(done, len(texts))

        missing = [i for i, v in enumerate(slots) if v is None]
        if missing:
            raise EmbeddingError(f"{self.name}: no vector produced for {len(missing)} input(s)")

        return EmbeddingResult(
            vectors=[v for v in slots if v is not None],
            model=self.model,
            tokens=tokens or None,
        )

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """POST with retry on transient failures. Never logs the key."""
        payload = json.dumps(body).encode()
        last = ""
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf-8", "replace")
                last = f"{exc.code}: {_short(detail)}"
                # 4xx other than rate limiting will not change on retry.
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise EmbeddingError(f"{self.name} {self.model} {last}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        raise EmbeddingError(f"{self.name} {self.model} failed after {self.max_retries} tries: {last}")


def _short(detail: str) -> str:
    """Collapse an HTML error page into one line."""
    stripped = detail.strip()
    if stripped[:1] == "<" or "<html" in stripped[:200].lower():
        return "HTML response — check the base URL"
    try:
        parsed = json.loads(stripped)
        for key in ("detail", "message", "error"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value[:200]
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"][:200]
    except (json.JSONDecodeError, AttributeError):
        pass
    return " ".join(stripped.split())[:200]


class NvidiaEmbedder(EmbeddingBackend):
    """NVIDIA NIM. Asymmetric, 2048 dims.

    `input_type` is required and is not an OpenAI parameter: "query" and
    "passage" produce different vectors for the same text, which is the whole
    reason to prefer this backend for retrieval.
    """

    name = "nvidia"
    model = "nvidia/nemotron-3-embed-1b"
    asymmetric = True
    env_key = "NVIDIA_API_KEY"
    batch_limit = 32

    @property
    def default_base_url(self) -> str:
        return os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> EmbeddingResult:
        data = self._post(
            f"{self.base_url}/embeddings",
            {
                "model": self.model,
                "input": list(texts),
                "input_type": "query" if is_query else "passage",
                "encoding_format": "float",
            },
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _UA,
            },
        )
        return _parse_openai_shape(data, self.model)


class OllamaEmbedder(EmbeddingBackend):
    """Local nomic-embed-text. No key, no network egress.

    Symmetric: the same text yields the same vector whether it is a query or a
    passage, so `is_query` is ignored. Kept as the default for offline use and
    so the test suite never needs credentials.
    """

    name = "ollama"
    model = "nomic-embed-text"
    asymmetric = False
    requires_key = False
    batch_limit = 16

    @property
    def default_base_url(self) -> str:
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> EmbeddingResult:
        data = self._post(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": list(texts)},
            {"Content-Type": "application/json", "Accept": "application/json"},
        )
        return _parse_openai_shape(data, self.model)


class OpenAIEmbedder(EmbeddingBackend):
    """OpenAI text-embedding-3-small. Symmetric, 1536 dims."""

    name = "openai"
    model = "text-embedding-3-small"
    asymmetric = False
    env_key = "OPENAI_API_KEY"
    batch_limit = 64

    @property
    def default_base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> EmbeddingResult:
        data = self._post(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": list(texts), "encoding_format": "float"},
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        return _parse_openai_shape(data, self.model)


def _parse_openai_shape(data: dict[str, Any], model: str) -> EmbeddingResult:
    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        raise EmbeddingError(f"{model}: response had no embedding data")
    # Order is guaranteed by `index`, not by position, so sort defensively:
    # a reordered batch would silently attach vectors to the wrong chunks.
    try:
        rows = sorted(rows, key=lambda r: r.get("index", 0))
    except TypeError:
        pass
    vectors = [row["embedding"] for row in rows]
    tokens = (data.get("usage") or {}).get("total_tokens")
    return EmbeddingResult(vectors=vectors, model=model, tokens=tokens)


_BACKENDS: dict[str, type[EmbeddingBackend]] = {
    "nvidia": NvidiaEmbedder,
    "ollama": OllamaEmbedder,
    "openai": OpenAIEmbedder,
}


def available() -> list[str]:
    return sorted(_BACKENDS)


def get_embedder(name: str, **options: Any) -> EmbeddingBackend:
    key = (name or "").strip().lower()
    if key not in _BACKENDS:
        raise EmbeddingError(f"unknown embedder {name!r}. Available: {', '.join(available())}")
    return _BACKENDS[key](**options)


__all__ = [
    "EmbeddingBackend",
    "EmbeddingError",
    "EmbeddingResult",
    "NvidiaEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "available",
    "get_embedder",
]
