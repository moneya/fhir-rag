"""LLM listwise reranking.

No reranker model is callable on this NVIDIA account (every `nv-rerankqa` id
returns 404), so reranking is done with a cheap instruct model over the
candidates an embedding search already found. Groq serves this in ~0.5s for 2
output tokens, which is fast enough to sit in the retrieval path.

The prompt shape was chosen by measurement, not taste. Asking for *"only the
passage numbers that answer the question"* collapses a seven-vaccination answer
to a single line — 1/7 recall. Asking the model to **rank all candidates and
return the top k** keeps 7/7 and orders them correctly. So the instruction is
always "rank", never "filter": deciding how many rows an answer needs is the
retriever's job via `shape`, not the reranker's.

Two rules this module will not bend:

**It never invents or drops candidates.** The output is a permutation of the
input. Ids outside range, duplicates and hallucinated numbers are discarded, and
any candidate the model omitted is appended in its original embedding order. A
reranker that silently loses documents is worse than no reranker.

**It fails closed to embedding order.** A timeout, a refusal, a prose answer or
an unparseable reply returns the input ranking unchanged rather than an empty or
partial list. Retrieval degrading to "merely unreranked" is acceptable; retrieval
returning nothing because an LLM had an off day is not.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from .index import Hit

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

# Cloudflare in front of Groq rejects urllib's default user agent with 403.
_INSTRUCTION = (
    "Rank ALL passages by how well each answers the question, most relevant "
    "first. Return the top {k} passage numbers as a comma-separated list. "
    "Every number must appear exactly once. Output only numbers and commas."
)


class RerankError(RuntimeError):
    """The reranker could not be reached or configured."""


@dataclass
class RerankOutcome:
    """Result of one rerank, including why it may have fallen back."""

    hits: list[Hit]
    reranked: bool
    reason: str = ""
    latency_s: float = 0.0
    tokens: int = 0
    raw: str = ""


def parse_ranking(text: str, n: int) -> list[int]:
    """Extract candidate indices from a model reply.

    Deliberately permissive about format — models wrap lists in prose, brackets
    or newlines — and strict about content: only in-range, first-occurrence ids
    survive.
    """
    seen: set[int] = set()
    order: list[int] = []
    for token in re.findall(r"\d+", text or ""):
        try:
            value = int(token)
        except ValueError:  # pragma: no cover - re guarantees digits
            continue
        if 0 <= value < n and value not in seen:
            seen.add(value)
            order.append(value)
    return order


def complete_ranking(order: Sequence[int], n: int) -> list[int]:
    """Make `order` a full permutation of range(n), preserving omitted items.

    Anything the model left out keeps its embedding rank and goes after what the
    model did rank. This is what guarantees no candidate is ever lost.
    """
    seen = set(order)
    return list(order) + [i for i in range(n) if i not in seen]


class GroqReranker:
    """Listwise reranker over an OpenAI-compatible chat endpoint."""

    name = "groq"
    default_model = "qwen/qwen3.8-27b"
    env_key = "GROQ_API_KEY"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_candidates: int = 40,
        cache: Any | None = None,
    ) -> None:
        self.model = model or self.default_model
        self.base_url = (
            base_url
            or os.environ.get("GROQ_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.cache = cache

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get(self.env_key, "").strip()
        if not key:
            raise RerankError(
                f"{self.env_key} is not set. Put it in .env, or retrieve with "
                f"rerank disabled."
            )
        return key

    def _chat(self, prompt: str) -> tuple[str, int]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # Enough for ~40 two-digit ids; reasoning models need headroom or
            # they spend the whole budget thinking and return an empty string.
            "max_tokens": 500,
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _UA,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        message = payload["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        tokens = (payload.get("usage") or {}).get("total_tokens", 0)
        return content, tokens

    def rerank(self, question: str, hits: Sequence[Hit], *, k: int) -> RerankOutcome:
        """Reorder `hits` by relevance to `question`.

        Returns the original order with `reranked=False` and a reason whenever
        the model cannot be used or understood.
        """
        hits = list(hits)
        if len(hits) <= 1:
            return RerankOutcome(hits=hits, reranked=False, reason="nothing to reorder")

        # Only the window the model sees is reordered; the tail keeps its
        # embedding order and is appended, so long candidate lists stay cheap.
        window = hits[: self.max_candidates]
        tail = hits[self.max_candidates :]

        numbered = "\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(window))
        prompt = (
            f"Question: {question}\n\nPassages:\n{numbered}\n\n"
            + _INSTRUCTION.format(k=min(k, len(window)))
        )

        cached = None
        if self.cache is not None:
            cached = self.cache.get(prompt, is_query=True)

        started = time.time()
        tokens = 0
        if cached is not None:
            reply = _decode_cached(cached)
        else:
            try:
                reply, tokens = self._chat(prompt)
            except RerankError:
                raise
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                return RerankOutcome(
                    hits=hits,
                    reranked=False,
                    reason=f"{type(exc).__name__}: falling back to embedding order",
                    latency_s=time.time() - started,
                )
            except (KeyError, json.JSONDecodeError) as exc:
                return RerankOutcome(
                    hits=hits,
                    reranked=False,
                    reason=f"unexpected response shape ({exc})",
                    latency_s=time.time() - started,
                )
            if self.cache is not None:
                self.cache.put_many([(prompt, _encode_reply(reply))], is_query=True)

        latency = time.time() - started
        order = parse_ranking(reply, len(window))
        if not order:
            return RerankOutcome(
                hits=hits,
                reranked=False,
                reason="no usable ids in reply",
                latency_s=latency,
                tokens=tokens,
                raw=reply[:200],
            )

        full = complete_ranking(order, len(window))
        reordered = [window[i] for i in full] + tail
        renumbered = [
            Hit(chunk=h.chunk, score=h.score, rank=rank)
            for rank, h in enumerate(reordered, start=1)
        ]
        return RerankOutcome(
            hits=renumbered,
            reranked=True,
            latency_s=latency,
            tokens=tokens,
            raw=reply[:200],
        )


def _encode_reply(reply: str) -> list[float]:
    """Store a reply in the float-vector cache as code points.

    Reusing EmbeddingCache keeps one cache format in the project; a reranked
    ordering is as deterministic at temperature 0 as an embedding is.
    """
    return [float(ord(c)) for c in reply]


def _decode_cached(vector: Sequence[float]) -> str:
    return "".join(chr(int(x)) for x in vector)


__all__ = [
    "GroqReranker",
    "RerankError",
    "RerankOutcome",
    "complete_ranking",
    "parse_ranking",
]
