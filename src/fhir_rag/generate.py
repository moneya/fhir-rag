"""Grounded answer generation over retrieved FHIR chunks.

Retrieval without generation is a search box; generation without grounding is a
liability. This module answers clinical questions from retrieved records only,
and treats every claim the model makes about its own sources as unverified until
checked in code.

Three guarantees, in descending order of how much damage their absence causes:

**Citations are verified, never trusted.** The model is asked to cite records as
`[n]`. Those ids are then checked against the context that was actually sent: out
of range ids are stripped and reported in `invalid_citations`. A plausible answer
citing record [7] when six were supplied is the single most dangerous output a
clinical RAG system can produce, because it looks sourced.

**Abstention is a first-class answer.** When the records do not contain the
answer, the model must reply `INSUFFICIENT EVIDENCE`, which is surfaced as
`abstained=True` rather than buried in prose. Measured on a deliberate
distractor — urine glucose present, question asked about blood glucose — the
model abstained instead of conflating the two. A system that always answers is
not more useful, it is less trustworthy.

**An answer with no valid citation is not grounded.** `grounded` is False when a
non-abstaining answer cites nothing verifiable, so callers can reject it without
re-reading the prose.

A measured caveat on `grounded`, worth stating plainly: it certifies that cited
records exist and were supplied, NOT that the conclusion drawn from them is
correct. In one benchmark run the model was asked about impaired glucose
regulation, did not retrieve the patient's `Prediabetes` diagnosis, and answered
from two HbA1c results of 5.82% and 6.03% instead — which is the prediabetic
range, so the conclusion was clinically right while the code-derived ground truth
scored it a miss. Grounding is a provenance check, not a correctness oracle, and
conflating the two is how RAG systems get oversold.

The prompt deliberately does not ask for JSON. Clinical text contains brackets,
units and quotes that break naive JSON emission, and a malformed envelope would
discard an otherwise correct answer. Inline `[n]` citations survive any prose.
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

ABSTAIN = "INSUFFICIENT EVIDENCE"

_SYSTEM = (
    "You answer questions about a patient's medical record. You use only the "
    "numbered records provided, never outside knowledge, and you never guess."
)

_INSTRUCTION = (
    "Answer the question using ONLY the numbered records above. Cite every "
    "record you rely on inline as [n]. If the records do not contain the "
    f"answer, reply exactly: {ABSTAIN}. Be concise and clinical."
)


class GenerationError(RuntimeError):
    """The generator could not be reached or configured."""


@dataclass
class Answer:
    """A generated answer plus everything needed to audit it."""

    text: str
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    cited_chunks: list[str] = field(default_factory=list)
    abstained: bool = False
    grounded: bool = False
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the answer is usable: either grounded, or an honest refusal."""
        return not self.error and (self.abstained or self.grounded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "cited_chunks": self.cited_chunks,
            "abstained": self.abstained,
            "grounded": self.grounded,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "error": self.error,
        }


def extract_citations(text: str, n_context: int) -> tuple[list[int], list[int]]:
    """Split `[n]` references into (valid, invalid) 1-based ids.

    Invalid ids are returned rather than silently dropped: a model citing a
    record that was never supplied is a hallucination signal worth reporting,
    not a formatting quirk worth hiding.
    """
    seen: set[int] = set()
    valid: list[int] = []
    invalid: list[int] = []
    for match in re.findall(r"\[(\d+)\]", text or ""):
        value = int(match)
        if value in seen:
            continue
        seen.add(value)
        if 1 <= value <= n_context:
            valid.append(value)
        else:
            invalid.append(value)
    return sorted(valid), sorted(invalid)


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    records = "\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
    return f"Records:\n{records}\n\nQuestion: {question}\n\n{_INSTRUCTION}"


class GroqGenerator:
    """Answer generation over an OpenAI-compatible chat endpoint."""

    name = "groq"
    default_model = "qwen/qwen3.8-27b"
    env_key = "GROQ_API_KEY"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 45.0,
        max_tokens: int = 500,
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
        self.max_tokens = max_tokens
        self.cache = cache

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get(self.env_key, "").strip()
        if not key:
            raise GenerationError(
                f"{self.env_key} is not set. Put it in .env — generation needs a "
                f"model, unlike ingestion and offline embedding."
            )
        return key

    def _chat(self, prompt: str) -> tuple[str, int, int]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
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
        usage = payload.get("usage") or {}
        return (
            (message.get("content") or "").strip(),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    def answer(self, question: str, hits: Sequence[Hit]) -> Answer:
        """Answer `question` from `hits` only.

        Never raises for transport problems: a failed call returns an Answer with
        `error` set and `ok` False, so a caller loops over patients without one
        timeout aborting the batch.
        """
        hits = list(hits)
        if not hits:
            # No retrieval means no evidence. Asking the model anyway invites it
            # to answer from training data, which is exactly the failure mode
            # grounding exists to prevent.
            return Answer(
                text=ABSTAIN,
                abstained=True,
                grounded=False,
                model=self.model,
            )

        prompt = build_prompt(question, hits)
        started = time.time()
        try:
            reply, prompt_tokens, completion_tokens = self._chat(prompt)
        except GenerationError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                detail = f" {exc.code}"
            return Answer(
                text="",
                error=f"{type(exc).__name__}{detail}",
                latency_s=time.time() - started,
                model=self.model,
            )
        except (KeyError, json.JSONDecodeError) as exc:
            return Answer(
                text="",
                error=f"unexpected response shape ({exc})",
                latency_s=time.time() - started,
                model=self.model,
            )

        latency = time.time() - started
        abstained = ABSTAIN in reply.upper()
        valid, invalid = extract_citations(reply, len(hits))

        return Answer(
            text=reply,
            citations=valid,
            invalid_citations=invalid,
            cited_chunks=[hits[i - 1].chunk.chunk_id for i in valid],
            abstained=abstained,
            # An answer is grounded only if it points at records that exist.
            grounded=bool(valid) and not abstained,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )


__all__ = [
    "ABSTAIN",
    "Answer",
    "GenerationError",
    "GroqGenerator",
    "build_prompt",
    "extract_citations",
]
