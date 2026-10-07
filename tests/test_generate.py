"""Tests for grounded generation.

The dangerous failures in clinical RAG are not crashes, they are confident
answers. These tests pin the three behaviours that prevent one:

  * a citation pointing at a record that was never supplied is reported, not
    silently accepted
  * abstention is detected and surfaced, not buried in prose
  * an empty context never reaches the model, because a model asked a clinical
    question with no records will answer from training data

No API key required: a stub returns canned replies, including the malformed and
hallucinated ones a real model produces.
"""

from __future__ import annotations

import urllib.error

import pytest

from fhir_rag.generate import (
    ABSTAIN,
    Answer,
    GenerationError,
    GroqGenerator,
    build_prompt,
    extract_citations,
)
from fhir_rag.index import Hit
from fhir_rag.ingest import Chunk


def chunk(n: int, text: str | None = None, kind: str = "Observation") -> Chunk:
    return Chunk(
        patient_id="p1",
        patient_name="Test Patient",
        resource_type=kind,
        resource_id=f"{kind}-{n}",
        text=text or f"record number {n}",
    )


def hits(n: int) -> list[Hit]:
    return [
        Hit(chunk=chunk(i), score=0.9 - i * 0.01, rank=i + 1)
        for i in range(n)
    ]


class StubGenerator(GroqGenerator):
    def __init__(self, reply: str | Exception, **kw):
        kw.setdefault("api_key", "test-key-not-real")
        super().__init__(**kw)
        self.reply = reply
        self.prompts: list[str] = []

    def _chat(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, 100, 20


# -- citation extraction --------------------------------------------------

def test_extracts_valid_citations():
    valid, invalid = extract_citations("Yes [1] and also [3].", 3)
    assert valid == [1, 3]
    assert invalid == []


def test_reports_out_of_range_citations_rather_than_dropping_them():
    """Citing [7] when 3 records were supplied is a hallucination signal."""
    valid, invalid = extract_citations("Per [1] and [7].", 3)
    assert valid == [1]
    assert invalid == [7]


def test_zero_is_not_a_valid_citation():
    """Context is 1-based; [0] means the model miscounted."""
    valid, invalid = extract_citations("See [0].", 3)
    assert valid == []
    assert invalid == [0]


def test_duplicate_citations_counted_once():
    valid, _ = extract_citations("[2] confirms [2] again.", 3)
    assert valid == [2]


def test_no_citations_is_empty_not_an_error():
    assert extract_citations("The patient is fine.", 3) == ([], [])


# -- prompt ---------------------------------------------------------------

def test_prompt_numbers_records_from_one():
    prompt = build_prompt("q", hits(3))
    assert "[1]" in prompt and "[3]" in prompt
    assert "[0]" not in prompt


def test_prompt_demands_abstention_wording():
    assert ABSTAIN in build_prompt("q", hits(1))


def test_prompt_restricts_to_supplied_records():
    prompt = build_prompt("q", hits(1))
    assert "ONLY the numbered records" in prompt


# -- grounding ------------------------------------------------------------

def test_grounded_answer_maps_citations_to_chunk_ids():
    answer = StubGenerator("Yes, hypertension [1] with elevated reading [2].").answer(
        "high blood pressure?", hits(3)
    )
    assert answer.grounded
    assert answer.citations == [1, 2]
    assert answer.cited_chunks == ["Observation/Observation-0", "Observation/Observation-1"]
    assert answer.ok


def test_answer_without_citations_is_not_grounded():
    """Prose with no source is exactly what grounding is meant to exclude."""
    answer = StubGenerator("Yes, the patient has hypertension.").answer("q", hits(3))
    assert answer.grounded is False
    assert answer.abstained is False
    assert answer.ok is False


def test_hallucinated_citation_is_surfaced():
    answer = StubGenerator("Confirmed by [9].").answer("q", hits(3))
    assert answer.invalid_citations == [9]
    assert answer.citations == []
    assert answer.grounded is False, "an answer citing only a fake record is not grounded"


def test_mixed_valid_and_invalid_citations():
    answer = StubGenerator("Per [1] and [9].").answer("q", hits(3))
    assert answer.citations == [1]
    assert answer.invalid_citations == [9]
    assert answer.grounded is True  # one real source is still a real source
    assert answer.cited_chunks == ["Observation/Observation-0"]


# -- abstention -----------------------------------------------------------

def test_abstention_is_detected():
    answer = StubGenerator(ABSTAIN).answer("q", hits(3))
    assert answer.abstained
    assert answer.grounded is False
    assert answer.ok, "an honest refusal is a usable outcome"


def test_abstention_detected_case_insensitively():
    assert StubGenerator("Insufficient evidence").answer("q", hits(2)).abstained


def test_abstention_wins_over_stray_citations():
    """A refusal that also cites must not be counted as a grounded answer."""
    answer = StubGenerator(f"{ABSTAIN} (closest was [1])").answer("q", hits(3))
    assert answer.abstained
    assert answer.grounded is False


def test_empty_context_abstains_without_calling_the_model():
    """A clinical question with no records must never reach the model.

    Asked with an empty context, a model answers from training data — the exact
    failure grounding exists to prevent.
    """
    generator = StubGenerator("Yes, the patient has diabetes.")
    answer = generator.answer("does the patient have diabetes?", [])
    assert answer.abstained
    assert answer.grounded is False
    assert generator.prompts == [], "no API call for an empty context"


# -- transport failures ---------------------------------------------------

def test_network_error_returns_an_error_answer_not_an_exception():
    """One timeout must not abort a batch over 30 patients."""
    answer = StubGenerator(urllib.error.URLError("no route")).answer("q", hits(3))
    assert answer.error
    assert answer.ok is False
    assert answer.text == ""
    assert answer.grounded is False


def test_http_error_records_the_status_code():
    error = urllib.error.HTTPError("u", 429, "rate limited", {}, None)
    answer = StubGenerator(error).answer("q", hits(3))
    assert "429" in answer.error


def test_malformed_response_is_an_error_not_a_crash():
    answer = StubGenerator(KeyError("choices")).answer("q", hits(3))
    assert "unexpected response shape" in answer.error


def test_error_answer_is_never_mistaken_for_abstention():
    """Both have no usable text; only one is a safe outcome."""
    answer = StubGenerator(urllib.error.URLError("down")).answer("q", hits(3))
    assert answer.abstained is False
    assert answer.ok is False


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="GROQ_API_KEY"):
        _ = GroqGenerator().api_key


# -- bookkeeping ----------------------------------------------------------

def test_usage_and_model_are_recorded():
    answer = StubGenerator("Yes [1].").answer("q", hits(2))
    assert answer.prompt_tokens == 100
    assert answer.completion_tokens == 20
    assert answer.model == "qwen/qwen3.8-27b"
    assert answer.latency_s >= 0


def test_to_dict_is_json_serialisable():
    import json

    answer = StubGenerator("Yes [1].").answer("q", hits(2))
    json.dumps(answer.to_dict())


def test_citations_are_sorted_for_stable_reporting():
    answer = StubGenerator("Per [3], [1], [2].").answer("q", hits(3))
    assert answer.citations == [1, 2, 3]
