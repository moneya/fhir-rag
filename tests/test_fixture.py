"""Tests for the committed CI fixture and int8 quantization.

The fixture exists so the retrieval gate runs on a fresh clone with no API key.
Two things must hold or it is worse than useless:

  * quantization round-trips — a fixture that does not survive save/load would
    surface later as a confusing gate failure, not an obvious one
  * the fixture still separates the ranking policies — a gate whose fixture
    scores the same under every policy passes real regressions

The second is the one worth having. It is checked here against the real
`retrieve()` code path, not a mock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fhir_rag.diversify import cap_per_resource_type
from fhir_rag.index import VectorIndex
from fhir_rag.ingest import Chunk
from fhir_rag.queries import build_queries
from fhir_rag.retrieve import retrieve

FIXTURES = Path(__file__).parent / "fixtures"
INDEX = FIXTURES / "index_fixture.json"
QUERIES = FIXTURES / "query_vectors.json"


@pytest.fixture(scope="module")
def index() -> VectorIndex:
    return VectorIndex.load(INDEX)


@pytest.fixture(scope="module")
def query_vectors() -> dict:
    return json.loads(QUERIES.read_text(encoding="utf-8"))


# -- the fixture ships and is intact --------------------------------------

def test_fixture_files_are_committed():
    assert INDEX.exists(), "fixture index missing — the CI gate cannot run"
    assert QUERIES.exists(), "fixture query vectors missing"


def test_fixture_stays_small_enough_to_commit():
    """If this grows past a few MB it does not belong in git."""
    assert INDEX.stat().st_size < 3_000_000


def test_fixture_declares_its_encoding():
    """A gate reading quantized vectors must be able to tell that it is."""
    payload = json.loads(INDEX.read_text(encoding="utf-8"))
    assert payload["encoding"] == "int8-base64"


def test_fixture_has_expected_shape(index):
    assert len(index) == 328
    assert index.dims == 2048
    assert index.model == "nvidia/nemotron-3-embed-1b"


def test_fixture_is_one_patient(index):
    assert len({c.patient_id for c in index.chunks}) == 1


def test_fixture_retains_both_pathology_queries(index):
    """Without these two the fixture gates nothing interesting."""
    pid = index.chunks[0].patient_id
    covered = {
        q.id for q in build_queries(pid)
        if any(q.relevant(c) for c in index.chunks)
    }
    assert {"prediabetes", "hypertension-diagnosis"} <= covered


def test_fixture_keeps_measurements_outnumbering_diagnoses(index):
    """The pathology is that Observations drown Conditions — preserve the ratio."""
    kinds = [c.resource_type for c in index.chunks]
    assert kinds.count("Observation") > 4 * kinds.count("Condition")


# -- quantization ---------------------------------------------------------

def test_quantized_round_trip_preserves_ranking(tmp_path):
    """int8 is lossy; it must not be lossy enough to reorder results."""
    chunks = [
        Chunk(
            patient_id="p1", patient_name="T", resource_type="Observation",
            resource_id=f"o{i}", text=f"observation {i}",
        )
        for i in range(12)
    ]
    # Distinct, well-separated directions so order is unambiguous.
    vectors = [[1.0 - i * 0.05, i * 0.03, 0.1] for i in range(12)]

    exact = VectorIndex(model="m")
    exact.add(chunks, vectors)
    path = tmp_path / "q.json"
    exact.save(path, quantize=True)
    loaded = VectorIndex.load(path)

    query = [1.0, 0.0, 0.0]
    before = [h.chunk.chunk_id for h in exact.search(query, k=12)]
    after = [h.chunk.chunk_id for h in loaded.search(query, k=12)]
    assert before == after


def test_quantization_is_opt_in(tmp_path):
    """The default must stay exact: lossiness is never silent."""
    chunk = Chunk(
        patient_id="p", patient_name="n", resource_type="Observation",
        resource_id="o1", text="t",
    )
    index = VectorIndex(model="m")
    index.add([chunk], [[0.6, 0.8, 0.0]])

    plain = tmp_path / "plain.json"
    index.save(plain)
    payload = json.loads(plain.read_text(encoding="utf-8"))
    assert "encoding" not in payload
    assert isinstance(payload["vectors"][0], list)


def test_quantized_file_is_much_smaller(tmp_path):
    chunks = [
        Chunk(
            patient_id="p", patient_name="n", resource_type="Observation",
            resource_id=f"o{i}", text="t",
        )
        for i in range(40)
    ]
    vectors = [[(i * 7 % 100) / 100 - 0.5 for _ in range(512)] for i in range(40)]
    index = VectorIndex(model="m")
    index.add(chunks, vectors)

    plain, quant = tmp_path / "p.json", tmp_path / "q.json"
    index.save(plain)
    index.save(quant, quantize=True)
    assert quant.stat().st_size * 4 < plain.stat().st_size


def test_negative_dimensions_survive_quantization(tmp_path):
    """Signed bytes are the easy thing to get wrong; -0.5 must stay negative."""
    chunk = Chunk(
        patient_id="p", patient_name="n", resource_type="Observation",
        resource_id="o1", text="t",
    )
    index = VectorIndex(model="m")
    index.add([chunk], [[-0.8, 0.6, 0.0]])
    path = tmp_path / "q.json"
    index.save(path, quantize=True)

    restored = VectorIndex.load(path).vectors[0]
    assert restored[0] < 0, "sign lost in quantization"
    assert restored[1] > 0
    assert abs(restored[0] - (-0.8)) < 0.02


# -- the gate has teeth ---------------------------------------------------

def test_fixture_separates_shape_aware_from_flat(index, query_vectors):
    """The regression the gate exists to catch must be visible in the fixture.

    Flat top-k is the pre-shape-aware policy. If the fixture scored the same
    under both, the committed gate would pass a real regression.
    """
    pid = index.chunks[0].patient_id
    shaped_total = flat_total = count = 0.0

    for query in build_queries(pid):
        relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
        if not relevant or query.id not in query_vectors:
            continue
        vector = query_vectors[query.id]
        shaped = retrieve(
            index, vector, k=10, patient_id=pid,
            resource_types=query.resource_types, shape=query.shape,
        )
        flat = index.search(
            vector, k=10, patient_id=pid, resource_types=query.resource_types
        )
        shaped_total += len([h for h in shaped if h.chunk.chunk_id in relevant]) / len(relevant)
        flat_total += len([h for h in flat if h.chunk.chunk_id in relevant]) / len(relevant)
        count += 1

    assert count >= 5
    assert shaped_total / count > flat_total / count + 0.1, (
        "fixture does not distinguish shape-aware retrieval from flat top-k, "
        "so a gate built on it is decorative"
    )


def test_fixture_shows_prediabetes_failing_under_flat_retrieval(index, query_vectors):
    """The original 0.000 failure, frozen so it cannot silently return."""
    pid = index.chunks[0].patient_id
    query = next(q for q in build_queries(pid) if q.id == "prediabetes")
    relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
    assert relevant

    flat = index.search(
        query_vectors["prediabetes"], k=10, patient_id=pid,
        resource_types=query.resource_types,
    )
    assert not [h for h in flat if h.chunk.chunk_id in relevant], (
        "prediabetes no longer fails under flat retrieval — the fixture has "
        "lost the pathology it was chosen for"
    )

    shaped = retrieve(
        index, query_vectors["prediabetes"], k=10, patient_id=pid,
        resource_types=query.resource_types, shape=query.shape,
    )
    assert [h for h in shaped if h.chunk.chunk_id in relevant]


def test_fixture_shows_dense_queries_breaking_under_a_global_cap(index, query_vectors):
    """The opposite regression: capping every type ruins list answers."""
    pid = index.chunks[0].patient_id
    query = next(q for q in build_queries(pid) if q.id == "flu-vaccination")
    relevant = {c.chunk_id for c in index.chunks if query.relevant(c)}
    assert len(relevant) > 2

    wide = index.search(
        query_vectors["flu-vaccination"], k=60, patient_id=pid,
        resource_types=query.resource_types,
    )
    capped = cap_per_resource_type(wide, cap=2, k=10)
    capped_recall = len([h for h in capped if h.chunk.chunk_id in relevant]) / len(relevant)

    shaped = retrieve(
        index, query_vectors["flu-vaccination"], k=10, patient_id=pid,
        resource_types=query.resource_types, shape=query.shape,
    )
    shaped_recall = len([h for h in shaped if h.chunk.chunk_id in relevant]) / len(relevant)

    assert shaped_recall > capped_recall
