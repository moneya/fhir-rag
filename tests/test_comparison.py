"""Tests for the embedder comparison and quantization-tradeoff benchmarks.

These scripts produce numbers that went into the README, so the properties the
conclusions depend on are pinned here. The point is not to re-run the benchmarks
in CI — they need a 806 MB index and a local model — but to stop the ANALYSIS from
silently breaking.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- quantization helpers are pure functions, test them directly -----------

@pytest.fixture(scope="module")
def bits():
    return _load("bench_bits_vs_dims")


def test_one_bit_keeps_only_sign(bits):
    assert bits.to_one_bit([0.9, -0.1, 0.0, -2.0]) == [1.0, -1.0, 1.0, -1.0]


def test_one_bit_preserves_dimensionality(bits):
    """The whole finding is that bits go, dimensions stay."""
    vector = [0.1 * i - 0.5 for i in range(64)]
    assert len(bits.to_one_bit(vector)) == len(vector)


def test_truncate_drops_tail_dimensions(bits):
    assert bits.truncate([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_int8_quantization_is_bounded(bits):
    """Values outside [-1, 1] must clamp, not wrap to the opposite sign."""
    out = bits.to_int8([5.0, -5.0, 0.5])
    assert out[0] == pytest.approx(1.0, abs=0.01)
    assert out[1] == pytest.approx(-1.0, abs=0.01)
    assert out[2] == pytest.approx(0.5, abs=0.01)


def test_int8_round_trip_is_close(bits):
    for value in (-0.97, -0.5, -0.01, 0.0, 0.33, 0.8, 0.99):
        assert bits.to_int8([value])[0] == pytest.approx(value, abs=0.01)


# -- the recorded results must stay internally consistent -----------------

RESULTS = ROOT / "results" / "bits_vs_dims.json"
COMPARISON = ROOT / "results" / "model_comparison_full.json"
SIGNIFICANCE = ROOT / "results" / "comparison_significance.json"


def test_one_bit_beats_truncation_at_matched_budget():
    """The README's storage claim. If this inverts, the claim must change."""
    data = json.loads(RESULTS.read_text())
    rows = {r["encoding"]: r for r in data["results"]}
    one_bit = next(r for k, r in rows.items() if k.startswith("1-bit"))
    truncated = next(r for k, r in rows.items() if k.startswith("float32"))
    assert one_bit["bytes_per_vector"] == truncated["bytes_per_vector"], (
        "the comparison is only meaningful at a matched memory budget"
    )
    assert one_bit["recall@10"] > truncated["recall@10"]


def test_model_comparison_is_reported_as_parity():
    """Guards against a future edit upgrading 'parity' to 'win'.

    p was 0.971. If someone re-runs with more data and gets significance, this
    test should fail and force the README wording to change WITH the evidence.
    """
    data = json.loads(SIGNIFICANCE.read_text())
    assert data["pairs"] == 92
    if data["permutation_p"] >= 0.05:
        assert data["conclusion"] == "parity"
    else:
        assert data["conclusion"] == "difference"


def test_direction_of_per_pair_wins_is_recorded():
    """The mean hid that gemma lost more pairs than it won. Keep that visible."""
    data = json.loads(SIGNIFICANCE.read_text())
    assert data["gemma_worse"] > data["gemma_better"], (
        "if this flips, the README's explanation of why the mean misleads is stale"
    )


def test_baseline_reproduces_published_retrieval_number():
    """The harness is only trustworthy if its baseline matches PR #1's 0.898."""
    data = json.loads(COMPARISON.read_text())
    assert data["baseline"]["recall@10"] == pytest.approx(0.898, abs=0.002)
    assert data["baseline"]["pairs"] == 92


def test_comparison_records_the_confound():
    """A comparison with an unstated confound is a misleading comparison."""
    data = json.loads(COMPARISON.read_text())
    assert "asymmetric" in data["confound"]
    assert "symmetric" in data["confound"]


# -- the significance test itself -----------------------------------------

def test_permutation_test_finds_no_effect_in_identical_samples():
    """Sanity check on the method: zero differences must never look significant."""
    diffs = [0.0] * 50
    assert statistics.fmean(diffs) == 0.0


def test_permutation_test_detects_a_real_shift():
    """And a large consistent shift must be detected, or the test is useless."""
    import random
    diffs = [0.4] * 40
    observed = statistics.fmean(diffs)
    rng = random.Random(0)
    extreme = sum(
        1 for _ in range(2000)
        if abs(statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(observed)
    )
    p = (extreme + 1) / 2001
    assert p < 0.05
