"""Build the committable CI fixture index.

The real index is 806 MB of JSON floats, gitignored, and needs a live API call to
rebuild — which is why the CI gate currently skips. A skipped gate under a green
badge is worse than no gate, so this writes a small quantized index that ships in
the repo and lets the gate run on every push with no key and no download.

Patient 35f80d0e was chosen by measurement, not convenience: it is the smallest
patient that retains BOTH pathology queries the shape-aware policy exists to fix
(prediabetes and hypertension-diagnosis), with 154 Observations against 25
Conditions so "measurements drown diagnoses" is still reproducible in it.

The fixture deliberately does NOT reproduce the 30-patient headline numbers.
Fewer patients means fewer (patient, query) pairs and different averages; a gate
claiming otherwise would be lying. Its job is to catch a ranking-policy
regression anywhere, for free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhir_rag.index import VectorIndex

FIXTURE_PATIENT = "35f80d0e-eaad-ef68-b762-3a1c12a872c1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/index_30.json")
    parser.add_argument("--out", default="tests/fixtures/index_fixture.json")
    parser.add_argument("--patient", default=FIXTURE_PATIENT)
    args = parser.parse_args()

    full = VectorIndex.load(args.source)
    keep = [
        (c, v) for c, v in zip(full.chunks, full.vectors)
        if c.patient_id == args.patient
    ]
    if not keep:
        raise SystemExit(
            f"patient {args.patient} not found in {args.source} — "
            f"rebuild the source index or pick another patient"
        )

    fixture = VectorIndex(model=full.model, asymmetric=full.asymmetric)
    fixture.add([c for c, _ in keep], [list(v) for _, v in keep])
    fixture.save(args.out, quantize=True)

    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"{len(fixture)} chunks -> {args.out} ({size_mb:.2f} MB, int8-base64)")

    # Round-trip check: a fixture that does not survive save/load is worthless,
    # and the failure would surface as a confusing gate failure later.
    reloaded = VectorIndex.load(args.out)
    assert len(reloaded) == len(fixture), "chunk count changed on reload"
    assert reloaded.dims == fixture.dims, "dimensionality changed on reload"
    worst = max(
        abs(a - b)
        for va, vb in zip(fixture.vectors, reloaded.vectors)
        for a, b in zip(va, vb)
    )
    print(f"round-trip ok: {len(reloaded)} chunks, max per-dim drift {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
