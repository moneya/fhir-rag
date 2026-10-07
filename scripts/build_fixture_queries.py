"""Freeze the fixture's query vectors so CI needs no embedding API.

The fixture questions are fixed, so their embeddings are constants. Storing them
is what makes the gate runnable on a fresh clone with no key. They are small
(one vector per query, not per chunk) so they stay as plain floats — no
quantization, no extra lossiness on the query side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/whitefang/lab/evalkit/src")

from evalkit.env import load_env

load_env()

from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.queries import build_queries

fixture = VectorIndex.load(ROOT / "tests" / "fixtures" / "index_fixture.json")
embedder = get_embedder("nvidia", cache=EmbeddingCache(fixture.model))

pid = fixture.chunks[0].patient_id
vectors = {}
for query in build_queries(pid):
    if not any(query.relevant(c) for c in fixture.chunks):
        continue
    vectors[query.id] = [round(x, 6) for x in embedder.embed_query(query.question)]

out = ROOT / "tests" / "fixtures" / "query_vectors.json"
out.write_text(json.dumps(vectors), encoding="utf-8")
print(f"{len(vectors)} query vectors -> {out} ({out.stat().st_size / 1e3:.0f} KB)")
print("queries:", sorted(vectors))
