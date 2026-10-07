"""Vector index.

Deliberately numpy-free: a cosine search over a few thousand chunks is
microseconds in pure Python, and the point of this project is retrieval quality,
not ANN engineering. Swapping in pgvector or FAISS later is a change behind
`search()`, not a redesign.

Vectors are L2-normalised once at insert, so cosine similarity is a dot product
and ranking never depends on document length.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ingest import Chunk


def normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class VectorIndex:
    """In-memory cosine index over rendered FHIR chunks."""

    def __init__(self, *, model: str = "", asymmetric: bool = False) -> None:
        self.model = model
        self.asymmetric = asymmetric
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dims(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunks but {len(vectors)} vectors — refusing to "
                f"attach embeddings to the wrong text"
            )
        for chunk, vector in zip(chunks, vectors):
            self.chunks.append(chunk)
            self.vectors.append(normalise(vector))

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 5,
        patient_id: str | None = None,
        resource_types: Iterable[str] | None = None,
    ) -> list[Hit]:
        """Top-k by cosine similarity, optionally scoped to one patient.

        Scoping matters clinically: "which medication is the patient on" must
        never return another patient's prescription, and a filter is a guarantee
        where a high similarity score is only a hope.
        """
        if not self.vectors:
            return []
        query = normalise(query_vector)
        if len(query) != self.dims:
            raise ValueError(
                f"query has {len(query)} dims but index has {self.dims} — "
                f"index was built with a different embedding model"
            )
        allowed = frozenset(resource_types) if resource_types else None

        scored: list[tuple[float, int]] = []
        for i, vector in enumerate(self.vectors):
            chunk = self.chunks[i]
            if patient_id and chunk.patient_id != patient_id:
                continue
            if allowed and chunk.resource_type not in allowed:
                continue
            scored.append((dot(query, vector), i))

        scored.sort(key=lambda pair: (-pair[0], self.chunks[pair[1]].chunk_id))
        return [
            Hit(chunk=self.chunks[i], score=score, rank=rank)
            for rank, (score, i) in enumerate(scored[:k], start=1)
        ]

    def save(self, path: Path | str, *, quantize: bool = False) -> None:
        """Write the index to disk.

        `quantize=True` stores int8-scaled dimensions as base64 instead of JSON
        floats, which is ~160x smaller and makes a committable CI fixture
        possible. It is lossy — measured at 0.033 recall on one query of five —
        so it is opt-in and recorded in the payload, never the default. A gate
        reading a quantized fixture must know it is reading one.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "asymmetric": self.asymmetric,
            "dims": self.dims,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        if quantize:
            payload["encoding"] = "int8-base64"
            payload["vectors"] = [
                base64.b64encode(
                    bytes(
                        (max(-127, min(127, round(x * 127.0))) + 256) % 256
                        for x in vector
                    )
                ).decode("ascii")
                for vector in self.vectors
            ]
        else:
            payload["vectors"] = self.vectors
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: Path | str) -> VectorIndex:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        index = cls(model=payload.get("model", ""), asymmetric=payload.get("asymmetric", False))
        index.chunks = [Chunk(**_chunk_fields(c)) for c in payload["chunks"]]
        if payload.get("encoding") == "int8-base64":
            index.vectors = [
                normalise([
                    (byte - 256 if byte > 127 else byte) / 127.0
                    for byte in base64.b64decode(blob)
                ])
                for blob in payload["vectors"]
            ]
        else:
            index.vectors = payload["vectors"]
        return index


def _chunk_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": raw["patient_id"],
        "patient_name": raw["patient_name"],
        "resource_type": raw["resource_type"],
        "resource_id": raw["resource_id"],
        "text": raw["text"],
        "date": raw.get("date"),
        "codes": raw.get("codes") or [],
    }


__all__ = ["Hit", "VectorIndex", "dot", "normalise"]
