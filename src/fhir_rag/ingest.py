"""FHIR bundle ingestion.

A Synthea bundle is a flat list of 400+ resources with no narrative, so the work
is turning codes and references into sentences a retriever can match a question
against.

Two decisions drive everything here:

**Billing resources are excluded.** Claim and ExplanationOfBenefit are ~20% of a
typical bundle and contain no clinical fact that isn't already in the resource
they point at. Embedding them buries real findings under insurance boilerplate.

**One chunk per clinical event, not per fixed token window.** A FHIR resource is
already the natural unit: splitting a lab result in half produces two useless
fragments, and merging twenty produces a chunk that matches everything. Chunks
carry their source resource type and id so a retrieved answer can be traced back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

# Resources that describe money rather than medicine.
BILLING_TYPES = frozenset({
    "Claim",
    "ExplanationOfBenefit",
    "Coverage",
    "PaymentNotice",
    "Invoice",
})

# Resources with no retrievable clinical content in Synthea output.
STRUCTURAL_TYPES = frozenset({
    "Provenance",
    "SupplyDelivery",
    "DocumentReference",  # base64 blobs of the same text we already render
})

# Clinical resources we render, in rough order of retrieval usefulness.
CLINICAL_TYPES = (
    "Patient",
    "Condition",
    "MedicationRequest",
    "MedicationAdministration",
    "AllergyIntolerance",
    "Observation",
    "Procedure",
    "Immunization",
    "DiagnosticReport",
    "Encounter",
    "CarePlan",
    "Device",
)


@dataclass
class Chunk:
    """One retrievable clinical statement."""

    patient_id: str
    patient_name: str
    resource_type: str
    resource_id: str
    text: str
    date: str | None = None
    codes: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.resource_type}/{self.resource_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "text": self.text,
            "date": self.date,
            "codes": self.codes,
        }


def _coding_text(node: dict[str, Any] | None) -> str:
    """Human label for a CodeableConcept, preferring text over a raw code."""
    if not node:
        return ""
    if node.get("text"):
        return str(node["text"])
    for coding in node.get("coding", []) or []:
        if coding.get("display"):
            return str(coding["display"])
    for coding in node.get("coding", []) or []:
        if coding.get("code"):
            return str(coding["code"])
    return ""


def _codes(node: dict[str, Any] | None) -> list[str]:
    """`system|code` pairs, kept for exact-match evaluation of retrieval."""
    if not node:
        return []
    out = []
    for coding in node.get("coding", []) or []:
        code = coding.get("code")
        if code:
            system = (coding.get("system") or "").rsplit("/", 1)[-1]
            out.append(f"{system}|{code}" if system else str(code))
    return out


def _when(resource: dict[str, Any]) -> str | None:
    for key in (
        "onsetDateTime", "recordedDate", "authoredOn", "effectiveDateTime",
        "performedDateTime", "occurrenceDateTime", "issued", "date",
    ):
        value = resource.get(key)
        if isinstance(value, str) and value:
            return value[:10]
    for key in ("performedPeriod", "effectivePeriod", "period"):
        period = resource.get(key)
        if isinstance(period, dict) and period.get("start"):
            return str(period["start"])[:10]
    return None


def _quantity(node: dict[str, Any]) -> str:
    value = node.get("value")
    if value is None:
        return ""
    unit = node.get("unit") or node.get("code") or ""
    if isinstance(value, float):
        value = round(value, 2)
    return f"{value} {unit}".strip()


def _observation_value(resource: dict[str, Any]) -> str:
    if "valueQuantity" in resource:
        return _quantity(resource["valueQuantity"])
    if "valueCodeableConcept" in resource:
        return _coding_text(resource["valueCodeableConcept"])
    for key in ("valueString", "valueBoolean", "valueInteger"):
        if key in resource:
            return str(resource[key])
    if "component" in resource:
        parts = []
        for component in resource["component"]:
            label = _coding_text(component.get("code"))
            if "valueQuantity" in component:
                parts.append(f"{label} {_quantity(component['valueQuantity'])}".strip())
            elif "valueCodeableConcept" in component:
                parts.append(f"{label} {_coding_text(component['valueCodeableConcept'])}".strip())
        return "; ".join(p for p in parts if p)
    return ""


def patient_name(patient: dict[str, Any]) -> str:
    """Readable name. Synthea suffixes digits to every name part."""
    import re

    names = patient.get("name") or []
    if not names:
        return "Unknown"
    entry = next((n for n in names if n.get("use") == "official"), names[0])
    given = " ".join(entry.get("given") or [])
    family = entry.get("family") or ""
    full = f"{given} {family}".strip()
    return re.sub(r"\d+", "", full).strip() or "Unknown"


def _age(patient: dict[str, Any]) -> str:
    birth = patient.get("birthDate")
    if not birth:
        return ""
    try:
        born = date.fromisoformat(birth)
    except ValueError:
        return ""
    end = patient.get("deceasedDateTime")
    ref = date.fromisoformat(end[:10]) if isinstance(end, str) and end else date.today()
    years = ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))
    return f"{years}-year-old"


def render(resource: dict[str, Any], name: str) -> str:
    """One sentence describing a clinical resource.

    Phrased the way a clinician would state the fact, because the question will
    be phrased that way too: "patient is taking X", not "MedicationRequest: X".
    """
    kind = resource.get("resourceType")

    if kind == "Patient":
        bits = [b for b in (_age(resource), resource.get("gender")) if b]
        line = f"{name} is a {' '.join(bits)} patient." if bits else f"{name} is a patient."
        address = (resource.get("address") or [{}])[0]
        city, state = address.get("city"), address.get("state")
        if city and state:
            line += f" Lives in {city}, {state}."
        if resource.get("deceasedDateTime"):
            line += f" Deceased {str(resource['deceasedDateTime'])[:10]}."
        return line

    if kind == "Condition":
        label = _coding_text(resource.get("code"))
        status = (resource.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code", "")
        when = _when(resource)
        line = f"{name} has a diagnosis of {label}"
        if status and status != "active":
            line = f"{name} had a {status} diagnosis of {label}"
        return line + (f", recorded {when}." if when else ".")

    if kind in ("MedicationRequest", "MedicationAdministration"):
        label = _coding_text(resource.get("medicationCodeableConcept")) or "a medication"
        when = _when(resource)
        verb = "was prescribed" if kind == "MedicationRequest" else "was administered"
        line = f"{name} {verb} {label}"
        return line + (f" on {when}." if when else ".")

    if kind == "AllergyIntolerance":
        return f"{name} has a recorded allergy to {_coding_text(resource.get('code'))}."

    if kind == "Observation":
        label = _coding_text(resource.get("code"))
        value = _observation_value(resource)
        when = _when(resource)
        line = f"{name} had {label}" + (f" of {value}" if value else "")
        return line + (f" measured {when}." if when else ".")

    if kind == "Procedure":
        label = _coding_text(resource.get("code"))
        when = _when(resource)
        return f"{name} underwent {label}" + (f" on {when}." if when else ".")

    if kind == "Immunization":
        label = _coding_text(resource.get("vaccineCode"))
        when = _when(resource)
        return f"{name} received a {label} immunization" + (f" on {when}." if when else ".")

    if kind == "DiagnosticReport":
        label = _coding_text(resource.get("code"))
        when = _when(resource)
        return f"{name} had a diagnostic report: {label}" + (f" on {when}." if when else ".")

    if kind == "Encounter":
        label = _coding_text((resource.get("type") or [{}])[0])
        when = _when(resource)
        reason = _coding_text((resource.get("reasonCode") or [{}])[0])
        line = f"{name} had an encounter: {label or 'visit'}"
        if reason:
            line += f" for {reason}"
        return line + (f" on {when}." if when else ".")

    if kind == "CarePlan":
        label = _coding_text((resource.get("category") or [{}])[0])
        return f"{name} is on a care plan: {label}."

    if kind == "Device":
        return f"{name} has a device: {_coding_text(resource.get('type'))}."

    label = _coding_text(resource.get("code"))
    return f"{name}: {kind}" + (f" {label}" if label else "") + "."


def _primary_code_node(resource: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("code", "medicationCodeableConcept", "vaccineCode", "type"):
        node = resource.get(key)
        if isinstance(node, dict):
            return node
        if isinstance(node, list) and node and isinstance(node[0], dict):
            return node[0]
    return None


def chunks_from_bundle(bundle: dict[str, Any], *, include: Iterable[str] | None = None) -> list[Chunk]:
    """Render one bundle into retrievable chunks."""
    entries = [e.get("resource", {}) for e in bundle.get("entry", []) or []]
    patient = next((r for r in entries if r.get("resourceType") == "Patient"), None)
    if patient is None:
        return []

    pid = patient.get("id", "unknown")
    name = patient_name(patient)
    allowed = frozenset(include) if include else frozenset(CLINICAL_TYPES)

    out: list[Chunk] = []
    for resource in entries:
        kind = resource.get("resourceType")
        if not kind or kind in BILLING_TYPES or kind in STRUCTURAL_TYPES:
            continue
        if kind not in allowed:
            continue
        text = render(resource, name).strip()
        # A bare "Name: Type." carries no retrievable fact.
        if not text or text.endswith(f"{kind}."):
            continue
        out.append(Chunk(
            patient_id=pid,
            patient_name=name,
            resource_type=kind,
            resource_id=resource.get("id", ""),
            text=text,
            date=_when(resource),
            codes=_codes(_primary_code_node(resource)),
        ))
    return out


def load_bundle(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_bundles(directory: Path | str) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(Path(directory).glob("*.json")):
        try:
            yield path, load_bundle(path)
        except (json.JSONDecodeError, OSError):
            continue


def ingest_directory(
    directory: Path | str,
    *,
    limit: int | None = None,
    include: Iterable[str] | None = None,
) -> list[Chunk]:
    out: list[Chunk] = []
    for i, (_, bundle) in enumerate(iter_bundles(directory)):
        if limit is not None and i >= limit:
            break
        out.extend(chunks_from_bundle(bundle, include=include))
    return out


__all__ = [
    "BILLING_TYPES",
    "CLINICAL_TYPES",
    "STRUCTURAL_TYPES",
    "Chunk",
    "chunks_from_bundle",
    "ingest_directory",
    "iter_bundles",
    "load_bundle",
    "patient_name",
    "render",
]
