"""Benchmark queries with ground truth derived from FHIR codes.

Every `relevant` predicate matches on a coded identifier, never on a substring
of the rendered text. That distinction is the whole point: matching text would
make the benchmark circular — the retriever is scored on embeddings of that same
text, so a keyword rule would reward lexical overlap rather than retrieval.

Codes were taken from the corpus itself (see `fhir-rag codes`), so every query
here has real ground truth in the Synthea sample set.
"""

from __future__ import annotations

from .evaluate import Query
from .ingest import Chunk

# --- code sets, verified present in the Synthea sample data ---------------

RXNORM_DIABETES = {
    "860975",   # 24HR metformin hydrochloride 500 MG extended release
    "860974", "861007", "861004",  # other metformin strengths
    "106892",   # insulin isophane human
    "311041",   # insulin glargine
    "310490",   # glipizide
}

LOINC_HBA1C = {"4548-4"}
LOINC_GLUCOSE_SERUM = {"2345-7"}
LOINC_BP_SYSTOLIC = {"8480-6"}
LOINC_BODY_WEIGHT = {"29463-7"}

SNOMED_HYPERTENSION = {"59621000"}
SNOMED_PREDIABETES = {"714628002"}
SNOMED_KIDNEY_DISORDER = {"127013003", "90781000119102"}

CVX_INFLUENZA = {"140", "150", "155", "158", "161"}


def _has_code(chunk: Chunk, codes: set[str]) -> bool:
    """True if the chunk's primary code is in `codes`.

    FHIR codes arrive as `system|code`; the system prefix varies by resource
    (`rxnorm`, `loinc.org`, `sct`, `cvx`) so only the code half is compared.
    """
    for entry in chunk.codes:
        code = entry.split("|")[-1]
        if code in codes:
            return True
    return False


def _of_type(chunk: Chunk, *types: str) -> bool:
    return chunk.resource_type in types


# --- the benchmark -------------------------------------------------------

def build_queries(patient_id: str | None = None) -> list[Query]:
    """Clinical questions a real user would ask of a patient record."""
    return [
        Query(
            id="diabetes-medication",
            shape="list",
            question="Which diabetes medication is this patient taking?",
            relevant=lambda c: _of_type(c, "MedicationRequest", "MedicationAdministration")
            and _has_code(c, RXNORM_DIABETES),
            patient_id=patient_id,
            note="RxNorm diabetes drugs. The phrasing shares no words with "
                 "'metformin hydrochloride', so lexical overlap cannot win.",
        ),
        Query(
            id="hba1c-results",
            shape="list",
            question="What were this patient's most recent HbA1c results?",
            relevant=lambda c: _of_type(c, "Observation") and _has_code(c, LOINC_HBA1C),
            patient_id=patient_id,
            note="LOINC 4548-4. Tests whether an acronym retrieves a lab "
                 "rendered as 'Hemoglobin A1c/Hemoglobin.total in Blood'.",
        ),
        Query(
            id="blood-sugar-labs",
            shape="list",
            question="Show blood sugar measurements from blood tests.",
            relevant=lambda c: _of_type(c, "Observation")
            and _has_code(c, LOINC_GLUCOSE_SERUM),
            patient_id=patient_id,
            note="Serum glucose only. Urine glucose (25428-4) is a different "
                 "code and is deliberately NOT relevant — a lexical matcher "
                 "cannot tell them apart, a good retriever should.",
        ),
        Query(
            id="hypertension-diagnosis",
            shape="single",
            question="Does this patient have high blood pressure?",
            relevant=lambda c: _of_type(c, "Condition") and _has_code(c, SNOMED_HYPERTENSION),
            patient_id=patient_id,
            note="SNOMED 59621000, rendered as 'Essential hypertension'. "
                 "Lay phrasing vs clinical term.",
        ),
        Query(
            id="prediabetes",
            shape="single",
            question="Is there any indication of impaired glucose regulation?",
            relevant=lambda c: _of_type(c, "Condition") and _has_code(c, SNOMED_PREDIABETES),
            patient_id=patient_id,
            note="SNOMED 714628002 (Prediabetes). No shared vocabulary at all.",
        ),
        Query(
            id="kidney-problems",
            shape="single",
            question="Any kidney-related conditions on record?",
            relevant=lambda c: _of_type(c, "Condition") and _has_code(c, SNOMED_KIDNEY_DISORDER),
            patient_id=patient_id,
        ),
        Query(
            id="systolic-bp",
            shape="list",
            question="What is the patient's systolic blood pressure reading?",
            relevant=lambda c: _of_type(c, "Observation") and _has_code(c, LOINC_BP_SYSTOLIC),
            patient_id=patient_id,
            note="LOINC 8480-6, which Synthea stores as a component inside a "
                 "blood-pressure panel rather than a standalone observation.",
        ),
        Query(
            id="flu-vaccination",
            shape="list",
            question="When was the patient last vaccinated against influenza?",
            relevant=lambda c: _of_type(c, "Immunization") and _has_code(c, CVX_INFLUENZA),
            patient_id=patient_id,
        ),
        Query(
            id="body-weight",
            shape="list",
            question="How much does the patient weigh?",
            relevant=lambda c: _of_type(c, "Observation") and _has_code(c, LOINC_BODY_WEIGHT),
            patient_id=patient_id,
        ),
    ]


__all__ = ["build_queries", "RXNORM_DIABETES", "LOINC_HBA1C", "SNOMED_HYPERTENSION"]
