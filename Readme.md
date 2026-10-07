# fhir-rag

Retrieval over FHIR patient records, with recall measured against ground truth
derived from clinical codes rather than hand-labelled opinion.

The hard part of RAG is not building it — it is knowing whether retrieval
actually worked. This project aims to answer that with a number you can
reproduce.

## Why FHIR, and why Synthea

Every clinical fact in a FHIR record carries a coded identity: RxNorm for drugs,
LOINC for labs, SNOMED for conditions, CVX for vaccines. So for a question like
*"which diabetes medication is this patient taking?"*, the set of correct answers
is **derivable** — it is exactly the `MedicationRequest` resources whose RxNorm
code appears in the diabetes drug set.

That turns `recall@k` into a computed metric instead of a claim.
[Synthea](https://synthetichealth.github.io/synthea/) provides 111 synthetic
patient bundles with the same structure as real records and none of the privacy
problems.

Ground-truth rules will match on **codes, never on text**. Matching substrings
would be circular: the retriever is scored on embeddings of that same text, so a
keyword rule would reward lexical overlap rather than retrieval quality.

## Planned scope

- Bundle ingestion that renders coded resources into retrievable sentences
- Config-selectable embedding backends (NVIDIA NIM, Ollama, OpenAI)
- A cosine index with per-patient scoping
- A retrieval benchmark whose ground truth comes from codes, never from text
- Cost and latency gating via [evalkit](https://github.com/moneya/evalkit)

## Corpus

Not vendored — 28 MB, and reproducible:

```bash
mkdir -p data/raw && cd data/raw
curl -sLO https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip
unzip -q synthea_sample_data_fhir_latest.zip
```

## Credentials

```bash
cp .env.example .env
```

`.env` is gitignored. No key is required for the offline Ollama path.

## Licence

MIT — see [LICENSE](LICENSE).
