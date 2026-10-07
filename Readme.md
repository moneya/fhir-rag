# fhir-rag

Retrieval over FHIR patient records, with recall measured against ground truth
derived from clinical codes rather than hand-labelled opinion.

The hard part of RAG is not building it — it is knowing whether retrieval
actually worked. This project answers that with a number you can reproduce.

## Why FHIR, and why Synthea

Every clinical fact in a FHIR record carries a coded identity: RxNorm for drugs,
LOINC for labs, SNOMED for conditions, CVX for vaccines. So for a question like
*"which diabetes medication is this patient taking?"*, the set of correct answers
is **derivable** — it is exactly the `MedicationRequest` resources whose RxNorm
code appears in the diabetes drug set.

That makes `recall@k` a computed metric instead of a claim. [Synthea](https://synthetichealth.github.io/synthea/)
provides 111 synthetic patient bundles with the same structure as real records
and none of the privacy problems.

Ground-truth rules match on **codes, never on text**. Matching substrings would
be circular: the retriever is scored on embeddings of that same text, so a
keyword rule would reward lexical overlap rather than retrieval quality.

## Ingestion

A Synthea bundle is ~466 resources with no narrative. Two decisions shape what
gets indexed:

**Billing resources are dropped.** `Claim` and `ExplanationOfBenefit` are 88 of
466 entries in a typical bundle — insurance boilerplate containing no clinical
fact that isn't already in the resource it points at. Embedding them buries real
findings.

**One chunk per clinical event, not per fixed token window.** A FHIR resource is
already the natural unit. Splitting a lab result in half yields two useless
fragments; merging twenty yields a chunk that matches everything.

Resources render as sentences phrased the way a clinician would state them,
because the question will be phrased that way too:

```
Alexandra Mosciski was prescribed 24 HR Metformin hydrochloride 500 MG on 2016-03-06.
Alexandra Mosciski had Hemoglobin A1c/Hemoglobin.total in Blood of 5.85 % measured 2016-12-05.
Alexandra Mosciski has a diagnosis of Prediabetes (finding), recorded 2015-08-17.
```

## Embedding backends

Three providers behind one interface, selected by config:

| Backend | Model | Dims | Asymmetric | Key needed |
|---|---|---|---|---|
| `nvidia` | `nemotron-3-embed-1b` | 2048 | yes | `NVIDIA_API_KEY` |
| `ollama` | `nomic-embed-text` | 768 | no | none (local) |
| `openai` | `text-embedding-3-small` | 1536 | no | `OPENAI_API_KEY` |

The abstraction exists for a concrete reason: **`input_type` is not part of the
OpenAI embeddings spec.** "OpenAI-compatible" is only half-true for embeddings,
so a single hardcoded client would either drop the parameter — losing asymmetric
retrieval — or send it to providers that reject it.

Asymmetric matters because retrieval is a relevance problem, not a similarity
one. *"Which diabetes medication is the patient taking?"* should match *"was
prescribed metformin"* — two sentences that are not similar as text. Measured on
the NVIDIA model, encoding the same string as a query versus a passage gives
cosine **0.777**, so the distinction is real and not a no-op flag.

## Caching and concurrency

An embedding is a pure function of (model, role, text), so it is computed once
and cached to JSONL on disk.

Measured on 326 chunks against the free NVIDIA tier:

| | Wall clock | Tokens billed |
|---|---|---|
| Sequential, no cache | 118.3 s | 13,860 |
| 8 concurrent batches | **7.3 s** (16.2x) | 13,860 |
| Warm cache | **0.125 s** | 0 |

Almost all of the original time was round-trip latency, not compute. The cache
is what makes iterating on retrieval practical — changing the query set no longer
costs an API run.

Two correctness details that are easy to get wrong and silent when wrong:

- **Query and passage roles are separate cache keys.** Asymmetric models return
  different vectors for the same text; sharing one entry would quietly corrupt
  retrieval while still returning plausible results.
- **Concurrent batches are reassembled by input position, not completion order.**
  A test forces the first batch to finish last and asserts the vectors stay
  aligned. Misalignment doesn't crash — it just makes every number meaningless.

The model itself is not bit-deterministic: embedding identical text twice gives a
max per-dimension delta of 6e-08 (cosine 1.0000000), i.e. GPU floating-point
ordering. Cache hits are byte-identical, which is why cached and live runs are
compared with cosine rather than equality.

## Results

30 patients, 17,472 chunks, 92 (patient, query) pairs with code-derived ground
truth, NVIDIA `nemotron-3-embed-1b`, k=10:

| Ranking policy | recall@10 | hit@10 | MRR |
|---|---|---|---|
| flat top-k | 0.767 | 0.826 | 0.779 |
| cap 2 per resource type | 0.523 | 0.957 | 0.811 |
| MMR (λ=0.7) | 0.618 | 0.891 | 0.784 |
| interleave by type | 0.497 | 0.957 | 0.823 |
| **shape-aware** | **0.898** | **0.957** | 0.813 |

### Why a global diversification strategy cannot win

The first flat-ranking run had two queries at **0.000 recall**: hypertension and
prediabetes. Diagnosing it showed the retriever was not wrong — the ground truth
ranked 28th of 2247 — but a patient carries ~100 near-identical "Blood pressure
panel" observations scoring 0.426–0.436, and the `Essential hypertension`
diagnosis they relate to scores 0.295. Restricted to `Condition` resources it is
rank 1 by a wide margin. Nothing is mis-scored; the signal is buried under
repetition.

Capping each resource type to 2 fixes exactly that:

| Query | avg relevant chunks | flat | cap 2 |
|---|---|---|---|
| prediabetes | 1.0 | 0.000 | **1.000** |
| hypertension-diagnosis | 1.0 | 0.125 | **0.875** |
| flu-vaccination | 7.4 | **1.000** | 0.332 |
| body-weight | 10.0 | **0.885** | 0.433 |
| diabetes-medication | 19.0 | **0.526** | 0.211 |

The split is perfectly clean: **capping wins when the answer is one chunk and
loses when the answer is legitimately a list.** Truncating "every influenza
vaccination" at two rows is not diversification, it is data loss. Interleaving by
type was tried on the theory that it would avoid the guess — it scored 0.497,
worse than flat, so the hypothesis was wrong.

So the policy is declared per query rather than chosen globally:

```python
Query(
    id="hypertension-diagnosis",
    question="Does this patient have high blood pressure?",
    shape="single",   # over-fetch 6x, then cap 2 per resource type
)
Query(
    id="flu-vaccination",
    question="When was the patient last vaccinated against influenza?",
    shape="list",     # flat top-k, the answer really is several rows
)
```

That reaches 0.898 recall@10 — above every single strategy, because each query
gets the better of the two. The trade-off is explicit in the data model instead
of hidden inside an average.

## Does the hosted embedding model earn its dependency?

Every number above came from `nvidia/nemotron-3-embed-1b` — 2048 dims, hosted,
requires a key, and the reason the full CI gate cannot run without one. So the
dependency was tested rather than assumed, against `embeddinggemma:300m` running
locally through Ollama: 621 MB, 768 dims, no key, no network egress.

Same 30 patients, same 17,472 chunks, same code-derived ground truth, same
`retrieve()` path, same k. Only the vectors differ.

| metric | nemotron (2048d, hosted) | embeddinggemma (768d, local) |
|---|---|---|
| recall@10 | 0.8978 | 0.8986 |
| hit@10 | 0.9565 | 1.0000 |
| MRR | 0.8127 | 0.8642 |
| precision@1 | 0.7500 | 0.7717 |
| throughput | 61 chunks/s (API, 8-way) | 97 chunks/s (local) |

The local model appears to win every metric. **It does not.** A paired
permutation test over the 92 (patient, query) pairs gives **p = 0.971**:

```
mean difference (gemma - nemotron): +0.0007
pairs where they differ at all:     22 of 92
  gemma better: 5
  gemma worse:  17
```

Gemma is worse on more than three times as many pairs as it is better on; the mean
tilts positive only because its few wins are larger. **The defensible claim is
parity on recall, not a win.**

Two further cautions kept in rather than dropped:

* `kidney-problems` shows a terrifying −0.500 per-query delta. It has **one**
  patient with ground truth. A single query cannot be read as a trend, so it is
  reported and excluded, not averaged in silently. Removing both thin cells
  (≤3 patients) leaves the direction unchanged: 0.9008 vs 0.9077.
* nemotron is **asymmetric** (separate query/passage encoding), EmbeddingGemma is
  **symmetric**. That asymmetry was the original reason for choosing nemotron, so
  this is not a single-variable comparison.

The useful conclusion is not "model X beats model Y". It is that on this task, a
768-dim model that runs on a laptop is **statistically indistinguishable** from a
2048-dim hosted one — so the key, the egress and the CI dependency are buying
convenience, not quality.

## Storage: cut bits before dimensions

The committed fixture is int8. The obvious next step for shrinking it is fewer
dimensions, and that instinct is wrong. Measured on the same vectors and ground
truth, at a **matched 256 bytes/vector** budget:

| encoding | dims | bytes/vector | recall@10 | MRR |
|---|---|---|---|---|
| int8, full dims | 2048 | 2048 | 1.000 | 0.700 |
| **1-bit, full dims** | 2048 | **256** | **1.000** | 0.695 |
| float32, truncated | 64 | 256 | **0.440** | 0.465 |
| int8, half dims | 1024 | 1024 | 1.000 | 0.700 |

Dropping to **one bit per dimension costs essentially nothing** (recall identical,
MRR −0.005) at 1/8 the bytes. Spending the same budget on fewer dimensions
**halves recall**. This reproduces the direction Qdrant reports for TurboQuant,
with a wider margin on this corpus.

Caveat: nemotron is not documented as Matryoshka-trained, so naive truncation is
probably unfairly weak for it. EmbeddingGemma is Matryoshka-trained and would be
the fairer test of the truncation leg.

## Reranking

No reranker model is callable on this NVIDIA account — every `nv-rerankqa` id
returns 404 — so reranking uses a cheap instruct model over the candidates the
embedding search already found. Groq serves `qwen/qwen3.8-27b` in ~0.5s for as
few as 2 output tokens, fast enough to sit in the retrieval path.

**The prompt shape was chosen by measurement.** Asking for *"only the passage
numbers that answer the question"* collapsed a seven-vaccination answer to one
line — 1/7 recall. Asking the model to **rank all candidates and return the top
k** kept 7/7 and ordered them correctly. The instruction is therefore always
"rank", never "filter": how many rows an answer needs is the retriever's job via
`shape`, not the reranker's.

### What it actually buys

10 patients, shape-aware retrieval with and without reranking, same index and
same cached query embeddings:

| Metric | shape-aware | + rerank | delta |
|---|---|---|---|
| recall@10 | 0.908 | 0.909 | +0.001 |
| hit@10 | 0.941 | 0.971 | +0.030 |
| MRR | 0.775 | 0.848 | +0.073 |
| **precision@1** | **0.706** | **0.824** | **+0.118** |
| precision@3 | 0.725 | 0.745 | +0.020 |

Recall barely moves, and that is the honest headline: shape-aware retrieval
already pulls the right chunks into the top ten, so there is little left to
recover. What reranking fixes is the **order within those ten** — precision@1
rises 17% relative, which is what a user or a generation step actually consumes.
Nobody reads to rank 10.

Latency: mean 0.81s, p50 0.61s, max 2.50s over 34 calls. Replies are cached, so
re-running the benchmark is free.

### Two rules the reranker will not bend

**It never invents or drops candidates.** The output is a permutation of the
input. Out-of-range ids, duplicates and hallucinated numbers are discarded, and
any candidate the model omitted is appended in its original embedding order. A
reranker that silently loses documents is worse than no reranker.

**It fails closed to embedding order.** A timeout, a refusal, a prose answer or
an unparseable reply returns the input ranking unchanged — never an empty or
partial list. Verified against the live API: an unreachable host and a real HTTP
4xx both fall back with all candidates intact. Retrieval degrading to "merely
unreranked" is acceptable; retrieval returning nothing because an LLM had an off
day is not.

## Generation

Retrieval without generation is a search box; generation without grounding is a
liability. Answers are produced from retrieved records only, via Groq
`qwen/qwen3.8-27b` (~0.6s, ~36 output tokens).

5 patients, k=5, reranked retrieval, 17 answers, 0 transport errors:

| Measure | Value |
|---|---|
| citation precision | 0.846 |
| citation recall | 0.523 |
| grounded | 0.941 |
| hallucinated record ids | **0** |
| mean latency | 0.58 s |
| mean output tokens | 36 |

**Control arm — abstained 5/5.** Each question was also asked against another
patient's unrelated immunisation records, where refusing is the only correct
answer. It refused every time. Without that arm, a model that always answers
would score well on everything above.

Abstention also survives near-miss distractors: asked for a *blood* glucose value
when only a *urine* glucose record was present, it returned `INSUFFICIENT
EVIDENCE` rather than conflating the two. Asked whether a patient had severe
depression, it read a PHQ-9 score of 2 and said no.

### Citations are verified, never trusted

The model cites records as `[n]`; those ids are then checked in code against the
context that was actually sent. Out-of-range ids are stripped into
`invalid_citations` rather than silently accepted. A plausible answer citing
record [7] when six were supplied is the most dangerous output a clinical RAG
system can produce, because it looks sourced. Zero occurred in this run, but the
check is what makes that claim meaningful.

An empty context never reaches the model at all — a model asked a clinical
question with no records answers from training data, which is precisely what
grounding exists to prevent.

### The most interesting failure

One answer was flagged as **unsourced**: retrieval missed the patient's
`Prediabetes` diagnosis, and the model answered anyway. Inspecting it (see
`scripts/repro_false_answer.py`) showed this:

> Yes, there is an indication of impaired glucose regulation. The patient's
> Hemoglobin A1c levels were 5.82% [1] and 6.03% [2], both of which fall within
> the range typically associated with prediabetes.

The patient **does** have a Prediabetes diagnosis, and 6.03% **is** the
prediabetic range (5.7–6.4%). The model reached a clinically correct conclusion
from valid evidence the ground-truth rule did not designate.

So the metric was wrong, not the model — and the label was changed from "false
answer" to "unsourced", counted for review rather than scored as an error.
`grounded` certifies that cited records exist and were supplied; it does **not**
certify that the conclusion is correct. Conflating provenance with correctness is
how RAG systems get oversold.

## CI gate

Retrieval quality is gated the same way a unit test is, using
[evalkit](https://github.com/moneya/evalkit):

```bash
python scripts/emit_metrics.py --patients 30 --out data/metrics.json
evalkit run evals/retrieval_gate.yaml
```

Measurement and policy are deliberately separate. `emit_metrics.py` knows how to
compute recall and nothing about what is acceptable; the thresholds live in
`evals/retrieval_gate.yaml`, so moving a bar is a config change visible in a PR
rather than an edit buried in Python.

**It runs with no API key and no cost** — 92 cache hits, 0 misses. One gate case
asserts `cache.misses <= 0`, because a CI job silently missing the cache is
making paid API calls on every push, which is a cost regression.

Thresholds sit below the measured values rather than at them. A gate pinned to
the exact current number fails on noise and gets disabled within a week, which is
worse than no gate:

| Metric | Measured | Gate |
|---|---|---|
| recall@10 | 0.898 | ≥ 0.85 |
| hit@10 | 0.957 | ≥ 0.92 |
| MRR | 0.813 | ≥ 0.75 |
| precision@1 | 0.750 | ≥ 0.70 |

The two most valuable cases guard against opposite regressions, because a single
macro number hides both:

- `sparse-answer-queries` — prediabetes and hypertension, which were 0.000 and
  0.125 before shape-aware ranking. A global diversification policy would take
  them back to zero while macro recall still looked respectable.
- `dense-answer-queries` — flu-vaccination and body-weight, which a global
  resource-type cap would drop from 1.000 to 0.332.

### Two gates, one of which always runs

The 30-patient gate needs an 806 MB index and a live embedding key, so on a fresh
clone it skips — and a skipped gate under a green badge is worse than no gate. So
there is a second, committed gate:

```bash
python scripts/emit_fixture_metrics.py --out data/fixture_metrics.json
evalkit run evals/fixture_gate.yaml
```

The index (1.0 MB) and query vectors (106 KB) ship in `tests/fixtures/`, so this
runs on every push with **no API key, no cache and no download**. Vectors are
stored int8-quantized as base64 — 160x smaller than JSON floats, and the encoding
is recorded in the payload so a gate always knows it is reading quantized data.

The fixture patient was chosen by measurement, not convenience: it is the
smallest of 30 that retains **both** pathology queries, with 154 Observations
against 25 Conditions so "measurements drown diagnoses" is still reproducible.

It deliberately does **not** reproduce the 30-patient headline numbers — one
patient gives 5 (patient, query) pairs instead of 92, so the averages differ.
Claiming otherwise would be dishonest. Its job is narrower: catch a ranking-policy
regression, which it does because the fixture separates the candidate policies by
a wide margin:

| Policy | recall@10 on fixture |
|---|---|
| shape-aware | **1.000** |
| flat top-k | 0.800 — prediabetes collapses to 0.000 |
| global cap 2 | 0.760 — flu-vaccination and body-weight fall to 0.400 |

Note that MRR is 0.700 and precision@1 is 0.600 even here: all the right chunks
are retrieved, but a measurement still out-ranks a diagnosis at position 1. The
gate is pinned at those measured values rather than rounded up, because that is
the documented model limitation, not a bug to hide.

### Verified to fail, not just to pass

A gate that only ever passes is decorative. The fixture gate was tested by
breaking the **real code path** — injecting the pre-shape-aware policy into
`retrieve.py` — not by editing a metrics file:

```
FAIL  sparse-answer-queries
      json_path: per_query.prediabetes=0, want >= 1
FAIL  macro-retrieval
      json_path: retrieval.recall@10=0.8, want >= 1
FAIL  ranking-quality
      json_path: retrieval.mrr=0.6333, want >= 0.7

cases 2/5 passed   exit code 1
```

The 30-patient gate was checked the same way, with a regression injected into the
metrics file:

```
FAIL  recall-at-10
      json_path: retrieval.recall@10=0.61, want >= 0.85
FAIL  sparse-answer-queries
      json_path: per_query.prediabetes=0, want >= 0.9

cases 5/7 passed
exit code 1
```

## Install

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
cp .env.example .env     # add NVIDIA_API_KEY, or run offline with ollama
```

Fetch the corpus (28 MB, 111 Synthea patients — gitignored, not vendored):

```bash
mkdir -p data/raw && cd data/raw
curl -sLO https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip
unzip -q synthea_sample_data_fhir_latest.zip
```

## Usage

Ingestion needs no network or credentials:

```python
from fhir_rag.ingest import ingest_directory

chunks = ingest_directory("data/raw", limit=30)
print(len(chunks), "chunks")          # 17472
print(chunks[0].text)
```

Index and search:

```python
from fhir_rag.cache import EmbeddingCache
from fhir_rag.embeddings import get_embedder
from fhir_rag.index import VectorIndex
from fhir_rag.retrieve import retrieve

embedder = get_embedder(
    "nvidia",                                        # or "ollama" offline
    cache=EmbeddingCache("nvidia/nemotron-3-embed-1b"),
    concurrency=8,
)
vectors = embedder.embed_passages([c.text for c in chunks])

index = VectorIndex(model=embedder.model, asymmetric=embedder.asymmetric)
index.add(chunks, vectors.vectors)

hits = retrieve(
    index,
    embedder.embed_query("Does this patient have high blood pressure?"),
    k=10,
    patient_id=chunks[0].patient_id,
    shape="single",        # one diagnosis, so diversify; "list" for many rows
)
for hit in hits:
    print(f"{hit.rank}. {hit.score:.3f} {hit.chunk.text}")
```

Reproduce the benchmark tables:

```bash
python scripts/bench_diversify.py          # the five ranking policies
python scripts/diagnose_failures.py        # why a query scored 0.000
python scripts/bench_rerank.py 10          # rerank vs not: recall, hit, MRR
python scripts/bench_rerank_precision.py 10  # rerank vs not: precision@1/3/5
python scripts/bench_generate.py 5           # citations, abstention, control arm
python scripts/repro_false_answer.py         # the one unsourced answer, in full
```

Both reuse the on-disk embedding cache, so re-running costs nothing.

## Tests

```bash
pytest -q
```

Nothing in the test suite requires an API key: a fake backend encodes each
input's identity into its vector, so ordering bugs are detectable without
network access.

## Licence

MIT — see [LICENSE](LICENSE).
