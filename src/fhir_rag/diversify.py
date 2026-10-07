"""Result diversification.

A flat top-k over a clinical record is dominated by whichever resource type is
most *numerous*, not most relevant. A single patient carries ~100 near-identical
"Blood pressure panel" observations scoring 0.426-0.436; the Essential
hypertension diagnosis they relate to scores 0.295 and ranks 28th. Nothing is
mis-scored — the signal is simply buried under repetition.

Two complementary strategies, both standard in production retrieval:

`per_type_cap`  at most N chunks from any one resource type in the top k. Cheap,
                deterministic, and enough to surface a diagnosis alongside the
                measurements that triggered the question.

`mmr`           Maximal Marginal Relevance: iteratively pick the candidate that
                maximises relevance minus similarity to what is already chosen.
                Handles near-duplicate *text* even within one resource type.

Both operate on an over-fetched candidate list, so recall can only improve: the
flat ranking is still available by passing no strategy.
"""

from __future__ import annotations

from collections import Counter

from .index import Hit, dot


def cap_per_resource_type(hits: list[Hit], *, k: int, cap: int = 2) -> list[Hit]:
    """Keep ranking order but allow at most `cap` hits per resource type.

    Chosen over MMR as the default because it is deterministic, needs no extra
    vector maths, and targets the actual failure: repetition *across* resource
    types, where one type has two orders of magnitude more rows than another.
    """
    if cap <= 0:
        raise ValueError("cap must be >= 1")

    kept: list[Hit] = []
    seen: Counter[str] = Counter()
    overflow: list[Hit] = []

    for hit in hits:
        kind = hit.chunk.resource_type
        if seen[kind] < cap:
            kept.append(hit)
            seen[kind] += 1
        else:
            overflow.append(hit)
        if len(kept) == k:
            break

    # If capping left room (few distinct types), refill from what was skipped
    # rather than returning fewer results than asked for.
    if len(kept) < k:
        kept.extend(overflow[: k - len(kept)])

    return [Hit(chunk=h.chunk, score=h.score, rank=i) for i, h in enumerate(kept, start=1)]


def mmr(
    hits: list[Hit],
    vectors: list[list[float]],
    *,
    k: int,
    lambda_: float = 0.7,
) -> list[Hit]:
    """Maximal Marginal Relevance over already-ranked candidates.

    `lambda_` trades relevance against novelty: 1.0 is the flat ranking, 0.0
    picks purely for dissimilarity. Vectors must be normalised (the index stores
    them that way), so a dot product is cosine similarity.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    if len(hits) != len(vectors):
        raise ValueError(f"{len(hits)} hits but {len(vectors)} vectors")

    remaining = list(range(len(hits)))
    chosen: list[int] = []

    while remaining and len(chosen) < k:
        best_i, best_score = remaining[0], float("-inf")
        for i in remaining:
            relevance = hits[i].score
            if chosen:
                novelty = max(dot(vectors[i], vectors[j]) for j in chosen)
            else:
                novelty = 0.0
            score = lambda_ * relevance - (1.0 - lambda_) * novelty
            if score > best_score:
                best_i, best_score = i, score
        chosen.append(best_i)
        remaining.remove(best_i)

    return [
        Hit(chunk=hits[i].chunk, score=hits[i].score, rank=rank)
        for rank, i in enumerate(chosen, start=1)
    ]


def interleave_by_type(hits: list[Hit], *, k: int) -> list[Hit]:
    """Round-robin across resource types, best-first within each.

    A flat cap has to guess how many rows an answer needs: capping at 2 rescues
    a lone diagnosis but truncates a legitimate list of five vaccinations.
    Interleaving sidesteps the guess — every resource type gets its best hit
    before any type gets its second, so a buried diagnosis surfaces early while
    a genuinely repeated answer still fills the remaining slots in order.
    """
    buckets: dict[str, list[Hit]] = {}
    for hit in hits:
        buckets.setdefault(hit.chunk.resource_type, []).append(hit)

    # Visit types in order of their best hit, so the strongest match stays first.
    order = sorted(buckets, key=lambda kind: -buckets[kind][0].score)

    picked: list[Hit] = []
    depth = 0
    while len(picked) < k:
        added = False
        for kind in order:
            bucket = buckets[kind]
            if depth < len(bucket):
                picked.append(bucket[depth])
                added = True
                if len(picked) == k:
                    break
        if not added:
            break
        depth += 1

    return [Hit(chunk=h.chunk, score=h.score, rank=i) for i, h in enumerate(picked, start=1)]


__all__ = ["cap_per_resource_type", "interleave_by_type", "mmr"]
