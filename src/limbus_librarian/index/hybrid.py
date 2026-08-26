from __future__ import annotations

from collections import defaultdict

from limbus_librarian.models import RetrievalHit


def reciprocal_rank_fusion(
    hit_lists: list[list[RetrievalHit]],
    k: int = 60,
    limit: int = 50,
) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievalHit] = {}
    components: dict[str, dict[str, float]] = defaultdict(dict)
    for hits in hit_lists:
        for hit in hits:
            rrf = 1.0 / (k + hit.rank)
            scores[hit.chunk_id] += rrf
            components[hit.chunk_id][hit.retriever_name] = hit.score
            components[hit.chunk_id][f"rrf_{hit.retriever_name}"] = rrf
            if hit.chunk_id not in by_id:
                by_id[hit.chunk_id] = hit
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    fused: list[RetrievalHit] = []
    for rank, (chunk_id, score) in enumerate(ordered, start=1):
        hit = by_id[chunk_id].model_copy(deep=True)
        hit.score = score
        hit.rank = rank
        hit.retriever_name = "hybrid_rrf"
        hit.score_components = components[chunk_id]
        fused.append(hit)
    return fused
