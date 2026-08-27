from __future__ import annotations

import math
import re

from limbus_librarian.models import RetrievalHit


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def lexical_rerank(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    q = _tokens(query)
    scored: list[RetrievalHit] = []
    for hit in hits:
        d = _tokens(hit.text + " " + hit.title)
        overlap = len(q & d)
        denom = math.sqrt(len(q) * len(d)) if q and d else 1.0
        score = overlap / denom
        updated = hit.model_copy(deep=True)
        updated.score_components = {**hit.score_components, "rerank": score}
        updated.score = score
        scored.append(updated)
    scored.sort(key=lambda h: h.score, reverse=True)
    for i, hit in enumerate(scored, start=1):
        hit.rank = i
        hit.retriever_name = f"{hit.retriever_name}+rerank"
    return scored


class CrossEncoderReranker:
    """Optional sentence-transformers cross-encoder, loaded on first use."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-encoder reranking requires `pip install -e \".[rerank]\"`."
                ) from exc

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        model = self._load()
        pairs = [(query, f"{hit.title}\n{hit.text}") for hit in hits]
        scores = model.predict(pairs)
        updated: list[RetrievalHit] = []
        for hit, score in zip(hits, scores, strict=False):
            item = hit.model_copy(deep=True)
            item.score_components = {**hit.score_components, "rerank": float(score)}
            item.score = float(score)
            updated.append(item)
        updated.sort(key=lambda h: h.score, reverse=True)
        for i, hit in enumerate(updated, start=1):
            hit.rank = i
            hit.retriever_name = f"{hit.retriever_name}+cross_encoder"
        return updated
