from __future__ import annotations

from pathlib import Path

import bm25s

from limbus_librarian.chunking import chunk_documents
from limbus_librarian.index.common import chunk_to_hit, matches_filters
from limbus_librarian.models import Chunk, RetrievalHit, SourceDocument


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [c.embed_text for c in chunks]
        self.tokens = bm25s.tokenize(corpus, stopwords="en")
        self.index = bm25s.BM25()
        if chunks:
            self.index.index(self.tokens)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.chunks:
            self.index.save(str(path))

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[RetrievalHit]:
        if not self.chunks or k <= 0:
            return []
        q_tokens = bm25s.tokenize([query], stopwords="en")
        fetch = min(max(k * 4, k), len(self.chunks))
        results, scores = self.index.retrieve(q_tokens, k=fetch)
        hits: list[RetrievalHit] = []
        rank = 1
        for idx, score in zip(results[0], scores[0], strict=False):
            chunk = self.chunks[int(idx)]
            if not matches_filters(chunk, filters):
                continue
            hits.append(chunk_to_hit(chunk, float(score), rank, self.name))
            rank += 1
            if len(hits) >= k:
                break
        return hits


def build_bm25(docs: list[SourceDocument]) -> BM25Retriever:
    return BM25Retriever(chunk_documents(docs))
