from __future__ import annotations

from limbus_librarian.models import Chunk, RetrievalHit


def chunk_to_hit(chunk: Chunk, score: float, rank: int, retriever_name: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        title=chunk.title,
        url=chunk.url,
        section_path=chunk.section_path,
        metadata={
            "document_type": chunk.document_type,
            "entities": chunk.entities,
            "cantos": chunk.cantos,
            "source_id": chunk.source_id,
        },
        score=float(score),
        rank=rank,
        retriever_name=retriever_name,
    )


def matches_filters(chunk: Chunk, filters: dict | None) -> bool:
    if not filters:
        return True
    types = filters.get("document_types")
    if types and chunk.document_type not in types:
        return False
    cantos = filters.get("cantos")
    if cantos and not set(cantos) & set(chunk.cantos):
        return False
    return True
