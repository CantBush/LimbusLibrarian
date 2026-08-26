from __future__ import annotations

import re

from limbus_librarian.models import Chunk, RetrievalHit

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def canto_number(value: str | int | None) -> int | None:
    """Return a comparable canto number from values such as ``Canto IV``."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = str(value).strip().upper()
    match = re.search(r"(?:CANTO\s*)?([0-9]+|[IVXLC]+)\b", text)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    total = 0
    previous = 0
    for character in reversed(token):
        current = _ROMAN_VALUES.get(character)
        if current is None:
            return None
        total += -current if current < previous else current
        previous = max(previous, current)
    return total or None


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
    max_canto = canto_number(filters.get("max_canto"))
    if max_canto is not None:
        known = [number for canto in chunk.cantos if (number := canto_number(canto)) is not None]
        if known and any(number > max_canto for number in known):
            return False
    return True
