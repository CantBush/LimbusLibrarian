from __future__ import annotations

import re
from collections.abc import Iterable

from limbus_librarian.ingest.classify import detect_cantos
from limbus_librarian.models import DocumentType, QueryAnalysis

_DOCUMENT_TYPES = {
    "story_transcript",
    "character",
    "sinner",
    "abnormality",
    "faction",
    "location",
    "world",
    "event",
    "identity",
    "ego",
    "other",
}


def analyze_query(
    query: str,
    entity_matches: Iterable[tuple[str, str]] = (),
) -> QueryAnalysis:
    q = query.lower()
    if any(w in q for w in ("connect", "relationship", "related", "between")):
        qtype = "relationship"
    elif q.startswith("who"):
        qtype = "who"
    elif "where" in q and ("establish" in q or "story" in q or "canto" in q):
        qtype = "where_established"
    elif q.startswith("what happened") or "during canto" in q:
        qtype = "event"
    elif q.startswith("what"):
        qtype = "what"
    else:
        qtype = "other"

    matches = list(entity_matches)
    entities = [title for title, _document_type in matches]

    types: list[DocumentType] = []
    if qtype == "who":
        types = ["character", "sinner", "faction"]
    elif qtype == "event":
        types = ["story_transcript", "character"]
    elif qtype == "relationship":
        types = ["character", "sinner", "faction", "story_transcript"]
    elif "mirror" in q:
        types = ["world", "character", "sinner"]
    for _title, document_type in matches:
        if (
            document_type in _DOCUMENT_TYPES
            and document_type != "other"
            and document_type not in types
        ):
            types.append(document_type)  # type: ignore[arg-type]

    cantos = detect_cantos(query, [], "")

    return QueryAnalysis(
        question_type=qtype,  # type: ignore[arg-type]
        entities=entities,
        document_types=types,
        cantos=cantos,
        rewritten_query=query,
    )


def refine_query(query: str, analysis: QueryAnalysis) -> str:
    extra = " ".join(analysis.entities)
    refined = f"{query} {extra} lore story wiki".strip()
    return refined if refined != query else f"{query} background history"


def relevance_score(query: str, text: str) -> float:
    q = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    d = {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)
