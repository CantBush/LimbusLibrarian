from __future__ import annotations

import re

from limbus_librarian.models import QueryAnalysis

_ENTITY = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
KNOWN = [
    "Dongrang",
    "Yi Sang",
    "League of Nine",
    "The Mirror",
    "Mirror",
    "Canto IV",
    "K Corp",
]


def analyze_query(query: str) -> QueryAnalysis:
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

    entities = [name for name in KNOWN if name.lower() in q]
    if not entities:
        entities = [m.group(1) for m in _ENTITY.finditer(query) if m.group(1) not in {"Who", "What", "How", "Where", "Explain"}]

    types: list = []
    if qtype == "who":
        types = ["character", "sinner", "faction"]
    elif qtype == "event":
        types = ["story_transcript", "character"]
    elif qtype == "relationship":
        types = ["character", "sinner", "faction", "story_transcript"]
    elif "mirror" in q:
        types = ["world", "character", "sinner"]

    cantos = []
    if "canto iv" in q or "canto 4" in q:
        cantos = ["Canto IV"]

    return QueryAnalysis(
        question_type=qtype,  # type: ignore[arg-type]
        entities=entities,
        document_types=types,  # type: ignore[arg-type]
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
