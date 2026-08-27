from __future__ import annotations

import re

from limbus_librarian.models import DocumentType

LORE_TYPES: frozenset[str] = frozenset(
    {
        "story_transcript",
        "character",
        "sinner",
        "abnormality",
        "faction",
        "location",
        "world",
        "event",
    }
)

_SINNERS = {
    "yi sang",
    "faust",
    "don quixote",
    "ryoshu",
    "meursault",
    "hong lu",
    "heathcliff",
    "ishmael",
    "rodion",
    "sinclair",
    "outis",
    "gregor",
}


def classify_document(title: str, categories: list[str]) -> DocumentType:
    cats = {c.lower() for c in categories}
    title_l = title.lower()

    if (
        any("identit" in c for c in cats)
        or title_l.endswith(("/identity", "/identity story"))
    ):
        return "identity"
    if (
        "e.g.o" in title_l or any("e.g.o" in c or "ego" == c for c in cats)
    ) and "abnormality" not in cats:
        return "ego"
    if any("sinner" in c for c in cats) or title_l in _SINNERS:
        return "sinner"
    if any("abnormality" in c for c in cats):
        return "abnormality"
    if any("faction" in c or "syndicate" in c or "wing" in c for c in cats):
        return "faction"
    if any("location" in c or "district" in c for c in cats):
        return "location"
    if any("character" in c for c in cats):
        return "character"
    if any(c in {"story", "cantos"} or c.startswith("canto ") for c in cats):
        return "story_transcript"
    if title_l.startswith("canto ") or "intervallo" in title_l:
        return "story_transcript"
    if any("event" in c for c in cats):
        return "event"
    if any("world" in c or "lore" in c or "the city" in c for c in cats):
        return "world"
    if "league of nine" in title_l or "the mirror" == title_l:
        return "faction" if "league" in title_l else "world"
    return "other"


def is_lore_first(document_type: DocumentType) -> bool:
    return document_type in LORE_TYPES


def detect_cantos(title: str, categories: list[str], text: str) -> list[str]:
    blob = " ".join([title, *categories, text[:2000]])
    found = re.findall(
        r"Canto\s+(IV|V|VI|VII|VIII|IX|III|II|I|\d+)",
        blob,
        flags=re.IGNORECASE,
    )
    roman = {"I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V", "VI": "VI"}
    out: list[str] = []
    for match in found:
        key = match.upper()
        label = f"Canto {roman.get(key, key)}"
        if label not in out:
            out.append(label)
    return out
