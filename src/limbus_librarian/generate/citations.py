from __future__ import annotations

import re

from limbus_librarian.models import Citation, RetrievalHit

CITE_RE = re.compile(r"\[cite:([a-zA-Z0-9]+)\]")


def extract_citation_ids(answer: str) -> list[str]:
    return CITE_RE.findall(answer)


def validate_citations(
    answer: str,
    kept: list[RetrievalHit],
) -> tuple[str, list[Citation], bool]:
    allowed = {h.chunk_id: h for h in kept}
    ids = extract_citation_ids(answer)
    valid = all(cid in allowed for cid in ids)
    kept_ids: list[str] = []
    for cid in ids:
        if cid in allowed and cid not in kept_ids:
            kept_ids.append(cid)
    citations = [
        Citation(
            chunk_id=cid,
            doc_id=allowed[cid].doc_id,
            title=allowed[cid].title,
            url=allowed[cid].url,
            section_path=allowed[cid].section_path,
            snippet=allowed[cid].text[:280],
        )
        for cid in kept_ids
    ]
    citation_numbers = {citation.chunk_id: index for index, citation in enumerate(citations, 1)}
    cleaned = CITE_RE.sub(
        lambda match: (
            f"[{citation_numbers[match.group(1)]}]"
            if match.group(1) in citation_numbers
            else ""
        ),
        answer,
    )
    # Keep adjacent citations compact even when the model separated tags with spaces.
    cleaned = re.sub(r"(\[\d+\])\s+(?=\[\d+\])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    if not citations and kept:
        # attach sources even if the model omitted tags
        for hit in kept[:5]:
            citations.append(
                Citation(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    title=hit.title,
                    url=hit.url,
                    section_path=hit.section_path,
                    snippet=hit.text[:280],
                )
            )
    return cleaned, citations, valid
