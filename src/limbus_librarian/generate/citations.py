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
    valid = True
    kept_ids: list[str] = []
    for cid in ids:
        if cid not in allowed:
            valid = False
        elif cid not in kept_ids:
            kept_ids.append(cid)
    cleaned = answer
    if not valid:
        cleaned = CITE_RE.sub(
            lambda m: m.group(0) if m.group(1) in allowed else "",
            answer,
        )
        cleaned = re.sub(r" +", " ", cleaned).strip()
        ids = extract_citation_ids(cleaned)
        kept_ids = [i for i in ids if i in allowed]
        valid = all(i in allowed for i in ids)
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
