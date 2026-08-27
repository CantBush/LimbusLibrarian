from __future__ import annotations

import hashlib
import re

from limbus_librarian.models import Chunk, SourceDocument

_HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.M)
_MAX_CHARS = 1800
_TINY_SECTION_CHARS = 40


def _chunk_id(doc_id: str, revision_id: int, section_path: str, ordinal: int) -> str:
    payload = f"{doc_id}|{revision_id}|{section_path}|{ordinal}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _split_long(text: str) -> list[str]:
    if len(text) <= _MAX_CHARS:
        return [text.strip()] if text.strip() else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= _MAX_CHARS:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                parts.append(buf)
            if len(para) <= _MAX_CHARS:
                buf = para
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= _MAX_CHARS:
                        buf = f"{buf} {sent}".strip()
                    else:
                        if buf:
                            parts.append(buf)
                        buf = sent
            buf = buf
    if buf:
        parts.append(buf)
    return parts


def chunk_document(doc: SourceDocument) -> list[Chunk]:
    text = doc.raw_wikitext or doc.plain_text
    matches = list(_HEADING.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections.append((doc.title, doc.plain_text or text))
    else:
        intro = text[: matches[0].start()].strip()
        if intro:
            intro_plain = re.sub(r"\{\{[^}]+\}\}", "", intro)
            sections.append((doc.title, intro_plain))
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end]
            sections.append((match.group(2).strip(), body))

    chunks: list[Chunk] = []
    merged_sections: list[tuple[str, str]] = []
    for section_title, body in sections:
        cleaned = _strip_markup(body)
        if not cleaned:
            continue
        if merged_sections and len(cleaned) < _TINY_SECTION_CHARS:
            prev_title, prev_body = merged_sections[-1]
            merged_sections[-1] = (prev_title, f"{prev_body}\n\n{cleaned}")
        else:
            merged_sections.append((section_title, cleaned))

    for section_title, cleaned in merged_sections:
        pieces = _split_long(cleaned)
        path = section_title if section_title == doc.title else f"{doc.title}/{section_title}"
        for ordinal, piece in enumerate(pieces):
            embed = f"{doc.title} > {section_title}\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.doc_id, doc.revision_id, path, ordinal),
                    doc_id=doc.doc_id,
                    title=doc.title,
                    url=doc.url,
                    section_path=path,
                    section_title=section_title,
                    text=piece,
                    embed_text=embed,
                    token_count=max(1, len(piece.split())),
                    ordinal=ordinal,
                    document_type=doc.document_type,
                    entities=doc.entities,
                    cantos=doc.cantos,
                    source_id=doc.source_id,
                    revision_id=doc.revision_id,
                    license=doc.license,
                )
            )
    return chunks


def chunk_documents(docs: list[SourceDocument]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc))
    return out


def _strip_markup(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    for _ in range(6):
        updated = re.sub(r"\{\{[^{}]*\}\}", "", text, flags=re.S)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^\s*[|!].*$", "", text, flags=re.M)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
