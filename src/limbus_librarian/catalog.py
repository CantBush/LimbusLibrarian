from __future__ import annotations

import sqlite3
from pathlib import Path

from limbus_librarian.graph.store import GraphStore
from limbus_librarian.ingest.pipeline import load_documents
from limbus_librarian.models import SourceDocument


def document_summary(document: SourceDocument, limit: int = 420) -> str:
    paragraph = next(
        (
            part.strip()
            for part in document.plain_text.split("\n\n")
            if part.strip()
        ),
        "",
    )
    if len(paragraph) <= limit:
        return paragraph
    return paragraph[: limit - 1].rstrip() + "…"


class CatalogStore:
    """Read catalog identity from SQLite and rich text from the document archive."""

    def __init__(self, catalog_path: Path, documents_path: Path) -> None:
        self.catalog_path = catalog_path
        self.documents_path = documents_path
        self._documents: dict[str, SourceDocument] = {}
        self._catalog_ids: set[str] = set()
        self.refresh()

    def refresh(self) -> None:
        documents = load_documents(self.documents_path)
        self._documents = {document.doc_id: document for document in documents}
        self._catalog_ids = set()
        if self.catalog_path.exists():
            try:
                with sqlite3.connect(self.catalog_path) as connection:
                    rows = connection.execute("SELECT doc_id FROM documents").fetchall()
                self._catalog_ids = {str(row[0]) for row in rows}
            except sqlite3.Error:
                self._catalog_ids = set()
        if not self._catalog_ids:
            self._catalog_ids = set(self._documents)

    def list(
        self,
        *,
        document_types: set[str] | None = None,
        query: str = "",
        canto: str = "",
        page: int = 1,
        per_page: int = 24,
    ) -> dict:
        needle = query.strip().casefold()
        canto_needle = canto.strip().casefold()
        documents = []
        for doc_id in self._catalog_ids:
            document = self._documents.get(doc_id)
            if document is None:
                continue
            if document_types and document.document_type not in document_types:
                continue
            searchable = f"{document.title} {document.plain_text}".casefold()
            if needle and needle not in searchable:
                continue
            if canto_needle and not any(
                value.casefold() == canto_needle for value in document.cantos
            ):
                continue
            documents.append(document)
        documents.sort(key=lambda document: (document.title.casefold(), document.doc_id))
        total = len(documents)
        start = (page - 1) * per_page
        items = [
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "url": document.url,
                "document_type": document.document_type,
                "cantos": document.cantos,
                "summary": document_summary(document),
            }
            for document in documents[start : start + per_page]
        ]
        return {
            "items": items,
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page,
        }

    def get(self, doc_id: str) -> dict | None:
        if doc_id not in self._catalog_ids:
            return None
        document = self._documents.get(doc_id)
        if document is None:
            return None
        return {
            "doc_id": document.doc_id,
            "title": document.title,
            "url": document.url,
            "document_type": document.document_type,
            "cantos": document.cantos,
            "summary": document_summary(document, limit=1200),
            "sections": document.section_outline,
            "license": document.license,
            "attribution_text": document.attribution_text,
            "related": self.related(doc_id),
        }

    def related(self, doc_id: str, limit: int = 12) -> list[dict]:
        if doc_id not in self._catalog_ids:
            return []
        return GraphStore(self.catalog_path).related(doc_id, limit=limit)
