from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from limbus_librarian.ingest.classify import is_lore_first
from limbus_librarian.ingest.parse import raw_to_document
from limbus_librarian.models import SourceDocument
from limbus_librarian.sources import SourceConnector


def ingest_connector(
    connector: SourceConnector,
    documents_path: Path,
    catalog_path: Path,
    lore_first: bool = True,
) -> list[SourceDocument]:
    corpus_version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    documents_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    docs: list[SourceDocument] = []
    for listing in connector.list_pages():
        page = connector.fetch_page(listing["page_id"])
        doc = raw_to_document(page, corpus_version)
        if lore_first and not is_lore_first(doc.document_type):
            continue
        docs.append(doc)
    with documents_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(doc.model_dump_json() + "\n")
    _write_catalog(catalog_path, docs, corpus_version)
    return docs


def load_documents(documents_path: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    if not documents_path.exists():
        return docs
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(SourceDocument.model_validate_json(line))
    return docs


def _write_catalog(path: Path, docs: list[SourceDocument], corpus_version: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            document_type TEXT,
            page_id INTEGER,
            revision_id INTEGER,
            corpus_version TEXT
        )
        """
    )
    conn.execute("DELETE FROM documents")
    conn.executemany(
        """
        INSERT INTO documents (doc_id, title, url, document_type, page_id, revision_id, corpus_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                d.doc_id,
                d.title,
                d.url,
                d.document_type,
                d.page_id,
                d.revision_id,
                corpus_version,
            )
            for d in docs
        ],
    )
    conn.commit()
    conn.close()
