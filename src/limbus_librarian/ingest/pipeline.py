from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from limbus_librarian.ingest.classify import is_lore_first
from limbus_librarian.ingest.parse import raw_to_document
from limbus_librarian.models import SourceDocument
from limbus_librarian.sources import SourceConnector


@dataclass(frozen=True)
class IncrementalIngestResult:
    documents: list[SourceDocument]
    changed_doc_ids: set[str]
    deleted_doc_ids: set[str]
    fetched_page_ids: set[int]
    since: str
    until: str


def ingest_connector(
    connector: SourceConnector,
    documents_path: Path,
    catalog_path: Path,
    lore_first: bool = True,
    state_path: Path | None = None,
    batch_size: int = 50,
    restart: bool = False,
) -> list[SourceDocument]:
    documents_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    is_resume = state_path is not None and state_path.exists() and not restart
    state = _load_or_create_state(connector, state_path, restart)
    _write_state(state_path, state)
    corpus_version = state["corpus_version"]
    existing = (
        {doc.page_id: doc for doc in load_documents(documents_path)}
        if is_resume
        else {}
    )
    completed = set(state.get("completed_page_ids", [])) & set(existing)
    # Skipped non-lore pages have no document row, so preserve their completed state too.
    completed.update(state.get("skipped_page_ids", []))
    skipped_documents = {
        int(item["page_id"]): item
        for item in state.get("skipped_documents", [])
        if "page_id" in item
    }
    listings = state["listings"]
    pending = [item for item in listings if item["page_id"] not in completed]
    batch_size = max(1, batch_size)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        page_ids = [item["page_id"] for item in batch]
        fetch_many = getattr(connector, "fetch_pages", None)
        pages = (
            fetch_many(page_ids)
            if callable(fetch_many)
            else [connector.fetch_page(page_id) for page_id in page_ids]
        )
        returned_ids = {page.page_id for page in pages}
        missing_ids = set(page_ids) - returned_ids
        if missing_ids:
            raise RuntimeError(f"MediaWiki did not return page ids: {sorted(missing_ids)}")
        skipped = set(state.get("skipped_page_ids", []))
        for page in pages:
            doc = raw_to_document(page, corpus_version)
            if lore_first and not is_lore_first(doc.document_type):
                skipped.add(page.page_id)
                skipped_documents[page.page_id] = {
                    "page_id": page.page_id,
                    "title": doc.title,
                    "document_type": doc.document_type,
                    "categories": doc.categories,
                }
                continue
            existing[doc.page_id] = doc
            completed.add(doc.page_id)
        state["completed_page_ids"] = sorted(completed)
        state["skipped_page_ids"] = sorted(skipped)
        state["skipped_documents"] = [
            skipped_documents[page_id] for page_id in sorted(skipped_documents)
        ]
        state["status"] = "in_progress"
        docs = sorted(existing.values(), key=lambda doc: (doc.title, doc.page_id))
        _write_documents(documents_path, docs)
        _write_catalog(catalog_path, docs, corpus_version)
        _write_state(state_path, state)
    docs = sorted(existing.values(), key=lambda doc: (doc.title, doc.page_id))
    _write_documents(documents_path, docs)
    _write_catalog(catalog_path, docs, corpus_version)
    state["status"] = "complete"
    state["last_successful_change"] = state.get("started_at", _now_iso())
    _write_state(state_path, state)
    return docs


def ingest_incremental(
    connector: SourceConnector,
    documents_path: Path,
    catalog_path: Path,
    since: str,
    lore_first: bool = True,
) -> IncrementalIngestResult:
    """Apply MediaWiki recent changes without rediscovering the category corpus."""
    list_changes = getattr(connector, "list_recent_changes", None)
    fetch_many = getattr(connector, "fetch_pages", None)
    if not callable(list_changes) or not callable(fetch_many):
        raise TypeError("Incremental ingest requires recentchanges and batched page fetch support")

    existing = load_documents(documents_path)
    by_page_id = {document.page_id: document for document in existing}
    by_title = {document.title.casefold(): document for document in existing}
    latest: dict[tuple[str, int | str], dict] = {}
    for change in list_changes(since):
        page_id = int(change.get("page_id") or 0)
        title = str(change.get("title") or "")
        key: tuple[str, int | str] = (
            ("page", page_id) if page_id else ("title", title.casefold())
        )
        current = latest.get(key)
        marker = (str(change.get("timestamp") or ""), int(change.get("revision_id") or 0))
        current_marker = (
            str(current.get("timestamp") or ""),
            int(current.get("revision_id") or 0),
        ) if current else ("", -1)
        if current is None or marker >= current_marker:
            latest[key] = change

    documents = {document.doc_id: document for document in existing}
    changed_doc_ids: set[str] = set()
    deleted_doc_ids: set[str] = set()
    fetch_ids: list[int] = []
    for change in latest.values():
        page_id = int(change.get("page_id") or 0)
        title_key = str(change.get("title") or "").casefold()
        old = by_page_id.get(page_id) if page_id else by_title.get(title_key)
        if change.get("deleted"):
            if old is not None:
                documents.pop(old.doc_id, None)
                changed_doc_ids.add(old.doc_id)
                deleted_doc_ids.add(old.doc_id)
            continue
        revision_id = int(change.get("revision_id") or 0)
        if page_id and (
            old is None or revision_id == 0 or old.revision_id < revision_id
        ):
            fetch_ids.append(page_id)

    pages = fetch_many(sorted(set(fetch_ids)))
    returned_ids = {page.page_id for page in pages}
    for missing_page_id in set(fetch_ids) - returned_ids:
        old = by_page_id.get(missing_page_id)
        if old is not None:
            documents.pop(old.doc_id, None)
            changed_doc_ids.add(old.doc_id)
            deleted_doc_ids.add(old.doc_id)

    corpus_version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for page in pages:
        old = by_page_id.get(page.page_id)
        document = raw_to_document(page, corpus_version)
        if lore_first and not is_lore_first(document.document_type):
            if old is not None:
                documents.pop(old.doc_id, None)
                changed_doc_ids.add(old.doc_id)
                deleted_doc_ids.add(old.doc_id)
            continue
        documents[document.doc_id] = document
        changed_doc_ids.add(document.doc_id)

    result_documents = sorted(
        documents.values(),
        key=lambda document: (document.title.casefold(), document.page_id),
    )
    _write_documents(documents_path, result_documents)
    _write_catalog(catalog_path, result_documents, corpus_version)
    timestamps = [
        str(change.get("timestamp"))
        for change in latest.values()
        if change.get("timestamp")
    ]
    return IncrementalIngestResult(
        documents=result_documents,
        changed_doc_ids=changed_doc_ids,
        deleted_doc_ids=deleted_doc_ids,
        fetched_page_ids=returned_ids,
        since=since,
        until=max(timestamps, default=since),
    )


def incremental_since(state_path: Path, explicit: str | None = None) -> str:
    if explicit and explicit != "state":
        return explicit
    if not state_path.exists():
        raise ValueError("No ingest watermark exists; pass --since TIMESTAMP or run a full ingest")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    since = str(state.get("last_successful_change") or "")
    if not since:
        raise ValueError("No ingest watermark exists; pass --since TIMESTAMP or run a full ingest")
    return since


def record_incremental_success(
    state_path: Path,
    result: IncrementalIngestResult,
) -> None:
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    state.update(
        {
            "last_successful_change": result.until,
            "last_incremental_since": result.since,
            "last_incremental_changed_docs": len(result.changed_doc_ids),
            "status": "complete",
        }
    )
    _write_state(state_path, state)


def load_documents(documents_path: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    if not documents_path.exists():
        return docs
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(SourceDocument.model_validate_json(line))
    return docs


def _load_or_create_state(
    connector: SourceConnector,
    state_path: Path | None,
    restart: bool,
) -> dict:
    if state_path is not None and state_path.exists() and not restart:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_id") == connector.source_id and state.get("listings"):
            state.setdefault(
                "started_at",
                _corpus_version_timestamp(str(state.get("corpus_version") or ""))
                or _now_iso(),
            )
            return state
    started_at = _now_iso()
    return {
        "source_id": connector.source_id,
        "corpus_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "started_at": started_at,
        "listings": connector.list_pages(),
        "completed_page_ids": [],
        "skipped_page_ids": [],
        "status": "pending",
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _corpus_version_timestamp(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat().replace(
            "+00:00",
            "Z",
        )
    except ValueError:
        return None


def _write_documents(path: Path, docs: list[SourceDocument]) -> None:
    content = "".join(doc.model_dump_json() + "\n" for doc in docs)
    _atomic_write(path, content)


def _write_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(state, indent=2) + "\n")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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
            corpus_version TEXT,
            cantos TEXT NOT NULL DEFAULT '[]',
            section_outline TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL DEFAULT ''
        )
        """
    )
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
    }
    for column, declaration in (
        ("cantos", "TEXT NOT NULL DEFAULT '[]'"),
        ("section_outline", "TEXT NOT NULL DEFAULT '[]'"),
        ("summary", "TEXT NOT NULL DEFAULT ''"),
    ):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE documents ADD COLUMN {column} {declaration}")
    conn.execute("DELETE FROM documents")
    conn.executemany(
        """
        INSERT INTO documents (
            doc_id, title, url, document_type, page_id, revision_id, corpus_version,
            cantos, section_outline, summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(d.cantos),
                json.dumps(d.section_outline),
                next(
                    (
                        paragraph.strip()
                        for paragraph in d.plain_text.split("\n\n")
                        if paragraph.strip()
                    ),
                    "",
                )[:1200],
            )
            for d in docs
        ],
    )
    conn.commit()
    conn.close()
