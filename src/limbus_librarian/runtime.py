from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from limbus_librarian.chunking import chunk_documents
from limbus_librarian.config import Settings
from limbus_librarian.graph.store import GraphRetriever, GraphStore
from limbus_librarian.index.bm25 import BM25Retriever
from limbus_librarian.index.dense import NumpyDenseRetriever
from limbus_librarian.index.embed import Embedder
from limbus_librarian.index.searcher import HybridSearcher
from limbus_librarian.ingest.pipeline import ingest_connector, load_documents
from limbus_librarian.models import Chunk, SourceDocument
from limbus_librarian.sources.fixture import FixtureSourceConnector


def persist_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunk_checksum(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.embed_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def dense_manifest(chunks: list[Chunk], embedder: Embedder) -> dict:
    return {
        "provider": embedder.provider,
        "model": embedder.model if embedder.provider == "openai" else "sha256-random-v1",
        "dims": embedder.dims,
        "chunk_count": len(chunks),
        "chunk_checksum": chunk_checksum(chunks),
    }


def load_or_build_dense(
    chunks: list[Chunk],
    embedder: Embedder,
    vector_path: Path,
    manifest_path: Path,
) -> np.ndarray:
    expected = dense_manifest(chunks, embedder)
    if vector_path.exists() and manifest_path.exists():
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual == expected:
                with np.load(vector_path) as stored:
                    matrix = stored["vectors"].astype(np.float32)
                if matrix.shape == (len(chunks), embedder.dims):
                    return matrix
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    matrix = embedder.embed([chunk.embed_text for chunk in chunks])
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vector_path, vectors=matrix)
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return matrix


def bootstrap_from_fixtures(settings: Settings) -> HybridSearcher:
    connector = FixtureSourceConnector(settings.fixtures_dir)
    docs = ingest_connector(
        connector,
        settings.documents_path,
        settings.catalog_path,
        lore_first=True,
    )
    return rebuild_indexes(settings, docs)


def rebuild_indexes(
    settings: Settings,
    docs: list[SourceDocument] | None = None,
) -> HybridSearcher:
    docs = docs if docs is not None else load_documents(settings.documents_path)
    chunks = chunk_documents(docs)
    persist_chunks(settings.chunks_path, chunks)
    GraphStore(settings.catalog_path).rebuild(docs)
    embedder = Embedder(
        settings.embedding_model,
        api_key=settings.openai_api_key,
        dims=settings.embedding_dims,
    )
    bm25 = BM25Retriever(chunks)
    bm25.save(settings.bm25_path)
    matrix = load_or_build_dense(
        chunks, embedder, settings.dense_path, settings.dense_manifest_path
    )
    dense = NumpyDenseRetriever(chunks, embedder, matrix=matrix)
    graph = GraphRetriever(chunks, GraphStore(settings.catalog_path))
    return HybridSearcher(bm25=bm25, dense=dense, graph=graph, rerank=True)


def update_indexes_incremental(
    settings: Settings,
    docs: list[SourceDocument],
    changed_doc_ids: set[str],
) -> HybridSearcher:
    """Replace changed documents while reusing vectors for unchanged chunk IDs."""
    old_chunks = load_chunks(settings.chunks_path)
    current_doc_ids = {document.doc_id for document in docs}
    old_doc_ids = {chunk.doc_id for chunk in old_chunks}
    chunk_revisions: dict[str, set[int]] = {}
    for chunk in old_chunks:
        chunk_revisions.setdefault(chunk.doc_id, set()).add(chunk.revision_id)
    changed_doc_ids = set(changed_doc_ids) | (old_doc_ids - current_doc_ids)
    changed_doc_ids.update(
        document.doc_id
        for document in docs
        if chunk_revisions.get(document.doc_id) != {document.revision_id}
    )
    unchanged = [
        chunk
        for chunk in old_chunks
        if chunk.doc_id not in changed_doc_ids
        and chunk.doc_id in current_doc_ids
    ]
    changed_documents = [document for document in docs if document.doc_id in changed_doc_ids]
    chunks = unchanged + chunk_documents(changed_documents)
    chunks.sort(key=lambda chunk: (chunk.title.casefold(), chunk.doc_id, chunk.ordinal))

    embedder = Embedder(
        settings.embedding_model,
        api_key=settings.openai_api_key,
        dims=settings.embedding_dims,
    )
    old_vectors = _load_matching_vectors(
        old_chunks,
        embedder,
        settings.dense_path,
        settings.dense_manifest_path,
    )
    vectors_by_id = (
        {chunk.chunk_id: old_vectors[index] for index, chunk in enumerate(old_chunks)}
        if old_vectors is not None
        else {}
    )
    missing = [chunk for chunk in chunks if chunk.chunk_id not in vectors_by_id]
    if missing:
        embedded = embedder.embed([chunk.embed_text for chunk in missing])
        vectors_by_id.update(
            {chunk.chunk_id: embedded[index] for index, chunk in enumerate(missing)}
        )
    matrix = (
        np.vstack([vectors_by_id[chunk.chunk_id] for chunk in chunks]).astype(np.float32)
        if chunks
        else np.zeros((0, embedder.dims), dtype=np.float32)
    )

    persist_chunks(settings.chunks_path, chunks)
    bm25 = BM25Retriever(chunks)
    bm25.save(settings.bm25_path)
    settings.dense_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(settings.dense_path, vectors=matrix)
    settings.dense_manifest_path.write_text(
        json.dumps(dense_manifest(chunks, embedder), indent=2) + "\n",
        encoding="utf-8",
    )
    store = GraphStore(settings.catalog_path)
    store.rebuild(docs)
    return HybridSearcher(
        bm25=bm25,
        dense=NumpyDenseRetriever(chunks, embedder, matrix=matrix),
        graph=GraphRetriever(chunks, store),
        rerank=True,
    )


def _load_matching_vectors(
    chunks: list[Chunk],
    embedder: Embedder,
    vector_path: Path,
    manifest_path: Path,
) -> np.ndarray | None:
    if not vector_path.exists() or not manifest_path.exists():
        return None
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != dense_manifest(chunks, embedder):
            return None
        with np.load(vector_path) as stored:
            matrix = stored["vectors"].astype(np.float32)
        return matrix if matrix.shape == (len(chunks), embedder.dims) else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def searcher_from_disk(settings: Settings) -> HybridSearcher:
    chunks = load_chunks(settings.chunks_path)
    if not chunks:
        return bootstrap_from_fixtures(settings)
    docs = load_documents(settings.documents_path)
    if docs and not chunks:
        chunks = chunk_documents(docs)
    embedder = Embedder(
        settings.embedding_model,
        api_key=settings.openai_api_key,
        dims=settings.embedding_dims,
    )
    matrix = load_or_build_dense(
        chunks, embedder, settings.dense_path, settings.dense_manifest_path
    )
    store = GraphStore(settings.catalog_path)
    store.rebuild(docs)
    return HybridSearcher(
        bm25=BM25Retriever(chunks),
        dense=NumpyDenseRetriever(chunks, embedder, matrix=matrix),
        graph=GraphRetriever(chunks, store),
        rerank=True,
    )
