from __future__ import annotations

import json
from pathlib import Path

from limbus_librarian.chunking import chunk_documents
from limbus_librarian.config import Settings
from limbus_librarian.ingest.pipeline import ingest_connector, load_documents
from limbus_librarian.index.bm25 import BM25Retriever
from limbus_librarian.index.dense import NumpyDenseRetriever
from limbus_librarian.index.embed import Embedder
from limbus_librarian.index.searcher import HybridSearcher
from limbus_librarian.models import Chunk
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


def bootstrap_from_fixtures(settings: Settings) -> HybridSearcher:
    connector = FixtureSourceConnector(settings.fixtures_dir)
    docs = ingest_connector(
        connector,
        settings.documents_path,
        settings.catalog_path,
        lore_first=True,
    )
    chunks = chunk_documents(docs)
    persist_chunks(settings.chunks_path, chunks)
    embedder = Embedder(
        settings.embedding_model,
        api_key=settings.openai_api_key,
        dims=settings.embedding_dims,
    )
    bm25 = BM25Retriever(chunks)
    bm25.save(settings.bm25_path)
    dense = NumpyDenseRetriever(chunks, embedder)
    return HybridSearcher(bm25=bm25, dense=dense, rerank=True)


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
    return HybridSearcher(
        bm25=BM25Retriever(chunks),
        dense=NumpyDenseRetriever(chunks, embedder),
        rerank=True,
    )
