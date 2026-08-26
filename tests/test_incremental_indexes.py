from pathlib import Path

import numpy as np

from limbus_librarian.config import Settings
from limbus_librarian.index.embed import Embedder
from limbus_librarian.ingest.pipeline import load_documents
from limbus_librarian.runtime import (
    bootstrap_from_fixtures,
    load_chunks,
    update_indexes_incremental,
)


def test_incremental_indexes_reembed_only_changed_docs_and_delete_stale_chunks(
    tmp_path: Path,
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]
    settings = Settings(data_dir=tmp_path, embedding_dims=24, openai_api_key="")
    monkeypatch.setattr(
        type(settings),
        "fixtures_dir",
        property(lambda self: root / "data" / "fixtures"),
    )
    bootstrap_from_fixtures(settings)
    documents = load_documents(settings.documents_path)
    old_chunks = load_chunks(settings.chunks_path)
    changed = next(document for document in documents if document.title == "Dongrang")
    unchanged_ids = {
        chunk.chunk_id for chunk in old_chunks if chunk.doc_id != changed.doc_id
    }
    stale_ids = {chunk.chunk_id for chunk in old_chunks if chunk.doc_id == changed.doc_id}
    revised = changed.model_copy(
        update={
            "revision_id": changed.revision_id + 1,
            "raw_wikitext": changed.raw_wikitext + "\n\n== Update ==\nNew local fixture text.",
            "plain_text": changed.plain_text + "\n\nUpdate\nNew local fixture text.",
        }
    )
    revised_documents = [
        revised if document.doc_id == changed.doc_id else document for document in documents
    ]
    embedded_texts: list[str] = []

    def fake_embed(self, texts: list[str]):
        embedded_texts.extend(texts)
        matrix = np.zeros((len(texts), self.dims), dtype=np.float32)
        if texts:
            matrix[:, 0] = 1.0
        return matrix

    monkeypatch.setattr(Embedder, "embed", fake_embed)
    update_indexes_incremental(settings, revised_documents, {changed.doc_id})
    new_chunks = load_chunks(settings.chunks_path)
    new_ids = {chunk.chunk_id for chunk in new_chunks}

    assert unchanged_ids <= new_ids
    assert stale_ids.isdisjoint(new_ids)
    changed_chunks = [chunk for chunk in new_chunks if chunk.doc_id == changed.doc_id]
    assert embedded_texts == [chunk.embed_text for chunk in changed_chunks]
