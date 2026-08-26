from __future__ import annotations

import numpy as np

from limbus_librarian.index.common import chunk_to_hit, matches_filters
from limbus_librarian.index.embed import Embedder
from limbus_librarian.models import Chunk, RetrievalHit


class NumpyDenseRetriever:
    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        matrix: np.ndarray | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = matrix
        if self.matrix is None:
            self.matrix = (
                embedder.embed([c.embed_text for c in chunks])
                if chunks
                else np.zeros((0, embedder.dims), dtype=np.float32)
            )
        if len(self.matrix) != len(chunks):
            raise ValueError("Dense matrix row count does not match chunks")

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[RetrievalHit]:
        if not self.chunks or k <= 0:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)
        hits: list[RetrievalHit] = []
        rank = 1
        for idx in order:
            chunk = self.chunks[int(idx)]
            if not matches_filters(chunk, filters):
                continue
            hits.append(chunk_to_hit(chunk, float(scores[idx]), rank, self.name))
            rank += 1
            if len(hits) >= k:
                break
        return hits


class QdrantDenseRetriever:
    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        url: str,
        collection: str,
        rebuild: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.chunks = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.client = QdrantClient(url=url)
        self.collection = collection
        existing = [c.name for c in self.client.get_collections().collections]
        if rebuild or collection not in existing:
            if collection in existing:
                self.client.delete_collection(collection)
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embedder.dims, distance=Distance.COSINE),
            )
            if chunks:
                vectors = embedder.embed([c.embed_text for c in chunks])
                points = [
                    PointStruct(
                        id=i,
                        vector=vectors[i].tolist(),
                        payload={
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "document_type": c.document_type,
                            "cantos": c.cantos,
                        },
                    )
                    for i, c in enumerate(chunks)
                ]
                self.client.upsert(collection_name=collection, points=points)

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[RetrievalHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        if k <= 0:
            return []
        q_filter = None
        types = (filters or {}).get("document_types")
        if types:
            q_filter = Filter(
                must=[FieldCondition(key="document_type", match=MatchAny(any=list(types)))]
            )
        vector = self.embedder.embed([query])[0].tolist()
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=max(k, k * 4) if filters else k,
            query_filter=q_filter,
        )
        hits: list[RetrievalHit] = []
        for rank, point in enumerate(results.points, start=1):
            chunk_id = point.payload["chunk_id"]
            chunk = self.chunks.get(chunk_id)
            if chunk is None or not matches_filters(chunk, filters):
                continue
            hits.append(chunk_to_hit(chunk, float(point.score), rank, self.name))
            if len(hits) >= k:
                break
        return hits
