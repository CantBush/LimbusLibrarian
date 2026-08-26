from __future__ import annotations

from limbus_librarian.graph.store import GraphRetriever
from limbus_librarian.index.bm25 import BM25Retriever
from limbus_librarian.index.dense import NumpyDenseRetriever, QdrantDenseRetriever
from limbus_librarian.index.hybrid import reciprocal_rank_fusion
from limbus_librarian.models import RetrievalConfig, RetrievalHit
from limbus_librarian.rerank.lexical import lexical_rerank


class HybridSearcher:
    def __init__(
        self,
        bm25: BM25Retriever | None,
        dense: NumpyDenseRetriever | QdrantDenseRetriever | None,
        graph: GraphRetriever | None = None,
        rerank: bool = False,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.rerank = rerank

    def search(
        self,
        query: str,
        config: RetrievalConfig,
        filters: dict | None = None,
    ) -> list[RetrievalHit]:
        lists: list[list[RetrievalHit]] = []
        if config.use_bm25 and self.bm25 and config.k_bm25 > 0:
            lists.append(self.bm25.retrieve(query, config.k_bm25, filters))
        if config.use_dense and self.dense and config.k_dense > 0:
            lists.append(self.dense.retrieve(query, config.k_dense, filters))
        if config.use_graph and self.graph and config.k_graph > 0:
            lists.append(
                self.graph.retrieve(
                    query,
                    config.k_graph,
                    filters,
                    max_neighbors=config.graph_max_neighbors,
                )
            )
        if not lists:
            return []
        if len(lists) == 1:
            fused = lists[0][: config.k_fused]
            for i, hit in enumerate(fused, start=1):
                hit.rank = i
        else:
            fused = reciprocal_rank_fusion(lists, k=config.rrf_k, limit=config.k_fused)
        if config.use_rerank and self.rerank:
            fused = lexical_rerank(query, fused)
        return fused[: config.k_final]
