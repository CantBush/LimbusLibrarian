from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentType = Literal[
    "story_transcript",
    "character",
    "sinner",
    "abnormality",
    "faction",
    "location",
    "world",
    "event",
    "identity",
    "ego",
    "other",
]

QuestionType = Literal[
    "who",
    "what",
    "relationship",
    "event",
    "where_established",
    "other",
]


class SourceDocument(BaseModel):
    doc_id: str
    source_id: str
    url: str
    title: str
    namespace: int = 0
    page_id: int
    revision_id: int
    document_type: DocumentType
    categories: list[str] = Field(default_factory=list)
    section_outline: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    cantos: list[str] = Field(default_factory=list)
    retrieved_at: str
    last_modified: str | None = None
    license: str = "CC-BY-SA-4.0"
    attribution_text: str = (
        "Text adapted from the Limbus Company Wiki (wiki.gg), CC BY-SA 4.0."
    )
    raw_wikitext: str = ""
    plain_text: str = ""
    infobox: dict[str, str] = Field(default_factory=dict)
    corpus_version: str = ""


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    section_path: str
    section_title: str
    text: str
    embed_text: str
    token_count: int
    ordinal: int
    document_type: DocumentType
    entities: list[str] = Field(default_factory=list)
    cantos: list[str] = Field(default_factory=list)
    source_id: str
    revision_id: int
    license: str = "CC-BY-SA-4.0"


class RetrievalHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    title: str
    url: str
    section_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    rank: int
    retriever_name: str
    score_components: dict[str, float] = Field(default_factory=dict)
    kept: bool | None = None
    relevant_score: float | None = None


class QueryAnalysis(BaseModel):
    question_type: QuestionType = "other"
    entities: list[str] = Field(default_factory=list)
    document_types: list[DocumentType] = Field(default_factory=list)
    cantos: list[str] = Field(default_factory=list)
    rewritten_query: str = ""


class TraceStep(BaseModel):
    name: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AskTrace(BaseModel):
    query: str
    config_id: str
    hops: int = 0
    analysis: QueryAnalysis | None = None
    hits: list[RetrievalHit] = Field(default_factory=list)
    kept_hits: list[RetrievalHit] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    refined_queries: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    section_path: str
    snippet: str


class AskAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool = False
    disclaimer: str = (
        "Limbus Librarian is unofficial and not affiliated with Project Moon. "
        "Answers are grounded in retrieved corpus text; wiki writing is CC BY-SA 4.0."
    )
    trace: AskTrace | None = None


class RetrievalConfig(BaseModel):
    id: str
    k_dense: int = 40
    k_bm25: int = 40
    k_graph: int = 16
    k_fused: int = 50
    k_final: int = 8
    use_dense: bool = True
    use_bm25: bool = True
    use_graph: bool = False
    use_rerank: bool = False
    rerank_backend: Literal["none", "lexical", "cross_encoder"] = "none"
    use_refine: bool = False
    graph_max_neighbors: int = 8
    rrf_k: int = 60
    min_kept: int = 2
    relevance_threshold: float = 0.15
    max_hops: int = 0

    @property
    def effective_rerank_backend(self) -> Literal["none", "lexical", "cross_encoder"]:
        """Resolve the backend while retaining old `use_rerank` config compatibility."""
        if "rerank_backend" in self.model_fields_set:
            return self.rerank_backend
        return "lexical" if self.use_rerank else "none"
