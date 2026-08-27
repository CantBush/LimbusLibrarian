from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from limbus_librarian.generate import generate_answer
from limbus_librarian.generate.citations import validate_citations
from limbus_librarian.graph.heuristics import analyze_query, refine_query, relevance_score
from limbus_librarian.index.searcher import HybridSearcher
from limbus_librarian.models import (
    AskAnswer,
    AskTrace,
    Citation,
    QueryAnalysis,
    RetrievalConfig,
    RetrievalHit,
    TraceStep,
)

RELATIONSHIP_GRAPH_OVERLAY = True
_GRAPH_OVERLAY_K = 16
_GRAPH_OVERLAY_NEIGHBORS = 8


class GraphState(TypedDict, total=False):
    query: str
    working_query: str
    config: RetrievalConfig
    hops: int
    hits: list[RetrievalHit]
    kept: list[RetrievalHit]
    answer: str
    citations: list[Citation]
    citations_ok: bool
    gen_retries: int
    refused: bool
    trace: AskTrace
    api_key: str
    generate_model: str
    filters: dict[str, Any]
    history: list[str]


def maybe_overlay_relationship_graph(
    config: RetrievalConfig,
    analysis: QueryAnalysis | None,
    *,
    enabled: bool | None = None,
) -> RetrievalConfig:
    """Add graph retrieval for relationship questions on the default vector_only recipe."""
    if enabled is None:
        enabled = RELATIONSHIP_GRAPH_OVERLAY
    if (
        not enabled
        or analysis is None
        or analysis.question_type != "relationship"
        or config.id != "vector_only"
        or config.use_graph
    ):
        return config
    return config.model_copy(
        update={
            "use_graph": True,
            "k_graph": _GRAPH_OVERLAY_K,
            "graph_max_neighbors": _GRAPH_OVERLAY_NEIGHBORS,
        }
    )


def compiled_ask_graph(searcher: HybridSearcher):
    compiled = getattr(searcher, "_ask_graph", None)
    if compiled is None:
        compiled = build_ask_graph(searcher)
        searcher._ask_graph = compiled
    return compiled


def build_ask_graph(searcher: HybridSearcher):
    def analyze_node(state: GraphState) -> dict[str, Any]:
        entity_matches = (
            searcher.graph.store.match_entities(state["query"])
            if searcher.graph is not None
            else []
        )
        analysis = analyze_query(state["query"], entity_matches)
        filters = dict(state.get("filters") or {})
        trace = state.get("trace") or AskTrace(
            query=state["query"], config_id=state["config"].id
        )
        trace.analysis = analysis
        trace.steps.append(TraceStep(name="analyze_query", detail=analysis.model_dump()))
        return {
            "working_query": analysis.rewritten_query or state["query"],
            "filters": filters,
            "trace": trace,
        }

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        analysis = state["trace"].analysis if state.get("trace") is not None else None
        cfg = maybe_overlay_relationship_graph(state["config"], analysis)
        hits = searcher.search(
            state["working_query"],
            cfg,
            filters=state.get("filters"),
            entity_titles=(
                list(analysis.entities)
                if analysis is not None
                else None
            ),
        )
        trace = state["trace"]
        hop = state.get("hops", 0)
        trace.hits = hits
        trace.steps.append(
            TraceStep(
                name="retrieve",
                detail={
                    "query": state["working_query"],
                    "hop": hop,
                    "n_hits": len(hits),
                    "chunk_ids": [h.chunk_id for h in hits],
                    "filters": state.get("filters") or {},
                    "use_graph": cfg.use_graph,
                },
            )
        )
        return {"hits": hits, "trace": trace}

    def grade_node(state: GraphState) -> dict[str, Any]:
        cfg = state["config"]
        kept: list[RetrievalHit] = []
        for hit in state.get("hits", []):
            score = relevance_score(state["query"], f"{hit.title} {hit.text}")
            item = hit.model_copy(deep=True)
            item.relevant_score = score
            item.kept = score >= cfg.relevance_threshold
            if item.kept:
                kept.append(item)
        if not kept and state.get("hits"):
            kept = list(state["hits"][: cfg.min_kept])
            for item in kept:
                item.kept = True
        trace = state["trace"]
        trace.kept_hits = kept
        trace.steps.append(
            TraceStep(
                name="grade_and_filter",
                detail={
                    "kept": [h.chunk_id for h in kept],
                    "threshold": cfg.relevance_threshold,
                },
            )
        )
        return {"kept": kept, "trace": trace}

    def should_refine(state: GraphState) -> str:
        cfg = state["config"]
        kept = state.get("kept") or []
        hops = state.get("hops", 0)
        if len(kept) >= cfg.min_kept:
            return "generate"
        if cfg.use_refine and hops < cfg.max_hops:
            return "refine"
        return "refuse"

    def refine_node(state: GraphState) -> dict[str, Any]:
        analysis = state["trace"].analysis or analyze_query(state["query"])
        refined = refine_query(state["working_query"], analysis)
        if refined == state["working_query"]:
            refined = f"{state['query']} background history"
        trace = state["trace"]
        hops = state.get("hops", 0) + 1
        trace.hops = hops
        trace.refined_queries.append(refined)
        trace.steps.append(TraceStep(name="refine_query", detail={"refined": refined}))
        return {"working_query": refined, "hops": hops, "trace": trace}

    def refuse_node(state: GraphState) -> dict[str, Any]:
        trace = state["trace"]
        trace.steps.append(TraceStep(name="refuse", detail={"reason": "insufficient_context"}))
        return {
            "answer": (
                "I could not find enough relevant sources in the loaded corpus "
                "to answer this. Try a more specific question about a character, "
                "canto, or faction."
            ),
            "refused": True,
            "trace": trace,
        }

    def generate_node(state: GraphState) -> dict[str, Any]:
        text = generate_answer(
            state["query"],
            state.get("kept") or [],
            state.get("generate_model") or "gpt-5.6-terra",
            state.get("api_key") or "",
            history=state.get("history") or [],
        )
        cleaned, citations, ok = validate_citations(text, state.get("kept") or [])
        trace = state["trace"]
        retries = state.get("gen_retries", 0)
        if not ok and retries < 1:
            trace.steps.append(TraceStep(name="citation_retry", detail={"ok": False}))
            return {
                "answer": cleaned,
                "citations": citations,
                "citations_ok": False,
                "gen_retries": retries + 1,
                "trace": trace,
            }
        trace.steps.append(TraceStep(name="generate", detail={"ok": ok}))
        return {
            "answer": cleaned,
            "citations": citations,
            "citations_ok": ok,
            "refused": False,
            "trace": trace,
        }

    def after_generate(state: GraphState) -> str:
        if state.get("citations_ok", True):
            return "end"
        if state.get("gen_retries", 0) < 1:
            return "generate"
        return "end"

    graph = StateGraph(GraphState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("refine", refine_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        should_refine,
        {"generate": "generate", "refine": "refine", "refuse": "refuse"},
    )
    graph.add_edge("refine", "retrieve")
    graph.add_edge("refuse", END)
    graph.add_conditional_edges(
        "generate",
        after_generate,
        {"generate": "generate", "end": END},
    )
    return graph.compile()


def run_ask(
    query: str,
    searcher: HybridSearcher,
    config: RetrievalConfig,
    api_key: str = "",
    generate_model: str = "gpt-5.6-terra",
    debug: bool = False,
    filters: dict[str, Any] | None = None,
    history: list[str] | None = None,
) -> AskAnswer:
    compiled = compiled_ask_graph(searcher)
    result = compiled.invoke(
        {
            "query": query,
            "config": config,
            "hops": 0,
            "gen_retries": 0,
            "api_key": api_key,
            "generate_model": generate_model,
            "filters": filters or {},
            "history": (history or [])[-4:],
            "trace": AskTrace(query=query, config_id=config.id),
        }
    )
    kept = result.get("kept") or []
    answer = result.get("answer") or ""
    citations = result.get("citations") or []
    if not citations and kept:
        _, citations, _ = validate_citations(answer, kept)
    return AskAnswer(
        answer=answer,
        citations=citations,
        refused=bool(result.get("refused")),
        trace=result.get("trace") if debug else None,
    )
