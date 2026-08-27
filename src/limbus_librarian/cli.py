from __future__ import annotations

import argparse
import json
from pathlib import Path

from limbus_librarian.config import get_settings
from limbus_librarian.config_loader import load_named_config
from limbus_librarian.eval import (
    ResolvedGoldItem,
    evaluate_retrieval,
    load_gold,
    resolve_gold,
    write_eval_report,
)
from limbus_librarian.eval.model_gates import LunaStructuredRewriter, run_luna_experiment
from limbus_librarian.graph import run_ask
from limbus_librarian.ingest.pipeline import (
    incremental_since,
    ingest_connector,
    ingest_incremental,
    load_documents,
    record_incremental_success,
)
from limbus_librarian.ingest.stats import build_ingest_stats, format_ingest_stats
from limbus_librarian.runtime import (
    bootstrap_from_fixtures,
    rebuild_indexes,
    searcher_from_disk,
    update_indexes_incremental,
)
from limbus_librarian.sources.mediawiki import MediaWikiSourceConnector

COMPARISON_CONFIGS = (
    "bm25_only",
    "vector_only",
    "hybrid",
    "hybrid_rerank",
    "hybrid_graph",
)
RERANK_COMPARISON_CONFIGS = (
    "hybrid",
    "hybrid_rerank",
    "hybrid_rerank_cross_encoder",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="limbus")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest-fixtures", help="Build corpus from local fixtures")
    wiki_p = sub.add_parser("ingest-wiki", help="Ingest lore pages from MediaWiki categories")
    wiki_p.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Category name to seed (repeatable; defaults to lore-first categories)",
    )
    wiki_p.add_argument("--category-depth", type=int)
    wiki_p.add_argument("--restart", action="store_true", help="Discard resume progress")
    wiki_p.add_argument(
        "--since",
        nargs="?",
        const="state",
        metavar="TIMESTAMP",
        help="Incrementally ingest recent changes since a timestamp or saved watermark",
    )
    sub.add_parser("ingest-stats", help="Summarize the local ingested wiki catalog")
    search_p = sub.add_parser("search", help="Run retrieval")
    search_p.add_argument("query")
    search_p.add_argument("--config", default="vector_only")
    ask_p = sub.add_parser("ask", help="Run the LangGraph ask path")
    ask_p.add_argument("query")
    ask_p.add_argument("--config", default="vector_only")
    ask_p.add_argument("--debug", action="store_true")
    eval_p = sub.add_parser("eval", help="Retrieval evaluation on gold set")
    eval_p.add_argument("--config", default="vector_only")
    eval_p.add_argument(
        "--gold",
        default="wiki_v1",
        help="Gold set name from data/eval/gold (default: wiki_v1)",
    )
    eval_modes = eval_p.add_mutually_exclusive_group()
    eval_modes.add_argument(
        "--compare",
        action="store_true",
        help="Compare lexical, vector, hybrid, reranked, and graph retrieval",
    )
    eval_modes.add_argument(
        "--rerank-compare",
        action="store_true",
        help="Explicitly compare hybrid, lexical rerank, and cross-encoder rerank",
    )
    eval_modes.add_argument(
        "--luna-experiment",
        action="store_true",
        help="Explicitly run bounded Luna rewrite/relevance A/B (requires API key)",
    )
    eval_p.add_argument(
        "--experiment-limit",
        type=int,
        choices=range(1, 9),
        default=6,
        help="Maximum Luna calls (1-8, default: 6)",
    )
    eval_p.add_argument("--k", type=int, default=8)
    serve_p = sub.add_parser("serve", help="Serve the API and local dashboard")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "limbus_librarian.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    if args.cmd == "ingest-fixtures":
        bootstrap_from_fixtures(settings)
        print(f"Wrote {settings.documents_path} and {settings.chunks_path}")
        return

    if args.cmd == "ingest-wiki":
        if args.since is not None and args.restart:
            parser.error("ingest-wiki --since cannot be combined with --restart")
        categories = tuple(args.categories or settings.wiki_category_list())
        connector = MediaWikiSourceConnector(
            settings.mediawiki_api,
            settings.user_agent,
            min_interval_s=1.0,
            categories=categories,
            category_depth=(
                args.category_depth
                if args.category_depth is not None
                else settings.wiki_category_depth
            ),
            batch_size=settings.wiki_batch_size,
        )
        if args.since is not None:
            since = incremental_since(settings.ingest_state_path, args.since)
            result = ingest_incremental(
                connector,
                settings.documents_path,
                settings.catalog_path,
                since=since,
                lore_first=True,
            )
            update_indexes_incremental(settings, result.documents, result.changed_doc_ids)
            record_incremental_success(settings.ingest_state_path, result)
            print(
                f"Applied {len(result.changed_doc_ids)} changed lore document(s), "
                f"deleted {len(result.deleted_doc_ids)} stale document(s), and "
                f"fetched {len(result.fetched_page_ids)} page(s)."
            )
            print("Restart `limbus serve` or POST /v1/reload to load the new indexes.")
            return
        docs = ingest_connector(
            connector,
            settings.documents_path,
            settings.catalog_path,
            lore_first=True,
            state_path=settings.ingest_state_path,
            batch_size=settings.wiki_batch_size,
            restart=args.restart,
        )
        rebuild_indexes(settings, docs)
        print(
            f"Ingested {len(docs)} lore documents; rebuilt chunks, BM25, and "
            f"{'OpenAI' if settings.openai_api_key else 'local'} dense indexes."
        )
        print("Restart `limbus serve` or POST /v1/reload to load the new indexes.")
        return

    if args.cmd == "ingest-stats":
        state = (
            json.loads(settings.ingest_state_path.read_text(encoding="utf-8"))
            if settings.ingest_state_path.exists()
            else {}
        )
        stats = build_ingest_stats(load_documents(settings.documents_path), state)
        print(format_ingest_stats(stats))
        return

    if args.cmd == "eval":
        gold_path = settings.gold_path_for(args.gold)
        if not gold_path.exists():
            raise FileNotFoundError(f"Unknown gold set: {args.gold}")
        gold = load_gold(gold_path)
        resolved = resolve_gold(gold, load_documents(settings.documents_path))
        runs_dir = Path(settings.data_dir) / "eval" / "runs"
        if args.luna_experiment:
            out = runs_dir / f"{gold_path.stem}.luna_experiment.json"
            if not settings.openai_api_key.strip():
                report = {
                    "status": "not_executed",
                    "reason": "OPENAI_API_KEY is not configured",
                    "gold_set": gold_path.stem,
                    "baseline_config": "vector_only",
                    "model": settings.utility_model,
                    "limit": args.experiment_limit,
                }
                write_eval_report(out, report)
                print("Luna experiment not executed: OPENAI_API_KEY is not configured.")
                print(f"Wrote {out}")
                return
            config = load_named_config(settings.configs_dir, "vector_only")
            searcher = searcher_from_disk(settings)
            rewriter = LunaStructuredRewriter(
                settings.openai_api_key,
                settings.utility_model,
            )
            report = run_luna_experiment(
                resolved,
                lambda query: searcher.search(query, config),
                rewriter,
                limit=args.experiment_limit,
                k=args.k,
            )
            report.update(
                {
                    "gold_set": gold_path.stem,
                    "baseline_config": config.id,
                    "model": settings.utility_model,
                    "k": args.k,
                }
            )
            write_eval_report(out, report)
            display_keys = ("status", "n", "heuristic", "luna_rewrite_and_grade")
            print(json.dumps({key: report[key] for key in display_keys}, indent=2))
            print(f"Wrote {out}")
            return
        if args.rerank_compare:
            config_ids = RERANK_COMPARISON_CONFIGS
        elif args.compare:
            config_ids = COMPARISON_CONFIGS
        else:
            config_ids = (args.config,)
        has_evaluable_labels = any(item.relevant_doc_ids for item in resolved)
        searcher = searcher_from_disk(settings) if has_evaluable_labels else None
        summaries: dict[str, dict] = {}
        for config_id in config_ids:
            config = load_named_config(settings.configs_dir, config_id)
            summary = evaluate_retrieval(
                resolved,
                lambda q, selected=config: searcher.search(q, selected) if searcher else [],
                k=args.k,
            )
            summary["gold_set"] = gold_path.stem
            summary["config_id"] = config.id
            summary["rerank_backend"] = config.effective_rerank_backend
            summary["k"] = args.k
            out = runs_dir / f"{gold_path.stem}.{config.id}.json"
            write_eval_report(out, summary)
            summaries[config.id] = summary

        if args.compare or args.rerank_compare:
            comparison = build_comparison_report(gold_path.stem, args.k, summaries)
            suffix = "rerank_comparison" if args.rerank_compare else "comparison"
            out = runs_dir / f"{gold_path.stem}.{suffix}.json"
            write_eval_report(out, comparison)
            print(format_comparison_table(summaries))
        else:
            summary = summaries[args.config]
            out = runs_dir / f"{gold_path.stem}.{args.config}.json"
            keys = (
                "n",
                "n_evaluated",
                "n_unresolved",
                "n_partial",
                "n_with_unresolved_labels",
                "recall@k",
                "mrr",
                "ndcg@k",
            )
            print(json.dumps({key: summary[key] for key in keys}, indent=2))
        unresolved = _unresolved_items(resolved)
        if unresolved:
            print(
                f"Unresolved gold labels: {len(unresolved)} item(s). "
                "Run the live ingest, then rerun eval; no wiki IDs were fabricated."
            )
            for item in unresolved:
                labels = item.unresolved_doc_ids + item.unresolved_titles
                print(f"  {item.item.id}: {', '.join(labels)}")
        print(f"Wrote {out}")
        return

    searcher = searcher_from_disk(settings)
    config = load_named_config(settings.configs_dir, args.config)

    if args.cmd == "search":
        hits = searcher.search(args.query, config)
        print(json.dumps([h.model_dump() for h in hits], indent=2))
        return
    if args.cmd == "ask":
        answer = run_ask(
            args.query,
            searcher,
            config,
            api_key=settings.openai_api_key,
            generate_model=settings.generate_model,
            debug=args.debug,
        )
        print(answer.model_dump_json(indent=2))
        return


def _unresolved_items(items: list[ResolvedGoldItem]) -> list[ResolvedGoldItem]:
    return [item for item in items if item.unresolved_doc_ids or item.unresolved_titles]


def build_comparison_report(gold_set: str, k: int, summaries: dict[str, dict]) -> dict:
    first_summary = next(iter(summaries.values()), {})
    return {
        "gold_set": gold_set,
        "k": k,
        "unresolved_labels": first_summary.get("unresolved_labels", []),
        "configs": [
            {
                key: summary[key]
                for key in (
                    "config_id",
                    "rerank_backend",
                    "n",
                    "n_evaluated",
                    "n_unresolved",
                    "n_partial",
                    "n_with_unresolved_labels",
                    "recall@k",
                    "mrr",
                    "ndcg@k",
                )
            }
            for summary in summaries.values()
        ],
    }


def format_comparison_table(summaries: dict[str, dict]) -> str:
    header = (
        f"{'config':<18} {'eval/total':>10} {'unresolved':>10} "
        f"{'Recall@K':>10} {'MRR':>8} {'nDCG@K':>10}"
    )
    separator = "-" * len(header)
    rows = [header, separator]
    for config_id, summary in summaries.items():
        rows.append(
            f"{config_id:<18} "
            f"{summary['n_evaluated']:>4}/{summary['n']:<5} "
            f"{summary['n_unresolved']:>10} "
            f"{summary['recall@k']:>10.3f} "
            f"{summary['mrr']:>8.3f} "
            f"{summary['ndcg@k']:>10.3f}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
