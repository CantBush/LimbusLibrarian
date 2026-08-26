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
from limbus_librarian.graph import run_ask
from limbus_librarian.ingest.pipeline import (
    incremental_since,
    ingest_connector,
    ingest_incremental,
    load_documents,
    record_incremental_success,
)
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
    search_p = sub.add_parser("search", help="Run retrieval")
    search_p.add_argument("query")
    search_p.add_argument("--config", default="hybrid")
    ask_p = sub.add_parser("ask", help="Run the LangGraph ask path")
    ask_p.add_argument("query")
    ask_p.add_argument("--config", default="hybrid_rerank_refine")
    ask_p.add_argument("--debug", action="store_true")
    eval_p = sub.add_parser("eval", help="Retrieval evaluation on gold set")
    eval_p.add_argument("--config", default="hybrid")
    eval_p.add_argument(
        "--gold",
        default="wiki_v1",
        help="Gold set name from data/eval/gold (default: wiki_v1)",
    )
    eval_p.add_argument(
        "--compare",
        action="store_true",
        help="Compare lexical, vector, hybrid, reranked, and graph retrieval",
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

    if args.cmd == "eval":
        gold_path = settings.gold_path_for(args.gold)
        if not gold_path.exists():
            raise FileNotFoundError(f"Unknown gold set: {args.gold}")
        gold = load_gold(gold_path)
        resolved = resolve_gold(gold, load_documents(settings.documents_path))
        config_ids = COMPARISON_CONFIGS if args.compare else (args.config,)
        has_evaluable_labels = any(item.relevant_doc_ids for item in resolved)
        searcher = searcher_from_disk(settings) if has_evaluable_labels else None
        summaries: dict[str, dict] = {}
        runs_dir = Path(settings.data_dir) / "eval" / "runs"
        for config_id in config_ids:
            config = load_named_config(settings.configs_dir, config_id)
            summary = evaluate_retrieval(
                resolved,
                lambda q, selected=config: searcher.search(q, selected) if searcher else [],
                k=args.k,
            )
            summary["gold_set"] = gold_path.stem
            summary["config_id"] = config.id
            summary["k"] = args.k
            out = runs_dir / f"{gold_path.stem}.{config.id}.json"
            write_eval_report(out, summary)
            summaries[config.id] = summary

        if args.compare:
            comparison = build_comparison_report(gold_path.stem, args.k, summaries)
            out = runs_dir / f"{gold_path.stem}.comparison.json"
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
