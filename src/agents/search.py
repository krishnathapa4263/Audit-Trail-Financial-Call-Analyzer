"""
Search Agent: executes each Planner sub-query through hybrid_search() (Stage 6)
+ rerank() (Stage 7), then merges results across all sub-queries into a single
ranked list for the Validator agent.

Merging detail: a chunk retrieved by more than one sub-query (e.g. both the
"2022 profit" and "2023 profit" sub-queries happen to pull the same summary
table) is a genuinely stronger signal, not noise -- so we deduplicate by
chunk_id, keeping each chunk's BEST rerank_score across all the sub-queries
that retrieved it, rather than just concatenating everything or arbitrarily
picking whichever sub-query ran first.
"""
import logging
from dataclasses import asdict
from datetime import datetime

from sqlalchemy import text

from src.indexing.retriever import engine, hybrid_search
from src.indexing.reranker import rerank
from src.indexing.retriever import get_latest_report_summaries, SearchResult

logger = logging.getLogger(__name__)


def search(
    sub_queries: list[str],
    company_name: str | None = None,
    report_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    top_k_per_subquery: int = 10,
    rerank_top_k_per_subquery: int = 5,
) -> list[dict]:
    """
    Runs hybrid_search + rerank for each sub-query under the same metadata
    filters (all sub-queries from one Planner call share the same company/
    report_type/date constraints -- those come from the ORIGINAL question,
    not per-sub-query), then merges into one deduplicated, re-sorted list.

    Returns a list of dicts (asdict()'d SearchResult) ready to be written
    into AgentState.search_results.
    """
    if not sub_queries:
        logger.warning("search() called with no sub_queries -- returning empty results")
        return []

    best_by_chunk_id: dict[int, dict] = {}

    for sub_query in sub_queries:
        raw_results = hybrid_search(
            sub_query,
            top_k=top_k_per_subquery,
            company_name=company_name,
            report_type=report_type,
            date_from=date_from,
            date_to=date_to,
        )
        if not raw_results:
            logger.warning(f"No hybrid search results for sub-query {sub_query!r}")
            continue

        reranked = rerank(sub_query, raw_results, top_k=rerank_top_k_per_subquery)

        for r in reranked:
            existing = best_by_chunk_id.get(r.chunk_id)
            if existing is None or (r.rerank_score or 0) > existing["rerank_score"]:
                best_by_chunk_id[r.chunk_id] = asdict(r)

    merged = sorted(best_by_chunk_id.values(), key=lambda d: d["rerank_score"] or 0, reverse=True)

    logger.info(f"search() across {len(sub_queries)} sub-quer(y/ies) -> {len(merged)} unique chunk(s) "
                f"(from {len(sub_queries)} searches)")

    return merged


def search_node(state: dict) -> dict:
    """LangGraph node wrapper for the search agent."""
    query = state.get("query", "")
    sub_queries = state.get("sub_queries", []) or [query]

    all_results = []
    is_summary_query = any(kw in query.lower() for kw in ["summary", "summarize", "overview", "recently scraped"])
    if is_summary_query:
        summaries = get_latest_report_summaries(limit=2)
        for s in summaries:
            if s.get("summary"):
                all_results.append(
                    SearchResult(
                        chunk_id=-1,
                        report_id=s["id"],
                        chunk_index=0,
                        chunk_text=f"DOCUMENT SUMMARY [{s['title']}]: {s['summary']}",
                        company_name=s["company_name"],
                        report_type=s["report_type"],
                        report_title=s["title"],
                        is_table_merged=False,
                        dense_rank=1,
                        sparse_rank=1,
                        rrf_score=1.0,
                        rerank_score=1.0,)
                )

    results = search(
        sub_queries=sub_queries,
        company_name=state.get("company_name"),
        report_type=state.get("report_type"),
        date_from=state.get("date_from"),
        date_to=state.get("date_to"),
    )

    all_results.extend(results)
    return {"search_results": all_results}

