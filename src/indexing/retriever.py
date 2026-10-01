"""
Core hybrid retrieval: metadata pre-filtering -> dense (cosine) + sparse
(IDF-weighted lexical) search in parallel -> Reciprocal Rank Fusion (RRF).

This is deliberately NOT agent-aware -- no query decomposition, no retries,
no LLM calls. It's the plain retrieval engine that the Search agent (Stage 9)
will wrap and call. Keeping it standalone means it can be tested and tuned
in isolation before any agent logic sits on top of it.

Sparse scoring detail: Postgres's built-in ts_rank has no IDF weighting --
a match on a generic word ("quarter", present in nearly every filing) scores
identically to a match on a specific word ("diluted", present in very few).
That's a real precision problem for financial text, which is dense with
generic boilerplate. We compute a small corpus-local IDF ourselves (within
the current metadata-filtered candidate pool): each matching query lexeme
contributes ln(1 + N/df) to a chunk's score, where N is the candidate pool
size and df is how many candidates that lexeme appears in. Rare, specific
matches are worth more than common ones -- the actual point of "BM25-style"
matching referenced in this project's design.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text

from src.database.connection import engine
from src.indexing.embeddings import embed_query
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int
    report_id: int
    chunk_index: int
    chunk_text: str
    company_name: str
    report_type: str | None
    report_title: str
    is_table_merged: bool
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    rerank_score: float | None = None  # populated by Stage 7's reranker, None until then


_LEXEME_RE = re.compile(r"'([^']+)'")


def get_latest_report_summaries(limit: int = 3) -> list[dict]:
    """Retrieves document-level summaries from the reports table for summary queries."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, company_name, title, report_type, summary 
                FROM reports 
                ORDER BY created_at DESC 
                LIMIT :limit
            """),
            {"limit": limit}
        ).fetchall()
        return [dict(r._mapping) for r in rows]


def _extract_lexemes(conn, query: str) -> list[str]:
    """Runs the query text through Postgres's own stemmer/stopword removal
    (plainto_tsquery) and pulls out the individual lexemes, so our IDF
    weighting uses the same normalization as the stored tsvectors."""
    result = conn.execute(
        text("SELECT plainto_tsquery('english', :query)::text AS q"), {"query": query}
    ).scalar()
    return _LEXEME_RE.findall(result or "")

HYBRID_SEARCH_SQL = """
WITH candidates AS (
    SELECT rc.id, rc.report_id, rc.chunk_index, rc.chunk, rc.embedding, rc.content_tsv, rc.is_table_merged,
           r.company_name, r.report_type, r.title AS report_title
    FROM report_chunks rc
    JOIN reports r ON r.id = rc.report_id
    WHERE (:company_name IS NULL OR r.company_name ILIKE :company_name)
      AND (:report_type IS NULL OR r.report_type = :report_type)
      AND (CAST(:date_from AS timestamptz) IS NULL OR r.date >= CAST(:date_from AS timestamptz))
      AND (CAST(:date_to AS timestamptz) IS NULL OR r.date <= CAST(:date_to AS timestamptz))
),
total_candidates AS (
    SELECT GREATEST(COUNT(*), 1) AS n FROM candidates
),
lexemes AS (
    SELECT unnest(CAST(:lexemes AS text[])) AS lexeme
),
lexeme_matches AS (
    SELECT c.id AS chunk_id, l.lexeme
    FROM candidates c
    CROSS JOIN lexemes l
    WHERE c.content_tsv @@ plainto_tsquery('english', l.lexeme)
),
doc_freq AS (
    SELECT lexeme, COUNT(DISTINCT chunk_id) AS df
    FROM lexeme_matches
    GROUP BY lexeme
),
sparse_scores AS (
    SELECT lm.chunk_id, SUM(LN(1.0 + tc.n::float / df.df)) AS score
    FROM lexeme_matches lm
    JOIN doc_freq df ON df.lexeme = lm.lexeme
    CROSS JOIN total_candidates tc
    GROUP BY lm.chunk_id
),
dense AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS rnk
    FROM candidates
),
sparse AS (
    SELECT chunk_id AS id, RANK() OVER (ORDER BY score DESC) AS rnk
    FROM sparse_scores
)
SELECT c.id AS chunk_id, c.report_id, c.chunk_index, c.chunk AS chunk_text,
       c.company_name, c.report_type, c.report_title, c.is_table_merged,
       d.rnk AS dense_rank, s.rnk AS sparse_rank,
       COALESCE(1.0 / (:rrf_k + d.rnk), 0) + COALESCE(1.0 / (:rrf_k + s.rnk), 0) AS rrf_score
FROM candidates c
LEFT JOIN dense d ON d.id = c.id
LEFT JOIN sparse s ON s.id = c.id
WHERE d.rnk IS NOT NULL OR s.rnk IS NOT NULL
ORDER BY rrf_score DESC
LIMIT :top_k;
"""

def hybrid_search(
    query: str,
    top_k: int = 10,
    company_name: str | None = None,
    report_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[SearchResult]:
    """
    Runs the full hybrid retrieval pipeline for a single query string.

    Metadata filters (company_name, report_type, date range) are applied
    BEFORE dense/sparse ranking and BEFORE IDF is computed -- i.e. both the
    ranking and the notion of "rare vs. common term" are relative to the
    filtered candidate pool, not the whole table. This matters: it's the
    difference between "best match among Apple's filings" and "best match
    overall, which might happen to be Apple."
    """
    query_vec = embed_query(query)

    with engine.connect() as conn:
        lexemes = _extract_lexemes(conn, query)
        rows = conn.execute(
            text(HYBRID_SEARCH_SQL),
            {
                "query": query,
                "lexemes": lexemes,
                "qvec": str(query_vec),
                "company_name": f"%{company_name}%" if company_name else None,
                "report_type": report_type,
                "date_from": date_from,
                "date_to": date_to,
                "rrf_k": settings.rrf_k,
                "top_k": top_k,
            },
        ).fetchall()

    logger.info(f"hybrid_search({query!r}, company_name={company_name!r}) -> {len(rows)} result(s), "
                f"lexemes={lexemes}")

    return [
        SearchResult(
            chunk_id=row.chunk_id,
            report_id=row.report_id,
            chunk_index=row.chunk_index,
            chunk_text=row.chunk_text,
            company_name=row.company_name,
            report_type=row.report_type,
            report_title=row.report_title,
            is_table_merged=row.is_table_merged,
            dense_rank=row.dense_rank,
            sparse_rank=row.sparse_rank,
            rrf_score=row.rrf_score,
        )
        for row in rows
    ]
