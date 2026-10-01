"""
Writer Agent

Generates n candidate answers with inline ALCE citations ([1], [2]) based on
validated chunks.

CHANGES from the previous version of this file:
  - Model now comes from settings.writer_model (config/settings.py), not a
    bare os.environ.get() call with its own separate hardcoded fallback --
    that split meant changing the model in one place silently didn't affect
    the other.
  - generate_candidates now catches per-candidate Groq API failures instead
    of letting one failed call crash the whole node; if ALL n candidates
    fail, it raises so the caller (writer_node) can fall back cleanly.
  - Slightly wider temperature spread across n candidates for more diversity
    going into the Stage 12 reranker (matches ALCE's "sample n, rerank"
    setup better than mostly-identical low-temperature outputs would).
  - rerank_node REMOVED from this file. It was defined here AND separately
    (differently) inside graph.py -- two competing implementations of the
    same node, which is exactly the kind of inconsistency that was causing
    issues. It now lives once, in reranker.py, next to the scoring logic it
    calls (matching the existing convention: search_node lives in search.py,
    validator_node lives in validator.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from groq import Groq

from config.settings import settings

logger = logging.getLogger(__name__)

NO_CONTEXT_FALLBACK = (
    "No relevant financial disclosures found to answer this question after "
    "exhausting retrieval attempts."
)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set in your .env -- required for answer generation.")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


@dataclass
class ValidatedChunk:
    index: int
    text: str
    source_document: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedCitation:
    statement: str
    cited_indices: list[int]
    has_citation: bool


@dataclass
class CandidateAnswer:
    candidate_id: int
    raw_text: str
    citations: list[ExtractedCitation] = field(default_factory=list)
    citation_density: float = 0.0
    num_factual_uncited_statements: int = 0
    num_out_of_range_citations: int = 0


def extract_citations_from_statement(statement_text: str) -> tuple[str, list[int]]:
    """Extracts integer bracketed citations (e.g., [1], [2]) and cleans statement text."""
    raw_indices = re.findall(r"\[(\d+)\]", statement_text)
    cited_indices = [int(idx) for idx in raw_indices]

    clean_text = re.sub(r"\[\d+\]", "", statement_text)
    clean_text = re.sub(r"^[\s*#\-–\d\.]+", "", clean_text).strip()
    clean_text = re.sub(r"\s+([.,;:?])", r"\1", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    return clean_text, cited_indices


def score_candidate(
    candidate_id: int,
    raw_text: str,
    validated_chunks: list[ValidatedChunk],
) -> CandidateAnswer:
    """Parses raw LLM text into a CandidateAnswer object."""
    valid_indices = {c.index for c in validated_chunks}

    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw_text) if s.strip()]

    citations: list[ExtractedCitation] = []
    out_of_range_count = 0

    for stmt in raw_sentences:
        clean_stmt, cited_indices = extract_citations_from_statement(stmt)
        if not clean_stmt:
            continue

        for idx in cited_indices:
            if idx not in valid_indices:
                out_of_range_count += 1

        has_cit = len(cited_indices) > 0
        citations.append(
            ExtractedCitation(
                statement=clean_stmt,
                cited_indices=cited_indices,
                has_citation=has_cit,
            )
        )

    num_factual_uncited = sum(
        1 for c in citations if not c.has_citation and any(char.isdigit() for char in c.statement)
    )

    total_citations = sum(len(c.cited_indices) for c in citations)
    num_statements = max(len(citations), 1)
    density = total_citations / num_statements

    return CandidateAnswer(
        candidate_id=candidate_id,
        raw_text=raw_text,
        citations=citations,
        citation_density=density,
        num_factual_uncited_statements=num_factual_uncited,
        num_out_of_range_citations=out_of_range_count,
    )


# Spread across n=4: a low-temp anchor plus increasing diversity, so the
# reranker actually has meaningfully different candidates to choose between.
_TEMPERATURES = [0.2, 0.5, 0.7, 0.9]


def generate_candidates(
    query: str,
    validated_chunks: list[ValidatedChunk],
    sub_queries: Optional[list[str]] = None,
    n: int = 4,
    model: Optional[str] = None,
) -> list[CandidateAnswer]:
    """Generates candidate answers and returns a list of CandidateAnswer instances.

    Raises RuntimeError only if EVERY candidate call fails -- a partial
    failure (e.g. 3 of 4 succeed) still returns whatever succeeded, since the
    reranker only needs at least one good candidate to work with."""
    if not validated_chunks:
        return []

    model = model or settings.writer_model
    groq_client = _get_client()

    context_str = "\n\n".join(
        f"[{c.index}] ({c.source_document}): {c.text}" for c in validated_chunks
    )

    prompt = f"""CONTEXT DOCUMENTS:
{context_str}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Provide a direct, concise factual response using ONLY facts directly mentioned in the context documents.
2. Every statement MUST end with the explicit citation bracket referencing its source document index (e.g. [1]).
3. Do not include conversational filler."""

    temps = (_TEMPERATURES * ((n // len(_TEMPERATURES)) + 1))[:n]
    candidates: list[CandidateAnswer] = []
    errors: list[str] = []

    for i, temp in enumerate(temps):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                temperature=temp,
                messages=[
                    {"role": "system", "content": "You are a precise financial analyst assistant. Cite all statements using bracketed indices like [1] or [2]."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = response.choices[0].message.content.strip()
            cand = score_candidate(candidate_id=i, raw_text=raw_text, validated_chunks=validated_chunks)
            candidates.append(cand)
        except Exception as e:
            logger.warning(f"Writer candidate {i} (temp={temp}) failed: {e}")
            errors.append(str(e))

    if not candidates:
        raise RuntimeError(f"All {n} Writer candidate generations failed: {errors}")

    if errors:
        logger.warning(f"Writer generated {len(candidates)}/{n} candidates "
                        f"({len(errors)} failed) -- proceeding with what succeeded")

    return candidates


def writer_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: Generates candidates or short-circuits to fallback if no chunks present."""
    validated_chunks = state.get("validated_chunks", [])
    query = state.get("query", "")

    if not validated_chunks:
        return {
            "final_answer": NO_CONTEXT_FALLBACK,
            "candidate_answers": [],
        }

    try:
        candidates = generate_candidates(
            query=query,
            validated_chunks=validated_chunks,
            sub_queries=state.get("sub_queries"),
            n=settings.writer_num_candidates,
        )
    except RuntimeError as e:
        logger.error(f"writer_node: {e}")
        return {
            "final_answer": NO_CONTEXT_FALLBACK,
            "candidate_answers": [],
            "error": f"writer_node: {e}",
        }

    return {"candidate_answers": candidates}
