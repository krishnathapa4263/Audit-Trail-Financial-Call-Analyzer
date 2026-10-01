"""
Validator Agent: validates retrieved financial chunks against user queries and
sub-queries using Natural Language Inference (NLI) / relevance checks, per
MimirRAG's Validator Agent and ALCE citation standards.

If retrieved chunks fail validation (irrelevant noise, boilerplate, or off-topic),
the Validator flags validation_passed=False and provides query reformulation to
trigger a re-planning/re-retrieval loop (up to max_retries).

CHANGES from the previous version of this file:
  - validator_node now ACCUMULATES validated chunks across retry rounds
    (MimirRAG Algorithm 1's A <- A U R) instead of discarding whatever was
    already validated every time it runs. Previously, a retry that found
    fewer/different chunks than the first round would silently lose earlier
    good chunks.
  - After merging, chunks are RE-INDEXED 1..N sequentially. Without this,
    a chunk validated in round 2 would get index=1 again (validate_chunks
    numbers its input starting at 1 every call), colliding with round 1's
    chunk #1 -- which would make the Writer's "[1]" citations ambiguous
    about which round's chunk they actually point at.
  - Added replan_node, which didn't exist before despite being described as
    built in the Stage 10 writeup. This is what actually drives the retry
    loop: it calls reformulate_query() and increments retry_count.
"""
import json
import logging
import os
from typing import Any

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from config.settings import settings
from src.agents.writer import ValidatedChunk

logger = logging.getLogger(__name__)

_client: Groq | None = None

VALIDATOR_MODEL = settings.validator_model


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set -- required for chunk validation.")
        _client = Groq(api_key=api_key)
    return _client


class ChunkValidation(BaseModel):
    chunk_id: int = Field(description="ID of the chunk being evaluated")
    is_valid: bool = Field(
        description="True if the chunk directly contains factual data, figures, tables, or relevant context for the question/sub-queries; False if irrelevant, boilerplate, or off-topic"
    )
    reason: str = Field(description="Brief explanation of why the chunk is valid or invalid")


class ValidationResponse(BaseModel):
    validations: list[ChunkValidation] = Field(description="List of validation decisions for each input chunk")


class ReformulationResponse(BaseModel):
    reformulated_sub_queries: list[str] = Field(
        description="List of 1 or more reformulated, expanded, or alternative search queries to locate missing evidence"
    )


_VALIDATOR_SYSTEM_PROMPT = """You are a financial validation and Natural Language Inference (NLI) agent in an agentic RAG system.
Your job is to strictly evaluate whether retrieved financial document chunks are relevant, fact-bearing, and supportive of answering the given question and sub-queries.

Evaluation Criteria:
1. VALID (is_valid: true):
   - The chunk contains concrete financial figures, metrics, statements, table rows, or qualitative context that directly addresses or provides partial evidence for the question or any sub-query.
   - For comparative questions, chunks containing figures for one of the comparison periods are VALID.

2. INVALID (is_valid: false):
   - The chunk contains generic legal boilerplate, forward-looking safe harbor disclaimers, signatures, table of contents, or index lists.
   - The chunk discusses an unrelated company, completely different metrics, or provides zero evidentiary value.
   - The chunk contradicts or cannot support any aspect of the question.

You will receive the original question, active sub-queries, and a list of candidate chunks with their IDs and text.
Respond with ONLY a JSON object with a 'validations' list containing:
{"chunk_id": <int>, "is_valid": <bool>, "reason": "<brief justification>"}
Evaluate EVERY chunk provided. Do not include any other text."""


_REPLAN_SYSTEM_PROMPT = """You are a financial query reformulation agent in an agentic RAG system.
Previous searches using the current sub-queries failed to retrieve relevant evidence from the filing corpus.
Your task is to generate reformulated, alternative, or broader sub-queries that are more likely to match
the actual terminology used in SEC filings (10-K, 10-Q) or financial reports.

Strategies:
1. Use standard financial statement synonyms (e.g. "operating profit" -> "operating income", "sales" -> "total net sales" or "revenue").
2. Broaden the query if it was overly narrow (e.g. search for the whole section or financial highlights table).
3. Separate composite metrics into underlying line items.

Example:
Input: Question: What was Apple's operating margin in 2021? | Failed queries: ['Apple operating margin 2021']
Response: {"reformulated_sub_queries": ["Apple operating income 2021", "Apple total net sales 2021"]}

Respond with ONLY a JSON object with exactly this key: reformulated_sub_queries (list of strings). No other text."""


def _extract_chunk_fields(c: Any, fallback_id: int) -> tuple[int, str, str]:
    """Helper to extract (chunk_id, text, source_document) from either dicts or objects."""
    if isinstance(c, dict):
        cid = c.get("chunk_id", c.get("index", fallback_id))
        text = c.get("chunk_text") or c.get("text") or ""
        source = c.get("report_title") or c.get("source_document") or "Unknown"
    else:
        cid = getattr(c, "chunk_id", getattr(c, "index", fallback_id))
        text = getattr(c, "chunk_text", getattr(c, "text", ""))
        source = getattr(c, "report_title", getattr(c, "source_document", "Unknown"))
    return int(cid), str(text), str(source)


def validate_chunks(
    query: str,
    sub_queries: list[str] | None = None,
    chunks: list[Any] | None = None,
) -> tuple[list[ValidatedChunk], bool]:
    """
    Validates retrieved candidate chunks against the original query and sub-queries.
    Filters out noise and irrelevant documents.

    Returns:
        tuple (validated_chunks, validation_passed)
        - validated_chunks: list of ValidatedChunk objects that passed validation
          (index is a per-call 1..N numbering -- see validator_node for why this
          gets REASSIGNED after accumulation across retry rounds)
        - validation_passed: True if at least one chunk passed validation this
          call, False otherwise.
    """
    chunks = chunks or []
    if not chunks:
        logger.warning("validate_chunks() called with empty chunks list")
        return [], False

    client = _get_client()

    formatted_chunks = []
    chunk_map = {}
    for idx, c in enumerate(chunks, start=1):
        cid, ctext, csource = _extract_chunk_fields(c, idx)
        chunk_map[cid] = (c, cid, ctext, csource)
        formatted_chunks.append({
            "chunk_id": cid,
            "report_title": csource,
            "text": ctext[:1200],
        })

    user_payload = {
        "original_query": query,
        "sub_queries": sub_queries or [query],
        "chunks": formatted_chunks,
    }

    last_error: Exception | None = None
    model_to_use = VALIDATOR_MODEL

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": _VALIDATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, indent=2)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw_content = response.choices[0].message.content
            if not raw_content:
                raise ValueError("Validator model returned empty content")

            parsed = json.loads(raw_content)
            result = ValidationResponse.model_validate(parsed)

            validation_by_id = {v.chunk_id: v for v in result.validations}

            validated_chunks: list[ValidatedChunk] = []
            for cid, (c, orig_id, ctext, csource) in chunk_map.items():
                decision = validation_by_id.get(cid)
                if decision and decision.is_valid:
                    v_chunk = ValidatedChunk(
                        index=orig_id,
                        text=ctext,
                        source_document=csource,
                        metadata={"validation_reason": decision.reason},
                    )
                    validated_chunks.append(v_chunk)
                elif decision:
                    logger.debug(f"Chunk {cid} rejected by validator: {decision.reason}")

            passed = len(validated_chunks) > 0
            logger.info(
                f"Validated {len(chunks)} chunks -> {len(validated_chunks)} passed "
                f"(validation_passed={passed})"
            )
            return validated_chunks, passed

        except Exception as e:
            last_error = e
            logger.warning(f"Validator attempt {attempt + 1} failed: {e}")

    logger.error(f"Validator failed after retries: {last_error}. Defaulting to empty validation.")
    return [], False


def reformulate_query(
    query: str,
    current_sub_queries: list[str],
    attempt: int = 1,
) -> list[str]:
    """
    Reformulates or broadens sub-queries when previous retrieval failed validation.
    Per MimirRAG's replan mechanism (Algorithm 1, line 19).
    """
    client = _get_client()
    user_msg = f"Question: {query} | Failed queries: {current_sub_queries}"

    try:
        response = client.chat.completions.create(
            model=VALIDATOR_MODEL,
            messages=[
                {"role": "system", "content": _REPLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content
        if raw_content:
            parsed = json.loads(raw_content)
            result = ReformulationResponse.model_validate(parsed)
            if result.reformulated_sub_queries:
                logger.info(
                    f"Reformulated {len(current_sub_queries)} sub-queries into: "
                    f"{result.reformulated_sub_queries} (attempt {attempt})"
                )
                return result.reformulated_sub_queries
    except Exception as e:
        logger.warning(f"Failed to reformulate query via LLM: {e}. Falling back to default expansion.")

    fallback = [query] if query not in current_sub_queries else [f"{query} financial summary"]
    return fallback


def _dedupe_by_text(chunks: list[ValidatedChunk]) -> list[ValidatedChunk]:
    """A reformulated sub-query can re-retrieve the same underlying chunk
    text that an earlier round already validated (e.g. it shows up for both
    the original and the broadened query). Dedupe on text so it doesn't get
    cited twice under two different indices."""
    seen: set[str] = set()
    out: list[ValidatedChunk] = []
    for c in chunks:
        if c.text not in seen:
            seen.add(c.text)
            out.append(c)
    return out


def _reindex(chunks: list[ValidatedChunk]) -> list[ValidatedChunk]:
    """Reassigns .index sequentially 1..N. Required after merging chunks from
    multiple retry rounds, since validate_chunks() numbers its input 1..N on
    every call -- without this, round 2's chunk #1 collides with round 1's
    chunk #1, and the Writer's "[1]" citations become ambiguous."""
    for i, c in enumerate(chunks, start=1):
        c.index = i
    return chunks


def validator_node(state: dict) -> dict:
    """LangGraph node wrapper for chunk validation. Accumulates validated
    chunks across retry rounds rather than replacing them each time."""
    query = state.get("query", "")
    sub_queries = state.get("sub_queries", []) or [query]
    search_results = state.get("search_results", [])
    existing_valid: list[ValidatedChunk] = state.get("validated_chunks", [])

    if not search_results:
        logger.info("validator_node: no search_results this round -- keeping "
                     "whatever was already accumulated")
        merged = _reindex(_dedupe_by_text(existing_valid))
        return {"validated_chunks": merged, "validation_passed": len(merged) > 0}

    new_valid, _ = validate_chunks(query=query, sub_queries=sub_queries, chunks=search_results)

    merged = _reindex(_dedupe_by_text(existing_valid + new_valid))
    passed = len(merged) > 0

    logger.info(f"validator_node: {len(existing_valid)} previously accumulated + "
                f"{len(new_valid)} newly validated -> {len(merged)} total after dedupe "
                f"(validation_passed={passed})")

    return {"validated_chunks": merged, "validation_passed": passed}


def replan_node(state: dict) -> dict:
    """LangGraph node: reformulates sub_queries and increments retry_count.
    Routes back to 'search' (see graph.py's route_validator)."""
    query = state.get("query", "")
    current_sub_queries = state.get("sub_queries", []) or [query]
    retry_count = state.get("retry_count", 0)

    new_sub_queries = reformulate_query(query, current_sub_queries, attempt=retry_count + 1)

    return {
        "sub_queries": new_sub_queries,
        "retry_count": retry_count + 1,
    }
