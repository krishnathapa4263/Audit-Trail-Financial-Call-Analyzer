"""
Shared state schema for the LangGraph agent pipeline.

This is the SINGLE canonical AgentState -- every node (planner, search,
validator, replan, writer, rerank) reads and writes these exact field names.
Previously, three different shapes of this same idea existed across
state.py, graph.py's inline AgentState, and the implicit shape validator.py/
search.py actually read from -- which is why fields like "query" vs
"original_query" and "validated_results" vs "validated_chunks" didn't line
up and silently broke the pipeline between stages. This file is now the only
place the schema is defined; every *_node function imports AgentState from
here.

Field-by-field, who writes it:

  query               -- set once at the start (initial_state)
  sub_queries         -- planner_node (initial decomposition);
                          replan_node (reformulated, on retry)
  company_name        -- planner_node
  report_type         -- planner_node
  date_from / date_to -- planner_node (parsed to datetime)
  search_results      -- search_node (overwritten each round -- current
                          round's hits only, not accumulated)
  validated_chunks    -- validator_node (ACCUMULATED across rounds: this
                          round's newly-validated chunks are merged with
                          whatever was already here, per MimirRAG's
                          A <- A U R accumulation, then re-indexed 1..N so
                          citations stay correct regardless of which retry
                          round a chunk came from)
  validation_passed   -- validator_node (True once validated_chunks is
                          non-empty; sticky across rounds -- once True, it
                          stays True even if a later round finds nothing new)
  retry_count         -- replan_node (incremented each time it runs)
  max_retries         -- set once at the start (initial_state)
  candidate_answers   -- writer_node
  final_answer        -- writer_node (ONLY on the no-context fallback path)
                          or rerank_node (on the normal path)
  final_answer_candidate_id -- rerank_node
  citation_scores     -- rerank_node
  error               -- any node, on unrecoverable failure

NOTE on serializability: validated_chunks (list[ValidatedChunk]) and
candidate_answers (list[CandidateAnswer]) are kept as dataclass instances,
not plain dicts. That's fine as long as the graph runs in-memory in a single
.invoke() call, which is all Stage 12 does. If Stage 13+ adds a LangGraph
checkpointer for persistence/resumability, convert these to
dataclasses.asdict() before they hit the checkpointer's serializer -- plain
dataclasses aren't JSON-serializable out of the box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- Input ---
    query: str

    # --- Planner output ---
    sub_queries: list[str]
    company_name: Optional[str]
    report_type: Optional[str]
    date_from: Optional[datetime]
    date_to: Optional[datetime]

    # --- Search output ---
    search_results: list[dict]

    # --- Validator output ---
    validated_chunks: list[Any]     # list[ValidatedChunk] -- Any here to avoid
                                     # a circular import with writer.py; the
                                     # real type is enforced at the call sites
    validation_passed: bool
    retry_count: int
    max_retries: int

    # --- Writer output ---
    candidate_answers: list[Any]    # list[CandidateAnswer]

    # --- Rerank output ---
    final_answer: Optional[str]
    final_answer_candidate_id: Optional[int]
    citation_scores: list[dict]

    # --- Observability (Stage 14 addition) ---
    trace_id: Optional[str]      # set by graph.py's run_query(), for feedback scoring
    trace_url: Optional[str]     # set by graph.py's run_query(), for a UI deep-link

    # --- Error propagation (any stage can set this) ---
    error: Optional[str]


def initial_state(
    query: str,
    max_retries: Optional[int] = None,
    company_name: Optional[str] = None,
    report_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> AgentState:
    """Builds a fresh AgentState for a new query, with every field explicitly
    initialized -- avoids relying on TypedDict's lack of runtime defaults.

    company_name/report_type/date_from/date_to let a caller pre-set metadata
    filters (e.g. the UI sidebar's filter toggles) BEFORE the Planner runs.
    planner_node only fills in whichever of these the caller left blank --
    see its docstring for that precedence rule -- so a manual sidebar
    selection always wins over the Planner's own guess."""
    if max_retries is None:
        # Deferred import to avoid a hard dependency on config.settings for
        # callers that want to pass max_retries explicitly (e.g. tests).
        from config.settings import settings
        max_retries = settings.max_validator_retries

    return AgentState(
        query=query,
        sub_queries=[],
        company_name=company_name,
        report_type=report_type,
        date_from=date_from,
        date_to=date_to,
        search_results=[],
        validated_chunks=[],
        validation_passed=False,
        retry_count=0,
        max_retries=max_retries,
        candidate_answers=[],
        final_answer=None,
        final_answer_candidate_id=None,
        citation_scores=[],
        trace_id=None,
        trace_url=None,
        error=None,
    )
