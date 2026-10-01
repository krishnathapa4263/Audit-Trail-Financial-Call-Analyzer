"""
Stage 12 — Citation Reranker (ALCE citation recall & precision via NLI)

Implements ALCE citation scoring (Gao et al.):
  - Citation Recall: Full set of cited passages entails statement via NLI.
  - Citation Precision: Leave-one-out check. A citation is precise if it alone
    entails the statement OR if removing it causes non-entailment (necessary).
  - Citation F1: Harmonic mean of recall and precision.

CHANGES from the previous version of this file:
  - NLI_MODEL now comes from settings.reranker_nli_model instead of reading
    os.environ directly -- same centralization fix as writer.py/planner.py.
  - Added rerank_node, moved here from writer.py (see writer.py's changelog
    comment for why: it was defined in two places with two different
    implementations, which is exactly the kind of drift that was causing
    issues). This is now the only place rerank_node is defined.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from config.settings import settings
from src.agents.writer import CandidateAnswer, ValidatedChunk, NO_CONTEXT_FALLBACK

logger = logging.getLogger(__name__)

NLI_MODEL = settings.reranker_nli_model

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set -- required for citation reranking.")
        _client = Groq(api_key=api_key)
    return _client


_NLI_SYSTEM_PROMPT = """You are an NLI fact-checking assistant. Determine whether the PREMISE fully supports or entails the HYPOTHESIS sentence.

Respond ONLY with a JSON object: {"entailed": true} or {"entailed": false}.

Rules:
- Standard financial synonyms and abbreviations in document context (e.g., AAPL = Apple, EPS = Earnings per share, Q2 2021 = Three Months Ended March 27, 2021) are valid matches.
- If all facts, numbers, dates, and metrics in HYPOTHESIS are supported by PREMISE, return true.
- If HYPOTHESIS contains specific numbers or facts contradicted by or absent from PREMISE, return false.
"""

_NLI_USER_TEMPLATE = """PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}
"""


def check_entailment(
    premise: str,
    hypothesis: str,
    client: Optional[Groq] = None,
    model: str = NLI_MODEL,
) -> bool:
    """Single NLI call checking if `premise` entails `hypothesis`.

    On any API failure, defaults to False (not entailed) rather than raising
    -- for a citation-quality check, treating an unverifiable claim as
    unsupported is the safer failure mode than silently skipping the check
    and letting a bad citation through uncounted."""
    groq_client = client or _get_client()
    try:
        response = groq_client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _NLI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _NLI_USER_TEMPLATE.format(
                        premise=premise, hypothesis=hypothesis
                    ),
                },
            ],
        )
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        raw = getattr(message, "content", None)
        if raw is None:
            raw = ""
        elif not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()
    except Exception as e:
        logger.warning(f"check_entailment call failed, defaulting to not-entailed: {e}")
        return False

    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            for key in ["entailed", "entailment", "supported", "result"]:
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.lower() in ["true", "yes", "entailed"]
        except Exception:
            pass

    lower_raw = raw.lower()
    if '"entailed": true' in lower_raw or '"entailed":true' in lower_raw:
        return True
    if '"entailed": false' in lower_raw or '"entailed":false' in lower_raw:
        return False

    return "true" in lower_raw and "false" not in lower_raw


@dataclass
class StatementScore:
    statement: str
    cited_indices: list[int]
    recalled: bool
    precise_citations: list[int]
    imprecise_citations: list[int]


@dataclass
class RerankedCandidate:
    candidate_id: int
    raw_text: str
    statement_scores: list[StatementScore]
    citation_recall: float
    citation_precision: float
    citation_f1: float

    def summary(self) -> str:
        return (
            f"[Candidate {self.candidate_id}] "
            f"recall={self.citation_recall:.2f} "
            f"precision={self.citation_precision:.2f} "
            f"F1={self.citation_f1:.2f}"
        )


def _chunk_text_by_index(validated_chunks: list[ValidatedChunk]) -> dict[int, str]:
    formatted = {}
    for c in validated_chunks:
        doc_header = f"Source Document: {c.source_document}"
        formatted[c.index] = f"[{doc_header}]\n{c.text}"
    return formatted


def _premise_for_indices(indices: list[int], chunk_texts: dict[int, str]) -> str:
    return "\n\n".join(chunk_texts[i] for i in indices if i in chunk_texts)


def score_statement(
    statement: str,
    cited_indices: list[int],
    chunk_texts: dict[int, str],
    client: Optional[Groq] = None,
) -> StatementScore:
    """Computes recall and leave-one-out necessity precision for a single statement."""
    valid_indices = [i for i in cited_indices if i in chunk_texts]

    if not valid_indices:
        return StatementScore(
            statement=statement,
            cited_indices=cited_indices,
            recalled=False,
            precise_citations=[],
            imprecise_citations=[],
        )

    full_premise = _premise_for_indices(valid_indices, chunk_texts)
    recalled = check_entailment(full_premise, statement, client=client)

    precise: list[int] = []
    imprecise: list[int] = []

    if not recalled:
        imprecise = list(valid_indices)
    elif len(valid_indices) == 1:
        precise = list(valid_indices)
    else:
        for idx in valid_indices:
            alone_premise = chunk_texts[idx]
            alone_entails = check_entailment(alone_premise, statement, client=client)
            if alone_entails:
                precise.append(idx)
                continue

            rest = [i for i in valid_indices if i != idx]
            rest_premise = _premise_for_indices(rest, chunk_texts)
            rest_entails = (
                check_entailment(rest_premise, statement, client=client) if rest else False
            )
            if not rest_entails:
                precise.append(idx)
            else:
                imprecise.append(idx)

    return StatementScore(
        statement=statement,
        cited_indices=cited_indices,
        recalled=recalled,
        precise_citations=precise,
        imprecise_citations=imprecise,
    )


def score_candidate_citations(
    candidate: CandidateAnswer,
    validated_chunks: list[ValidatedChunk],
    client: Optional[Groq] = None,
) -> RerankedCandidate:
    """Computes ALCE citation recall/precision/F1 for a candidate answer."""
    chunk_texts = _chunk_text_by_index(validated_chunks)
    groq_client = client or _get_client()

    cited_statements = [c for c in candidate.citations if c.has_citation]

    statement_scores = [
        score_statement(s.statement, s.cited_indices, chunk_texts, client=groq_client)
        for s in cited_statements
    ]

    if statement_scores:
        recall = sum(1 for s in statement_scores if s.recalled) / len(statement_scores)
        total_citations = sum(
            len(s.precise_citations) + len(s.imprecise_citations) for s in statement_scores
        )
        total_precise = sum(len(s.precise_citations) for s in statement_scores)
        precision = (total_precise / total_citations) if total_citations else 0.0
    else:
        recall = 0.0
        precision = 0.0

    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    return RerankedCandidate(
        candidate_id=candidate.candidate_id,
        raw_text=candidate.raw_text,
        statement_scores=statement_scores,
        citation_recall=recall,
        citation_precision=precision,
        citation_f1=f1,
    )


def rerank_candidates(
    candidates: list[CandidateAnswer],
    validated_chunks: list[ValidatedChunk],
    client: Optional[Groq] = None,
) -> list[RerankedCandidate]:
    """Scores all candidate answers and returns them sorted best-F1 first."""
    groq_client = client or _get_client()
    scored = [
        score_candidate_citations(c, validated_chunks, client=groq_client)
        for c in candidates
    ]
    return sorted(scored, key=lambda r: r.citation_f1, reverse=True)


def print_rerank_report(ranked: list[RerankedCandidate]) -> None:
    print("=" * 80)
    print("CITATION RERANK REPORT (best first)")
    print("=" * 80)
    for r in ranked:
        print(r.summary())
        for s in r.statement_scores:
            status = "RECALLED" if s.recalled else "NOT RECALLED"
            print(
                f"    [{status}] cites={s.cited_indices} "
                f"precise={s.precise_citations} imprecise={s.imprecise_citations} "
                f"| {s.statement[:90]}"
            )
        print()


def rerank_node(state: dict) -> dict:
    """LangGraph node: reranks writer_node's candidates using NLI citation
    scoring and sets final_answer. Skips entirely if writer_node already set
    the no-context fallback (see graph.py's route_writer)."""
    if state.get("final_answer") == NO_CONTEXT_FALLBACK:
        return {}

    candidates = state.get("candidate_answers", [])
    validated_chunks = state.get("validated_chunks", [])

    if not candidates or not validated_chunks:
        return {
            "final_answer": NO_CONTEXT_FALLBACK,
            "final_answer_candidate_id": None,
            "citation_scores": [],
        }

    ranked = rerank_candidates(candidates, validated_chunks)
    best = ranked[0]

    return {
        "final_answer": best.raw_text,
        "final_answer_candidate_id": best.candidate_id,
        "citation_scores": [
            {
                "candidate_id": r.candidate_id,
                "citation_recall": r.citation_recall,
                "citation_precision": r.citation_precision,
                "citation_f1": r.citation_f1,
            }
            for r in ranked
        ],
    }
