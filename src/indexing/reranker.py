"""
Cross-encoder reranking on top of Stage 6's hybrid search results.

Unlike the bi-encoder embedding model (Stage 5), which encodes query and
document SEPARATELY and compares vectors, a cross-encoder reads the query
and document TOGETHER through full attention -- much more accurate at
judging true relevance, but too slow to run over an entire corpus. So the
usual pattern (and the one used here): hybrid search narrows a large corpus
down to a manageable candidate set (Stage 6), then the reranker re-scores
just those candidates with a much more expensive but much more accurate
judgment of relevance.

BGE reranker models output raw, UNBOUNDED logits by default (can reasonably
range anywhere, not just 0-1) -- we explicitly apply a sigmoid activation to
get clean 0-1 relevance scores, which is BAAI's own documented recommendation
for interpreting these models' output.
"""
import logging
from dataclasses import replace

import torch
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.indexing.retriever import SearchResult

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading reranker model {settings.reranker_model_name}...")
        _model = CrossEncoder(settings.reranker_model_name, activation_fn=torch.nn.Sigmoid())
        logger.info("Reranker model loaded.")
    return _model


def rerank(query: str, results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
    """
    Re-scores and re-sorts search results by cross-encoder relevance to the
    query. Returns NEW SearchResult objects with rerank_score populated
    (original objects/list are not mutated). If top_k is given, truncates
    to the top_k results after reranking.
    """
    if not results:
        return []

    model = _get_reranker()
    pairs = [(query, r.chunk_text) for r in results]
    scores = model.predict(pairs)

    reranked = [replace(r, rerank_score=float(score)) for r, score in zip(results, scores)]
    reranked.sort(key=lambda r: r.rerank_score if r.rerank_score is not None else float("-inf"), reverse=True)

    logger.info(f"Reranked {len(results)} result(s) for query {query!r}")

    return reranked[:top_k] if top_k is not None else reranked
