"""
Wraps BAAI/bge-base-en-v1.5 for dense embedding generation.

BGE's asymmetric-encoding convention differs from Snowflake's arctic-embed
models: rather than a registered `prompt_name` baked into the model's
sentence-transformers config, BGE expects the query INSTRUCTION TEXT to be
manually prepended to the query string before encoding (this is the
documented pattern across BAAI's own docs, FlagEmbedding, and every
third-party BGE integration). Documents/chunks get no prefix at all.
Both sides use normalize_embeddings=True, matching BGE's official usage
(the model is trained/evaluated assuming normalized cosine similarity).

Standard BERT architecture -- no trust_remote_code, no custom attention
kernels, no GPU-only dependencies. Chosen after snowflake-arctic-embed-m-v2.0
(and a CPU-patched fork of it) both hit an unresolved custom-code bug on CPU.
"""
import logging

from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None

# BGE's documented query instruction (BAAI's own docs, FlagEmbedding, Xenova) --
# prepended to queries only, never to documents.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _get_model() -> SentenceTransformer:
    """Loads the model once and reuses it. First call downloads weights
    from Hugging Face if not already cached locally (one-time cost)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model {settings.embedding_model_name}...")
        _model = SentenceTransformer(settings.embedding_model_name)
        actual_dim = _model.get_embedding_dimension()
        if actual_dim != settings.embedding_dim:
            raise RuntimeError(
                f"Configured embedding_dim ({settings.embedding_dim}) doesn't match the "
                f"model's actual output dimension ({actual_dim}) -- update settings.py."
            )
        logger.info(f"Model loaded, embedding dimension: {actual_dim}")
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of document/chunk texts. NO query instruction prefix."""
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embeds a single search query with BGE's query instruction manually
    prepended for correct asymmetric retrieval alignment."""
    model = _get_model()
    prefixed = f"{_QUERY_INSTRUCTION}{text}"  
    embedding = model.encode([prefixed], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embedding[0].tolist()
