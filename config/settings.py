"""
Centralized configuration for ATFCA.
Reads from environment variables / .env file and exposes typed settings.

Field names below match the .env keys exactly (pydantic-settings matches
case-insensitively), so nothing extra needs to be renamed in your .env.

SECURITY NOTE: the previous version of this file had real Groq/Firecrawl/
Langfuse API keys as hardcoded class defaults. Those are now gone from
source -- pydantic-settings already reads them from your .env (that's what
env_file=".env" does), so the hardcoded defaults were doing nothing except
sitting in git history / getting pasted into places like chat logs. Put your
actual keys ONLY in your local .env file (which should be .gitignore'd).
Since the old keys were shared outside this file, rotate all three
(Groq, Firecrawl, Langfuse consoles) and put the NEW values in .env.

MODEL STRATEGY (as of Sept 2026): llama-3.3-70b-versatile and
llama-3.1-8b-instant were deprecated for free/developer-tier Groq usage on
2026-08-16. openai/gpt-oss-20b and openai/gpt-oss-120b are the current
active free/dev-tier models. Split by task:
  - gpt-oss-120b for the Writer (final answer quality matters most, and it
    only runs n=4 times per query -- worth the extra latency/cost).
  - gpt-oss-20b everywhere else (Planner, Validator, Reranker NLI) -- these
    are classification/extraction-style tasks where the smaller model is
    plenty, and the Reranker alone can fire 10+ calls per query, so keeping
    those fast and cheap matters more for staying inside free-tier rate
    limits than raw model size does.
If Groq changes their lineup again, check https://console.groq.com/docs/models
and https://console.groq.com/docs/deprecations
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database (single connection URL, as used in this project's .env) ---
    postgres_db_url: str = "postgresql://postgres:postgres@localhost:5432/atfca"

    # --- Embedding ---
    # Using BAAI/bge-base-en-v1.5 instead of snowflake-arctic-embed-m-v2.0.
    # The Snowflake model's custom GTE architecture (trust_remote_code) has an
    # unresolved bug on CPU (its unpad_inputs/use_memory_efficient_attention
    # code path crashes with garbage position IDs regardless of config
    # overrides -- confirmed across multiple attempts, including a CPU-patched
    # community fork that hit the same issue). BGE is a standard BERT
    # architecture with no custom code and no GPU-kernel dependencies, so it
    # just works everywhere. Same 768-dim output, strong MTEB retrieval score.
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Metadata extraction (Extractor agent, Stage 1) ---
    metadata_extraction_model: str = "openai/gpt-oss-20b"
    metadata_prefix_chars: int = 4000  # ~1024 tokens; matches MimirRAG's "first ~2-3 pages" approach

    # --- Chunking ---
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 300
    table_merge_max_chars: int = 3600

    # --- Hybrid retrieval (Stage 6) ---
    rrf_k: int = 60  # standard RRF damping constant, matches ALCE/MimirRAG's formula

    # --- Reranking (Stage 7 -- search-result reranker, NOT the citation reranker) ---
    reranker_model_name: str = "BAAI/bge-reranker-base"

    # --- Planner (Stage 9) ---
    planner_model: str = "openai/gpt-oss-20b"

    # --- Validator (Stage 10) ---
    validator_model: str = "openai/gpt-oss-20b"
    max_validator_retries: int = 3

    # --- Writer (Stage 11) ---
    writer_model: str = "openai/gpt-oss-120b"
    writer_num_candidates: int = 4

    # --- Citation Reranker NLI (Stage 12) ---
    reranker_nli_model: str = "openai/gpt-oss-20b"

    # --- API keys (put real values in .env -- see SECURITY NOTE above) ---
    firecrawl_api_key: str = os.getenv("FIRECRAWL_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_base_url: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    @property
    def database_url(self) -> str:
        """SQLAlchemy expects the psycopg2 dialect prefix; we normalize it here
        so your .env can keep the plain 'postgresql://' form."""
        return self.postgres_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)


settings = Settings()
