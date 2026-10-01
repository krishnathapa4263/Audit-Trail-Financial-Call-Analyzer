"""
Ingestion Pipeline Orchestrator.
Connects multi-format parsing, Firecrawl URL scraping, table-aware chunking,
LLM metadata extraction, and vector store indexing.
"""
import logging
import tempfile
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.scraper import scrape_url
from src.ingestion.chunker import chunk_markdown
from src.ingestion.metadata_extractor import extract_metadata
from src.indexing.vector_store import store_document  

logger = logging.getLogger(__name__)


def ingest_file(uploaded_file) -> dict:
    """
    Ingests any user-uploaded file buffer from Streamlit.
    """
    filename = uploaded_file.name
    suffix = Path(filename).suffix

    # Save Streamlit UploadedFile buffer to a temporary file path on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)

    try:
        logger.info(f"Processing uploaded file: {filename}")

        # 1. Parse document (Docling / Plain text)
        parsed_doc = parse_document(tmp_path)

        # 2. Extract structured metadata (Groq)
        metadata = extract_metadata(parsed_doc.markdown, source_filename=filename)

        # 3. Table-aware chunking
        chunks = chunk_markdown(parsed_doc.markdown)

        # 4. Save to PostgreSQL / pgvector
        store_document(chunks=chunks, metadata=metadata, source_filename=filename)

        return {
            "filename": filename,
            "company_name": metadata.company_name,
            "num_chunks": len(chunks),
        }
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def ingest_url(url: str) -> dict:
    """
    Scrapes an online SEC filing or transcript URL via Firecrawl and ingests it.
    """
    logger.info(f"Processing URL ingestion: {url}")

    # 1. Scrape web page using Firecrawl SDK
    scraped_page = scrape_url(url, only_main_content=True)

    if not scraped_page.markdown or not scraped_page.markdown.strip():
        raise ValueError(f"No readable content returned from URL: {url}")

    # 2. Extract structured metadata from scraped markdown
    metadata = extract_metadata(scraped_page.markdown, source_filename=url)

    # 3. Table-aware chunking
    chunks = chunk_markdown(scraped_page.markdown)

    # 4. Save to PostgreSQL / pgvector
    store_document(chunks=chunks, metadata=metadata, source_filename=url)

    return {
        "filename": url,
        "company_name": metadata.company_name,
        "num_chunks": len(chunks),
    }