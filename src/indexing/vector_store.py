"""
Takes the outputs of parser.py + chunker.py + metadata_extractor.py and writes
them into the Postgres schema defined in src/database/models.py -- embedding
every chunk along the way.

This is the single place where a parsed, chunked, metadata-tagged document
actually becomes searchable.
"""
import logging
from datetime import datetime

from sqlalchemy import select

from src.database.connection import get_session
from src.database.models import Company, Report, ReportChunk
from src.ingestion.chunker import Chunk
from src.ingestion.metadata_extractor import ReportMetadata
from src.indexing.embeddings import embed_documents

logger = logging.getLogger(__name__)


def _parse_date(date_str: str | None) -> datetime | None:
    """
    Safely parses the LLM-extracted date string (expected format YYYY-MM-DD)
    into a real datetime for the DB's DateTime column. LLMs occasionally
    ignore format instructions -- a malformed date should degrade to None
    with a logged warning, never crash the whole document's ingestion.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.warning(f"Could not parse extracted date {date_str!r} as YYYY-MM-DD -- storing as null instead")
        return None


def _get_or_create_company(session, company_name: str) -> Company:
    clean_name = company_name.strip()
    # Try exact match or case-insensitive match
    existing = session.execute(
        select(Company).where(Company.name.ilike(f"%{clean_name}%"))
    ).scalars().first()
    
    if existing:
        return existing
        
    company = Company(name=clean_name)
    session.add(company)
    session.flush()
    return company


def store_document(
    source_filename: str,
    metadata: ReportMetadata,
    chunks: list[Chunk],
) -> int:
    """
    Persists one fully-processed document: creates/reuses the Company row,
    creates a new Report row, embeds every chunk, and writes all ReportChunk
    rows. Returns the new report's id.

    Chunks are embedded in a single batch call for efficiency rather than
    one-by-one.
    """
    if not chunks:
        raise ValueError(f"No chunks provided for {source_filename!r} -- nothing to store.")

    chunk_texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(chunk_texts)} chunk(s) for {source_filename!r}...")
    embeddings = embed_documents(chunk_texts)

    with get_session() as session:
        company = _get_or_create_company(session, metadata.company_name)

        report = Report(
            company_id=company.id,
            file_name=source_filename,
            title=metadata.title,
            company_name=metadata.company_name,
            keywords=metadata.keywords,
            summary=metadata.summary,
            date=_parse_date(metadata.date),
            report_type=metadata.report_type,
        )
        session.add(report)
        session.flush()

        for chunk, embedding in zip(chunks, embeddings):
            session.add(ReportChunk(
                report_id=report.id,
                chunk_index=chunk.chunk_index,
                chunk=chunk.text,
                is_table_merged=chunk.is_table_merged,
                embedding=embedding,
            ))

        report_id = report.id
        logger.info(f"Stored report id={report_id} ({source_filename!r}) with {len(chunks)} chunk(s)")

    return report_id
