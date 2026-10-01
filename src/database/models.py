"""
Relational + vector schema for ATFCA.

Design follows the MimirRAG entity structure (Companies -> Reports -> Report Chunks)
plus a feedback table for the Streamlit thumbs-up/down -> Langfuse loop.

Each chunk carries BOTH a dense embedding column (pgvector, HNSW-indexed) and a
generated tsvector column (GIN-indexed) so hybrid dense+sparse retrieval and RRF
fusion can run natively in Postgres, without a second search system.
"""
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ForeignKey, String, Text, DateTime, Computed, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import settings


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    company_form: Mapped[str | None] = mapped_column(String(50))   # e.g. "Inc.", "A/S"
    sector: Mapped[str | None] = mapped_column(String(100))
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    reports: Mapped[list["Report"]] = relationship(back_populates="company", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Company id={self.id} name={self.name!r}>"


class Report(Base):
    """One filing / earnings call transcript / scraped document."""
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), nullable=False)

    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)          # "[Report Type] - [Year] ([Company Name])"
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)   # denormalized for fast metadata filtering
    company_form: Mapped[str | None] = mapped_column(String(50))
    sector: Mapped[str | None] = mapped_column(String(100))
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    summary: Mapped[str | None] = mapped_column(Text)
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   # fiscal period-end or publication date
    q_period: Mapped[str | None] = mapped_column(String(20))                 # e.g. "Q2-2024"
    report_type: Mapped[str | None] = mapped_column(String(50))              # 10-K, 10-Q, Earnings Call, Press Release
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    company: Mapped["Company"] = relationship(back_populates="reports")
    chunks: Mapped[list["ReportChunk"]] = relationship(back_populates="report", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Report id={self.id} title={self.title!r}>"


class ReportChunk(Base):
    """
    A single retrieval unit. Table-aware merged chunks (up to table_merge_max_chars)
    are stored here just like any other chunk -- the merge happens upstream in
    the chunker, not in this schema.
    """
    __tablename__ = "report_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"), nullable=False)

    chunk_index: Mapped[int] = mapped_column(nullable=False)   # position within the source document
    title: Mapped[str | None] = mapped_column(String(500))     # section/heading this chunk came from
    chunk: Mapped[str] = mapped_column(Text, nullable=False)   # the raw Markdown text (tables preserved)
    is_table_merged: Mapped[bool] = mapped_column(default=False)  # flagged if this chunk resulted from table-row merging

    # Dense retrieval
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim))

    # Sparse retrieval: Postgres-generated tsvector, kept in sync automatically by the DB.
    content_tsv = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk)", persisted=True),
    )

    extra_metadata: Mapped[dict | None] = mapped_column(JSONB)  # room for anything ad hoc (page no., etc.)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    report: Mapped["Report"] = relationship(back_populates="chunks")

    __table_args__ = (
        # Sparse lexical index (BM25-style via ts_rank over GIN)
        Index("ix_report_chunks_tsv", "content_tsv", postgresql_using="gin"),
        # Dense vector index (HNSW, cosine distance -- matches snowflake-arctic-embed's training objective)
        Index(
            "ix_report_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<ReportChunk id={self.id} report_id={self.report_id} idx={self.chunk_index}>"


class Feedback(Base):
    """Binary thumbs-up/down feedback captured in the Streamlit UI, mirrored to Langfuse."""
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[bool] = mapped_column(nullable=False)   # True = thumbs up, False = thumbs down
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} rating={self.rating}>"
