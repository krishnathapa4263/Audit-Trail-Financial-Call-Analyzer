"""
Parses local PDF files (user-uploaded filings, or PDFs saved from EDGAR)
into layout-aware Markdown using Docling.

Docling performs layout analysis (via DocLayNet) and table structure recognition
(via TableFormer) so that financial tables survive as proper Markdown pipe-tables
instead of being flattened into unreadable text -- this is the whole point of
using Docling over a plain PDF-to-text extractor for this project.
"""
import logging
from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

_converter = DocumentConverter()  # created once; reused across calls (loads models on first use)


@dataclass
class ParsedDocument:
    source_path: str
    markdown: str
    num_pages: int
    num_tables: int


def parse_document(file_path: str | Path) -> ParsedDocument:
    """
    Convert a single document into structured Markdown.

    Raises FileNotFoundError if the path doesn't exist, and re-raises any
    Docling conversion error after logging it (a corrupt/scanned/unreadable
    PDF should fail loudly here, not silently produce empty output).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    logger.info(f"Parsing {file_path.name} with Docling...")
    suffix = file_path.suffix.lower()

    # Direct read for plain text or markdown files
    if suffix in [".txt", ".md"]:
        text = file_path.read_text(encoding="utf-8")
        return ParsedDocument(
            source_path=str(file_path),
            markdown=text,
            num_pages=1,
            num_tables=0,
        )

    # Use Docling for PDFs, Office documents, HTML, and Images
    try:
        result = _converter.convert(str(file_path))
    except Exception as e:
        logger.error(f"Docling failed to convert {file_path.name}: {e}")
        raise

    doc = result.document
    markdown = doc.export_to_markdown()
    num_tables = len(doc.tables) if hasattr(doc, "tables") else 0
    num_pages = len(doc.pages) if hasattr(doc, "pages") else 1

    logger.info(f"Parsed {file_path.name}: {num_pages} page(s), {num_tables} table(s) detected")

    return ParsedDocument(
        source_path=str(file_path),
        markdown=markdown,
        num_pages=num_pages,
        num_tables=num_tables,
    )
