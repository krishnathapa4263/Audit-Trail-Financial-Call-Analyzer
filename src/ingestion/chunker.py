"""
Chunks parsed Markdown documents into retrieval-sized units.

Design: rather than running a generic splitter over the whole document and
trying to detect/repair table damage afterward, we first SEGMENT the markdown
into alternating text-blocks and table-blocks (a "table-block" = a contiguous
run of Markdown pipe-table rows). Table-blocks are then treated as atomic --
the generic character-based splitter is only ever applied to text-blocks, so
a table row can never be cut mid-line or mid-cell.

If a single table-block itself exceeds table_merge_max_chars (very large
tables), it is split strictly at row boundaries -- never mid-row -- so a
retrieved chunk always contains only complete, well-formed rows.
"""
import logging
from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_index: int
    text: str
    is_table_merged: bool = False   # True = this chunk is a (whole or row-grouped) table block


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def _segment_markdown(markdown: str) -> list[tuple[str, str]]:
    """
    Splits the raw markdown into an ordered list of (kind, content) segments,
    where kind is 'text' or 'table'. Consecutive table-row lines are grouped
    into a single 'table' segment; everything else is grouped into 'text'
    segments. This guarantees a table segment's boundaries always align with
    row boundaries, never mid-row.
    """
    lines = markdown.splitlines(keepends=True)
    segments: list[tuple[str, str]] = []
    current_kind: str | None = None
    buffer: list[str] = []

    for line in lines:
        kind = "table" if _is_table_line(line) else "text"
        if current_kind is None:
            current_kind = kind
            buffer = [line]
        elif kind == current_kind:
            buffer.append(line)
        else:
            assert current_kind is not None  # always true here: set on the first loop iteration
            segments.append((current_kind, "".join(buffer)))
            current_kind = kind
            buffer = [line]

    if buffer:
        assert current_kind is not None  # always true here: buffer is only non-empty after current_kind is set
        segments.append((current_kind, "".join(buffer)))

    return segments


def _split_text_segment(text: str) -> list[str]:
    """Generic character-based split for a non-table text segment."""
    if not text.strip():
        return []
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_max_chars,
        chunk_overlap=settings.chunk_overlap_chars,
        tokenizer=list,  # character-based, not token-based
        paragraph_separator="\n\n",
    )
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    return [n.get_content() for n in nodes if n.get_content().strip()]


def _split_table_segment(table_text: str, max_chars: int) -> list[str]:
    """
    Splits an oversized table segment strictly at row boundaries -- rows are
    never cut mid-line. Rows are greedily packed until the next row would
    exceed max_chars.
    """
    rows = table_text.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""

    for row in rows:
        if current and len(current) + len(row) > max_chars:
            chunks.append(current)
            current = row
        else:
            current += row

    if current.strip():
        chunks.append(current)

    return chunks


def chunk_markdown(markdown: str) -> list[Chunk]:
    """
    Full chunking pipeline: segment into text/table blocks, split each with
    the appropriate strategy, and number the results sequentially. Table
    chunks are flagged is_table_merged=True (whether they're a whole small
    table or one row-aligned piece of a large one).
    """
    if not markdown or not markdown.strip():
        return []

    segments = _segment_markdown(markdown)
    logger.info(f"Segmented document into {len(segments)} block(s) "
                f"({sum(1 for k, _ in segments if k == 'table')} table block(s))")

    all_texts: list[tuple[str, bool]] = []  # (text, is_table)
    for kind, content in segments:
        if kind == "table":
            if len(content) <= settings.table_merge_max_chars:
                all_texts.append((content, True))
            else:
                logger.info(f"Table block ({len(content)} chars) exceeds "
                            f"table_merge_max_chars ({settings.table_merge_max_chars}) -- "
                            f"splitting at row boundaries")
                for piece in _split_table_segment(content, settings.table_merge_max_chars):
                    all_texts.append((piece, True))
        else:
            for piece in _split_text_segment(content):
                all_texts.append((piece, False))

    final_chunks = [
        Chunk(chunk_index=i, text=text, is_table_merged=is_table)
        for i, (text, is_table) in enumerate(all_texts)
        if text.strip()
    ]

    n_table_chunks = sum(1 for c in final_chunks if c.is_table_merged)
    logger.info(f"Final chunk count: {len(final_chunks)} ({n_table_chunks} table chunk(s))")

    return final_chunks
