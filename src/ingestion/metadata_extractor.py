"""
Metadata extraction (the "Extractor agent" role, per MimirRAG's architecture).

Reads only the first ~1024 tokens of a parsed document -- title/company/date/
report-type information is reliably in the front matter of filings and
transcripts, so we don't need to send the whole (often huge) document to the
LLM just to pull out six short fields.

Uses Groq's OpenAI-compatible JSON mode for structured, parseable output.
"""
import json
import logging

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

from config.settings import settings

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set in your .env -- required for metadata extraction.")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


class ReportMetadata(BaseModel):
    """Matches the Report table schema in src/database/models.py."""
    title: str = Field(description="Format: '[Report Type] - [Year] ([Company Name])'")
    company_name: str = Field(description="Standardized primary company name")
    keywords: list[str] = Field(default_factory=list, description="Up to five keywords: central financial topics, sectors, or strategic actions")
    summary: str = Field(description="One or two sentence summary of the document")
    date: str | None = Field(default=None, description="Fiscal period-end date or publication date, format YYYY-MM-DD if determinable, else null")
    report_type: str = Field(description="e.g. 10-K, 10-Q, Annual Report, Earnings Call Transcript, Press Release")


_SYSTEM_PROMPT = """You are a financial document metadata extractor. Given the first portion of a \
financial filing, earnings call transcript, or investor report, extract structured metadata.

Respond with ONLY a JSON object with exactly these fields:
- title: string, format "[Report Type] - [Year] ([Company Name])"
- company_name: string, the standardized primary company name
- keywords: array of up to 5 short strings capturing central financial topics/sectors/strategic actions
- summary: string, one or two sentences summarizing the document
- date: string in YYYY-MM-DD format if a fiscal period-end or publication date is determinable, otherwise null
- report_type: string, e.g. "10-K", "10-Q", "Annual Report", "Earnings Call Transcript", "Press Release"

If a field cannot be determined from the given text, make your best reasonable inference from context \
rather than leaving it empty, except for `date`, which should be null if genuinely not determinable.
Respond with the JSON object only, no other text."""


def build_prefix(markdown: str, max_chars: int | None = None) -> str:
    """Takes the leading portion of a document -- where title/company/date info lives."""
    max_chars = max_chars if max_chars is not None else settings.metadata_prefix_chars
    return markdown[:max_chars]


def extract_metadata(markdown: str, source_filename: str = "") -> ReportMetadata:
    """
    Extracts structured metadata from a document's prefix via an LLM call.
    Raises ValueError if the model's output can't be parsed/validated even
    after one retry -- callers should decide whether to skip the document
    or fall back to filename-based heuristics.
    """
    prefix = build_prefix(markdown)
    if not prefix.strip():
        raise ValueError(f"Empty document prefix for {source_filename!r} -- nothing to extract from.")

    client = _get_client()
    user_content = f"Source filename: {source_filename}\n\nDocument text:\n{prefix}"

    last_error: Exception | None = None
    for attempt in range(2):  # one retry on parse/validation failure
        response = client.chat.completions.create(
            model=settings.metadata_extraction_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content

        if raw_content is None:
            last_error = ValueError("Model returned no text content (content was None)")
            logger.warning(f"Attempt {attempt + 1} for {source_filename!r}: {last_error}")
            continue

        try:
            parsed = json.loads(raw_content)
            metadata = ReportMetadata.model_validate(parsed)
            logger.info(f"Extracted metadata for {source_filename!r}: "
                        f"company={metadata.company_name!r}, type={metadata.report_type!r}, date={metadata.date!r}")
            return metadata
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed to parse/validate metadata for "
                            f"{source_filename!r}: {e}. Raw output: {raw_content!r}")

    raise ValueError(f"Failed to extract valid metadata for {source_filename!r} after retries: {last_error}")
