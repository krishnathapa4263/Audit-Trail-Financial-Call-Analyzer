"""
Wraps the Firecrawl SDK to scrape online SEC reports, financial earnings call
transcripts, or investor relations pages into clean Markdown.

Uses the current firecrawl-py (v4.x) API: FirecrawlApp(...).scrape(url, ...)
returning a Document with a `.markdown` attribute. (Older docs/tutorials
reference a `scrape_url()` method from the pre-v1 SDK -- that still exists on
this client for backwards compatibility, but `.scrape()` is the current,
actively maintained entrypoint, so that's what we use here.)
"""
import logging
from dataclasses import dataclass

from firecrawl import FirecrawlApp

from config.settings import settings

logger = logging.getLogger(__name__)

_app: FirecrawlApp | None = None


def _get_client() -> FirecrawlApp:
    """Lazily creates the Firecrawl client so importing this module never
    requires an API key to be set (useful for testing other parts in isolation)."""
    global _app
    if _app is None:
        if not settings.firecrawl_api_key:
            raise RuntimeError(
                "FIRECRAWL_API_KEY is not set in your .env -- required to scrape URLs."
            )
        _app = FirecrawlApp(api_key=settings.firecrawl_api_key)
    return _app


@dataclass
class ScrapedPage:
    source_url: str
    markdown: str
    title: str | None
    metadata: dict


def scrape_url(url: str, only_main_content: bool = True) -> ScrapedPage:
    """
    Scrape a single URL and return it as clean Markdown.

    only_main_content=True strips navs/footers/ads (Firecrawl's own content
    extraction) -- generally what we want for a financial filing or IR page,
    since boilerplate site chrome only adds noise to the retrieval corpus.
    """
    client = _get_client()
    logger.info(f"Scraping {url} via Firecrawl...")

    try:
        doc = client.scrape(url, formats=["markdown"], only_main_content=only_main_content)
    except Exception as e:
        logger.error(f"Firecrawl failed to scrape {url}: {e}")
        raise

    if not doc.markdown:
        logger.warning(f"Firecrawl returned no markdown content for {url}")

    metadata = doc.metadata if isinstance(doc.metadata, dict) else (doc.metadata.model_dump() if doc.metadata else {})
    title = metadata.get("title")

    logger.info(f"Scraped {url}: {len(doc.markdown or '')} chars of markdown")

    return ScrapedPage(
        source_url=url,
        markdown=doc.markdown or "",
        title=title,
        metadata=metadata,
    )
