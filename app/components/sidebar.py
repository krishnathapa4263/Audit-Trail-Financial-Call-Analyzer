"""
Stage 14 — app/components/sidebar.py

Per the architecture doc: "components/sidebar.py: UI sidebar handling file
uploads (PDFs), URL ingestion inputs, and metadata filter toggles."

Returns a plain dict of the currently-selected filters/settings for
chat_view.py to pass into stream_query(): {"max_retries", "company_name",
"report_type", "date_from", "date_to"}.

INGESTION HOOK -- READ THIS: the file-upload and URL-ingestion buttons below
call _ingest_pdf_file() / _ingest_url(), which try to import your actual
Stage 1-5 ingestion pipeline (Firecrawl scraping, Docling parsing, chunking,
embedding, writing into Postgres+pgvector). I don't have that module's real
location or function signatures -- they were never shared in this
conversation -- so both hooks currently try a guessed import path
(`src.ingestion.pipeline`) and, if it doesn't exist, show an honest
"not wired up yet" message rather than silently pretending to succeed. Point
IMPORT PATH / FUNCTION CALL below at your real ingestion entrypoint and
these become fully functional; the rest of the sidebar (upload widget, URL
input, spinners, success/error display) doesn't need to change.
"""


from datetime import datetime, time
from typing import Optional

import streamlit as st

from config.settings import settings

REPORT_TYPES = [
    "Any",
    "10-K",
    "10-Q",
    "Annual Report",
    "Earnings Call Transcript",
    "Press Release",
]


def _ingest_file(uploaded_file) -> tuple[bool, str]:
    """Ingest any uploaded file through the ingestion pipeline."""
    try:
        from src.ingestion.pipeline import ingest_file
    except ImportError:
        return False, (
            "Ingestion pipeline not found — ensure `src/ingestion/pipeline.py` exists."
        )
    try:
        result = ingest_file(uploaded_file)
        return True, f"Ingested {result['filename']} ({result['company_name']}) — {result['num_chunks']} chunks stored."
    except Exception as e:
        return False, f"Ingestion failed for {uploaded_file.name}: {e}"


def _ingest_url(url: str) -> tuple[bool, str]:
    """Scrape and ingest an online URL through Firecrawl."""
    try:
        from src.ingestion.pipeline import ingest_url
    except ImportError:
        return False, (
            "Ingestion pipeline not found — ensure `src/ingestion/pipeline.py` exists."
        )
    try:
        result = ingest_url(url)
        return True, f"Scraped and ingested {result['company_name']} from URL ({result['num_chunks']} chunks stored)."
    except Exception as e:
        return False, f"Ingestion failed for URL: {e}"


def render_sidebar() -> dict:
    with st.sidebar:
        st.caption(
            "Agentic RAG over SEC filings & earnings calls. Every answer is "
            "traceable back to the exact source excerpt that supports it."
        )

        # --- System status ---
        st.subheader("System status")
        groq_ok = bool(settings.groq_api_key)
        langfuse_ok = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        st.write(("🟢" if groq_ok else "🔴") + " Groq API key configured")
        st.write(("🟢" if langfuse_ok else "🟡") + " Langfuse tracing " + ("enabled" if langfuse_ok else "disabled"))
        if langfuse_ok:
            st.link_button("Open Langfuse dashboard", settings.langfuse_base_url, use_container_width=True)
        if not groq_ok:
            st.error("GROQ_API_KEY is not set — queries will fail until this is fixed in your .env.")

        st.divider()

        # --- Document ingestion ---
        st.subheader("Add documents")
        uploaded_files = st.file_uploader(
            "Upload financial documents",
            type=None,  # Accepts PDF, DOCX, XLSX, PPTX, CSV, TXT, MD, etc.
            accept_multiple_files=True,
            help="Parsed with Docling and chunked into the vector database.",
        )
        if uploaded_files and st.button("Ingest uploaded files", use_container_width=True):
            for f in uploaded_files:
                with st.spinner(f"Ingesting {f.name}..."):
                    ok, msg = _ingest_file(f)
                (st.success if ok else st.error)(msg)

        ingest_url_input = st.text_input(
            "Or scrape a filing/transcript URL",
            placeholder="https://www.sec.gov/...",
        )
        if ingest_url_input and st.button("Scrape & ingest URL", use_container_width=True):
            with st.spinner(f"Scraping {ingest_url_input}..."):
                ok, msg = _ingest_url(ingest_url_input)
            (st.success if ok else st.error)(msg)

        st.divider()

        # --- Metadata filter toggles ---
        st.subheader("Metadata filters")
        st.caption("Leave blank to let the Planner infer these from your question.")
        company_name = st.text_input("Company", value="", placeholder="e.g. Apple") or None
        report_type_choice = st.selectbox("Report type", REPORT_TYPES, index=0)
        report_type = None if report_type_choice == "Any" else report_type_choice

        use_date_range = st.checkbox("Set a date range")
        date_from: Optional[datetime] = None
        date_to: Optional[datetime] = None
        if use_date_range:
            d_from = st.date_input("From", value=None)
            d_to = st.date_input("To", value=None)
            if d_from:
                date_from = datetime.combine(d_from, time.min)
            if d_to:
                date_to = datetime.combine(d_to, time.min)

        st.divider()

        # --- Pipeline settings ---
        st.subheader("Settings")
        max_retries = st.number_input(
            "Max validator retries",
            min_value=0,
            max_value=10,
            value=settings.max_validator_retries,
            help="How many times the Validator can ask the Planner to reformulate "
                 "sub-queries before giving up and returning a no-context answer.",
        )

        # --- Session history ---
        past_queries = [
            m["content"] for m in st.session_state.get("messages", []) if m["role"] == "user"
        ]
        if past_queries:
            st.divider()
            st.subheader("This session")
            for q in reversed(past_queries[-10:]):
                st.caption(f"• {q if len(q) <= 60 else q[:57] + '...'}")

    return {
        "max_retries": int(max_retries),
        "company_name": company_name,
        "report_type": report_type,
        "date_from": date_from,
        "date_to": date_to,
    }
