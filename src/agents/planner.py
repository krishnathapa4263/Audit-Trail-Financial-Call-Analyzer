"""
Planner Agent: decomposes user queries into targeted sub-queries and extracts
metadata constraints (company, report type, date range), per MimirRAG's
Planner Agent design.

Complex/comparative queries (e.g. "Compare Apple's margin between 2023 and
2024") get broken into targeted sub-queries, one per aspect/period. Simple
queries pass through as a single sub-query. This runs BEFORE any retrieval --
the Planner only ever sees the raw question, never retrieved content.
"""
import json
import logging
from datetime import datetime

from groq import Groq
from httpx2 import query
from pydantic import BaseModel, Field, ValidationError

from config.settings import settings
from src.agents import state
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set in your .env -- required for query planning.")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


class PlannerOutput(BaseModel):
    sub_queries: list[str] = Field(description="1 or more targeted, self-contained sub-questions")
    company_name: str | None = Field(default=None, description="Company name mentioned in the query, if any")
    report_type: str | None = Field(
        default=None,
        description="Report type mentioned (10-K, 10-Q, Annual Report, Earnings Call Transcript), if any",
    )
    date_from: str | None = Field(default=None, description="Start of the relevant date range, YYYY-MM-DD, if determinable")
    date_to: str | None = Field(default=None, description="End of the relevant date range, YYYY-MM-DD, if determinable")


_SYSTEM_PROMPT = """You are a financial query planning agent. Given a user's natural-language question \
about financial filings or earnings calls, your job is to:

1. Decide whether the question is simple (answerable with one focused search) or complex/comparative \
(requires breaking into multiple targeted sub-questions -- e.g. comparing two time periods, or asking \
about multiple distinct metrics at once).
2. If complex, decompose it into a list of simple, self-contained sub-questions, each answerable by a \
single focused search. If simple, return the original question as the only sub-query (rephrase to be \
self-contained if it relies on conversational context).
3. Extract any metadata constraints mentioned or implied: company name, report type (10-K, 10-Q, Annual \
Report, Earnings Call Transcript, Press Release), and date range (as YYYY-MM-DD to YYYY-MM-DD, covering \
the full relevant fiscal period(s) mentioned -- e.g. "in 2023" becomes 2023-01-01 to 2023-12-31).

Example:
Question: "How did Novo Nordisk's profit change between 2022 and 2023?"
Response: {"sub_queries": ["What was Novo Nordisk's profit in 2022?", "What was Novo Nordisk's profit in 2023?"], \
"company_name": "Novo Nordisk", "report_type": null, "date_from": "2022-01-01", "date_to": "2023-12-31"}

Example:
Question: "What was Apple's diluted earnings per share last quarter?"
Response: {"sub_queries": ["What was Apple's diluted earnings per share last quarter?"], \
"company_name": "Apple", "report_type": null, "date_from": null, "date_to": null}

Respond with ONLY a JSON object with exactly these fields: sub_queries (array of strings, at least 1), \
company_name (string or null), report_type (string or null), date_from (string YYYY-MM-DD or null), \
date_to (string YYYY-MM-DD or null). No other text."""


def plan(query: str) -> PlannerOutput:
    """
    Decomposes a query and extracts metadata constraints via an LLM call.
    Raises ValueError if the model's output can't be parsed/validated even
    after one retry, or if a valid response somehow has an empty sub_queries
    list (which would leave nothing for the Search agent to do).
    """
    if not query.strip():
        raise ValueError("Empty query -- nothing to plan.")

    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(2):
        response = client.chat.completions.create(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content

        if raw_content is None:
            last_error = ValueError("Model returned no text content (content was None)")
            logger.warning(f"Attempt {attempt + 1} for {query!r}: {last_error}")
            continue

        try:
            parsed = json.loads(raw_content)
            result = PlannerOutput.model_validate(parsed)
            if not result.sub_queries:
                raise ValueError("Model returned an empty sub_queries list")
            logger.info(f"Planned {query!r} -> {len(result.sub_queries)} sub-quer(y/ies), "
                        f"company={result.company_name!r}, dates=[{result.date_from}, {result.date_to}]")
            return result
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed to parse/validate planner output for "
                            f"{query!r}: {e}. Raw output: {raw_content!r}")

    raise ValueError(f"Failed to plan query {query!r} after retries: {last_error}")


def _parse_date(date_str: str | None) -> datetime | None:
    """Planner returns dates as 'YYYY-MM-DD' strings (or null); search()
    downstream expects datetime objects. A malformed date from the model
    (rare, but LLMs do occasionally emit e.g. '2023' or 'Q1 2023') should
    degrade to "no date filter" rather than crashing the whole pipeline."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.warning(f"Planner returned an unparseable date {date_str!r} -- ignoring it "
                        f"(no date filter will be applied for this constraint)")
        return None


def planner_node(state: AgentState) -> dict:
    """LangGraph node wrapper for the planner agent. Entry point of the graph.

    Precedence rule (Stage 14 addition): if the caller pre-set company_name/
    report_type/date_from/date_to on the incoming state (e.g. via a UI
    sidebar's manual filter toggles, through initial_state()'s matching
    parameters), that value wins over the Planner's own extraction for that
    field. The Planner still fills in anything the caller left blank. This
    is a plain 'truthy value already present wins' check, not a separate
    "was this explicitly set" flag -- initial_state() defaults all four to
    None, so there's no ambiguity between "not set" and "set to empty"."""
    query = state.get("query", "")
    try:
        result = plan(query)
    except (ValueError, RuntimeError) as e:
        logger.error(f"Planner failed for {query!r}: {e}. Falling back to raw query.")
        return {
            "sub_queries": [query] if query.strip() else [],
            "company_name": state.get("company_name"),
            "report_type": state.get("report_type"),
            "date_from": state.get("date_from"),
            "date_to": state.get("date_to"),
            "error": f"planner_node: {e}",
        }

    # Precedence: Explicit company mentioned in query > UI sidebar filter > None
    company_name = result.company_name or state.get("company_name")
    report_type = result.report_type or state.get("report_type")

    return {
        "sub_queries": result.sub_queries,
        "company_name": company_name,
        "report_type": report_type,
        "date_from": state.get("date_from") or _parse_date(result.date_from),
        "date_to": state.get("date_to") or _parse_date(result.date_to),
    }
