"""
Stage 13 -- Observability (Langfuse v4 / OpenTelemetry).

DESIGN PRINCIPLE: tracing is ADDITIVE, never load-bearing. If Langfuse
credentials are missing, malformed, or the Langfuse backend is unreachable,
the agent pipeline must still run exactly as it would without this file
existing at all. Every function here is built around that guarantee --
nothing in this module ever raises out to the caller.

Nodes get wrapped with @observe(...) at graph-BUILD time, inside graph.py's
build_graph(), rather than by editing planner.py/search.py/validator.py/
writer.py/reranker.py directly:

    workflow.add_node("planner", traced(planner_node, "planner", as_type="agent"))
    workflow.add_node("search", traced(search_node, "search", as_type="retriever"))
    workflow.add_node("validator", traced(validator_node, "validator", as_type="evaluator"))
    workflow.add_node("replan", traced(replan_node, "replan", as_type="agent"))
    workflow.add_node("writer", traced(writer_node, "writer", as_type="chain"))
    workflow.add_node("rerank", traced(rerank_node, "rerank", as_type="evaluator"))

This keeps Stage 13 a self-contained addition on top of already-tested
agent logic, consistent with graph.py's own stated principle that IT is
the only place wiring happens.

Uses the current Langfuse Python SDK v4 (OpenTelemetry-based, released
March 2026). The legacy v2/v3 client API (Langfuse().trace(), .span(),
.generation()) is deprecated, and Langfuse Cloud stops accepting traces via
the legacy ingestion endpoint on November 16, 2026 -- do not use it for new
instrumentation. Verified against the actually-installed SDK version
(langfuse 4.15.6), not assumed from documentation alone.
"""
import logging
import os
from contextlib import contextmanager
from typing import Callable, Literal, Optional, TypeVar

from config.settings import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

NodeType = Literal[
    "agent", "tool", "chain", "retriever", "evaluator", "guardrail",
    "span", "generation", "embedding",
]

_tracing_enabled = False
_checked = False


def _resolve_and_export_credentials() -> bool:
    """
    Resolves Langfuse credentials from settings, falling back to already-set
    environment variables -- the same defensive pattern already used
    throughout this codebase (see validator.py/reranker.py's _get_client():
    `getattr(settings, "groq_api_key", None) or os.environ.get("GROQ_API_KEY")`).

    Exports the resolved values into os.environ via setdefault() (never
    overwriting anything the user already set directly), because the
    Langfuse v4 client -- both here and internally inside every
    @observe-wrapped node -- reads its configuration from the environment
    on first use (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL).
    This guarantees every part of the app that ends up calling get_client()
    resolves to the SAME correctly-configured global client, regardless of
    which module happens to trigger that first call.
    """
    public_key = getattr(settings, "langfuse_public_key", None) or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = getattr(settings, "langfuse_secret_key", None) or os.environ.get("LANGFUSE_SECRET_KEY")
    base_url = (
        getattr(settings, "langfuse_base_url", None)
        or os.environ.get("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )

    if not public_key or not secret_key:
        return False

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", base_url)
    return True


def is_tracing_enabled() -> bool:
    """
    Resolves (once, cached) whether Langfuse credentials are available.
    Safe to call as often as needed -- only does real work on the first call.
    """
    global _tracing_enabled, _checked
    if _checked:
        return _tracing_enabled

    _tracing_enabled = _resolve_and_export_credentials()
    _checked = True

    if _tracing_enabled:
        logger.info("Langfuse observability enabled.")
    else:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not found -- observability disabled. "
            "The pipeline will run normally; nothing will be traced."
        )
    return _tracing_enabled


def traced(fn: F, name: str, as_type: Optional[NodeType] = None) -> F:
    """
    Wraps a LangGraph node function with Langfuse's @observe(...) IF
    credentials are available; otherwise returns fn completely unmodified.
    This is the single call site graph.py's build_graph() uses to add
    tracing to each node -- tracing can be added, removed, or reconfigured
    from this one file without touching any agent module.

    Also wrapped in a try/except: if applying @observe itself somehow fails
    (e.g. an incompatible SDK version), the node still runs -- untraced --
    rather than crashing graph construction entirely.
    """
    if not is_tracing_enabled():
        return fn

    try:
        from langfuse import observe
        return observe(name=name, as_type=as_type)(fn)
    except Exception as e:
        logger.warning(f"Failed to wrap {name!r} with Langfuse tracing (node will run untraced): {e}")
        return fn


def flush() -> None:
    """
    Flushes any buffered trace data to Langfuse. The v4 SDK is OpenTelemetry-
    based and batches/sends spans asynchronously in the background -- without
    an explicit flush, a short-lived process (a CLI run, a test, a serverless
    invocation) can exit before the last trace's spans are actually sent.

    Call this once, after run_query() completes. Safe to call even if
    tracing was never enabled (no-ops silently), and never raises -- a
    flush failure (e.g. Langfuse temporarily unreachable) is logged and
    swallowed, exactly like every other function in this module.
    """
    if not is_tracing_enabled():
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed (traces may be delayed/lost, pipeline unaffected): {e}")


@contextmanager
def trace_context(name: str, as_type: NodeType = "chain"):
    """
    Stage 14 addition: opens ONE root trace span for an entire pipeline run
    (wrap this around a single run_query() call), so every node's
    @observe-wrapped span nests under that one trace instead of each node
    accidentally starting its own separate trace. Yields (trace_id,
    trace_url) -- both None if tracing is disabled or if opening the trace
    fails for any reason. Never raises -- same additive-only guarantee as
    every other function in this module.

    This is what makes UI feedback widgets possible: a thumbs up/down click
    needs a trace_id to attach the score to (see score_trace below), and
    individual node spans don't expose one on their own from outside the
    pipeline.
    """
    if not is_tracing_enabled():
        yield None, None
        return

    try:
        from langfuse import get_client
        client = get_client()
        with client.start_as_current_observation(name=name, as_type=as_type):
            trace_id = client.get_current_trace_id()
            trace_url = None
            if trace_id:
                # get_trace_url() makes its OWN network call (to resolve the
                # project id) separate from trace-id generation, which is
                # local/offline. A transient failure fetching the URL should
                # not throw away an already-valid trace_id -- feedback
                # scoring only needs trace_id, the URL is a nice-to-have for
                # the UI's deep-link.
                try:
                    trace_url = client.get_trace_url(trace_id=trace_id)
                except Exception as url_err:
                    logger.warning(f"Got trace_id but failed to resolve trace_url "
                                    f"(feedback scoring still works, UI deep-link won't): {url_err}")
            yield trace_id, trace_url
    except Exception as e:
        logger.warning(f"Failed to open trace context {name!r} (pipeline continues untraced): {e}")
        yield None, None


def score_trace(
    trace_id: Optional[str],
    name: str,
    value,
    comment: Optional[str] = None,
) -> bool:
    """
    Stage 14 addition: attaches a score (e.g. user feedback from the UI) to
    a specific trace. `value` can be numeric (e.g. 1.0/0.0 for thumbs up/
    down) or a string categorical score -- Langfuse supports both.

    Returns True if the score was sent, False if tracing is disabled,
    trace_id is None/falsy, or anything failed. Never raises -- a failed
    feedback submission should show up as "couldn't save feedback" in the
    UI, not as a crash.
    """
    if not is_tracing_enabled() or not trace_id:
        return False
    try:
        from langfuse import get_client
        get_client().create_score(trace_id=trace_id, name=name, value=value, comment=comment)
        return True
    except Exception as e:
        logger.warning(f"Failed to record score {name!r} on trace {trace_id!r}: {e}")
        return False