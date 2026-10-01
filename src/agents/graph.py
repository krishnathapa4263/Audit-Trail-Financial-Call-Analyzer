""" 
LangGraph Assembly — full pipeline: Planner -> Search -> Validator -> 
(replan retry loop) -> Writer -> Rerank & Output. 
CHANGES from the previous version of this file: 
  - Every node is now IMPORTED from its owning agent module rather than 
    redefined here. The old version of this file redefined writer_node and 
    rerank_node inline, which conflicted with (differently-behaving) copies 
    of the same functions living in writer.py -- whichever one actually got 
    imported by other code depended on import order, which is a nasty class 
    of bug. graph.py's only job now is wiring: build the StateGraph, add 
    nodes, add edges/conditional routing, compile. 
  - The previous version only wired "writer" and "rerank" -- planner, search, 
    validator, and the replan retry loop were never actually connected to 
    anything. This version wires the full pipeline. 
  - AgentState now imported from state.py (single canonical schema) instead 
    of being redefined inline here. 
""" 

from typing import Optional 
from langgraph.graph import StateGraph, END, START 
from config.settings import settings 
from src.agents.state import AgentState, initial_state 
from src.agents.planner import planner_node 
from src.agents.search import search_node 
from src.agents.validator import validator_node, replan_node 
from src.agents.writer import writer_node, NO_CONTEXT_FALLBACK 
from src.agents.reranker import rerank_node 
from src.observability.tracer import traced, flush, trace_context 


def route_validator(state: AgentState) -> str: 
    """After validator_node runs: 
      - validation_passed=True             -> writer (we have usable evidence) 
      - retries exhausted, still nothing    -> writer (writer_node will emit 
                                                the graceful fallback message 
                                                since validated_chunks is empty) 
      - otherwise                           -> replan (try again with 
                                                reformulated sub-queries) 
    """ 
    if state.get("validation_passed", False): 
        return "writer" 
    if state.get("retry_count", 0) >= state.get("max_retries", settings.max_validator_retries): 
        return "writer" 
    return "replan" 


def route_writer(state: AgentState) -> str: 
    """writer_node sets final_answer directly ONLY on the no-context fallback 
    path (empty validated_chunks) -- on the normal path it only sets 
    candidate_answers and leaves final_answer untouched. So checking for the 
    fallback sentinel here correctly distinguishes "nothing to rerank" from 
    "candidates ready, go rerank them" without an extra state field.""" 
    if state.get("final_answer") == NO_CONTEXT_FALLBACK: 
        return END 
    return "rerank" 


def build_graph() -> StateGraph: 
    """Builds and compiles the full ATFCA agent graph.""" 
    workflow = StateGraph(AgentState) 
    workflow.add_node("planner", traced(planner_node, "planner", as_type="agent")) 
    workflow.add_node("search", traced(search_node, "search", as_type="retriever")) 
    workflow.add_node("validator", traced(validator_node, "validator", as_type="evaluator")) 
    workflow.add_node("replan", traced(replan_node, "replan", as_type="agent")) 
    workflow.add_node("writer", traced(writer_node, "writer", as_type="chain")) 
    workflow.add_node("rerank", traced(rerank_node, "rerank", as_type="evaluator")) 
    workflow.add_edge(START, "planner") 
    workflow.add_edge("planner", "search") 
    workflow.add_edge("search", "validator") 
    workflow.add_conditional_edges( 
        "validator", 
        route_validator, 
        {"writer": "writer", "replan": "replan"}, 
    ) 
    workflow.add_edge("replan", "search")  # retry loop 
 
    workflow.add_conditional_edges( 
        "writer", 
        route_writer, 
        {"rerank": "rerank", END: END}, 
    ) 
    workflow.add_edge("rerank", END) 
    return workflow.compile() 
_compiled_graph = None 


def get_graph(): 
    global _compiled_graph 
    if _compiled_graph is None: 
        _compiled_graph = build_graph() 
    return _compiled_graph 


def run_query(query: str, max_retries: Optional[int] = None) -> AgentState: 
    """Convenience entrypoint: builds a fresh initial state and runs the 
    full graph end-to-end for a single question. 
    Wraps the whole run in ONE root trace_context so every node's span 
    nests under a single trace (rather than each node's @observe call 
    starting its own separate trace) -- this is what lets the UI capture 
    a trace_id/trace_url per query for feedback scoring and dashboard 
    deep-links. flush() runs in a finally block so buffered trace spans 
    are sent even if the graph raises -- per tracer.py's own design, this 
    never affects the query result: flush() is a no-op if tracing was 
    never enabled, and swallows its own failures if Langfuse is unreachable.""" 
    graph = get_graph() 
    state = initial_state(query, max_retries=max_retries) 
    try: 
        with trace_context("run_query", as_type="chain") as (trace_id, trace_url): 
            result = graph.invoke(state) 
            result["trace_id"] = trace_id 
            result["trace_url"] = trace_url 
            return result 
    finally: 
        flush() 

        
def stream_query( 
    query: str, 
    max_retries: Optional[int] = None, 
    company_name: Optional[str] = None, 
    report_type: Optional[str] = None, 
    date_from=None, 
    date_to=None, 
): 
    
    """Stage 14 addition: a streaming sibling to run_query(), for a UI that 
    wants to show live per-node progress (Planning... Searching... 
    Validating... Writing... Reranking...) instead of a single blocking 
    spinner. Yields (node_name, delta, accumulated_state) tuples as each 
    node finishes, using LangGraph's stream_mode="updates" (delta per node) 
    merged onto a running copy of state -- so callers get both "what just 
    happened" (node_name/delta, for status text) and "everything known so 
    far" (accumulated_state, for progressive rendering, e.g. showing 
    candidates as soon as the writer finishes without waiting for rerank). 
 
    company_name/report_type/date_from/date_to let a caller (e.g. the 
    sidebar's metadata filter toggles) pre-set filters that planner_node 
    will then only fill in where the caller left something blank -- see 
    planner_node's own docstring for that precedence rule. 
 
    The final yielded accumulated_state carries trace_id/trace_url, exactly 
    like run_query()'s return value. flush() still runs once, after the 
    generator is exhausted (in a finally block, so it runs even if the 
    caller breaks out of the loop early or the graph raises).""" 

    graph = get_graph() 
    state = initial_state( 
        query, 
        max_retries=max_retries, 
        company_name=company_name, 
        report_type=report_type, 
        date_from=date_from, 
        date_to=date_to, 
    ) 
    
    accumulated = dict(state) 
    try: 
        with trace_context("run_query", as_type="chain") as (trace_id, trace_url): 
            for update in graph.stream(state, stream_mode="updates"): 
                for node_name, delta in update.items(): 
                    accumulated.update(delta) 
                    yield node_name, delta, dict(accumulated) 
 
            accumulated["trace_id"] = trace_id 
            accumulated["trace_url"] = trace_url 
            yield "done", {}, dict(accumulated) 
    finally: 
        flush() 