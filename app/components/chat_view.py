""" 
Stage 14 — app/components/chat_view.py 
Per the architecture doc: "components/chat_view.py: Render logic for the 
main conversation UI, including streaming response text, citation drawers, 
source accordions, and feedback buttons (thumbs up/thumbs down)." 
Uses stream_query() (src/agents/graph.py) rather than run_query(), so the 
"streaming" part is real: each node's completion (planner, search, 
validator, replan-on-retry, writer, rerank) updates a live status line 
instead of a single blocking spinner. The final answer itself is not 
token-streamed (the underlying Groq calls in writer.py aren't set up for 
that), but the multi-agent pipeline's PROGRESS is genuinely live. 
""" 

from datetime import datetime 
 
import streamlit as st 
from src.agents.graph import stream_query 
from src.observability.tracer import score_trace 


NODE_STATUS_LABELS = { 
    "planner": "Decomposing the question...", 
    "search": "Searching filings...", 
    "validator": "Validating retrieved evidence...", 
    "replan": "Reformulating the query and retrying...", 
    "writer": "Drafting candidate answers...", 
    "rerank": "Scoring citations and selecting the best answer...", 
    "done": "Done.", 
} 


def _render_assistant_result(result: dict, msg_idx: int) -> None: 
    """Renders one assistant turn's full result: answer, trace link, metrics, 
    feedback widgets, and the three transparency expanders. Used both for a 
    just-completed turn and for replaying history on every rerun.""" 
    st.write(result.get("final_answer") or "_No answer produced._") 
    if result.get("trace_url"): 
        st.link_button("View full trace in Langfuse ↗", result["trace_url"]) 
    cols = st.columns(4) 
    cols[0].metric("Validated", "Yes" if result.get("validation_passed") else "No") 
    cols[1].metric("Retry rounds", result.get("retry_count", 0)) 
    cols[2].metric("Candidates", len(result.get("candidate_answers", []))) 
    best_f1 = result["citation_scores"][0]["citation_f1"] if result.get("citation_scores") else None 
    cols[3].metric("Best citation F1", f"{best_f1:.2f}" if best_f1 is not None else "—") 

    # --- Feedback widgets (keyed by message index, NOT trace_id -- multiple 
    # turns in one chat session can all have trace_id=None when Langfuse is 
    # disabled, which would collide on the same widget key otherwise) --- 

    trace_id = result.get("trace_id") 
    already_given = st.session_state.feedback_given.get(msg_idx) 
    fcol1, fcol2, fcol3 = st.columns([1, 1, 4]) 
    up_clicked = fcol1.button("  ", key=f"up_{msg_idx}", disabled=(already_given is not None), 
                               help="Helpful") 
    down_clicked = fcol2.button("  ", key=f"down_{msg_idx}", disabled=(already_given is not None), 
                                 help="Not helpful") 
    comment = fcol3.text_input("Optional comment", key=f"comment_{msg_idx}", 
                                 label_visibility="collapsed", placeholder="Optional comment...") 
    if up_clicked or down_clicked: 
        value = 1.0 if up_clicked else 0.0 
        sent = score_trace(trace_id, "user_feedback", value, comment=comment or None) 
        st.session_state.feedback_given[msg_idx] = "up" if up_clicked else "down" 
        
        # Store the message rather than showing it now: st.rerun() right 
        # after st.success()/st.info() would tear down this render before 
        # the message ever reaches the browser (see graph.py/tracer.py 
        # commit history for the same lesson learned on the earlier 
        # single-page version of this UI). 

        if sent: 
            st.session_state.feedback_messages[msg_idx] = ("success", "Thanks — feedback recorded on this run's trace.") 
        elif trace_id is None: 
            st.session_state.feedback_messages[msg_idx] = ("info", "Feedback noted, but tracing is disabled so it wasn't attached to a trace.") 
        else: 
            st.session_state.feedback_messages[msg_idx] = ("warning", "Feedback wasn't saved — couldn't reach Langfuse.") 
        st.rerun() 
    elif already_given: 
        st.caption(f"You marked this answer as {'   helpful' if already_given == 'up' else '   not helpful'}.") 
        pending = st.session_state.feedback_messages.get(msg_idx) 
        if pending: 
            kind, text = pending 
            getattr(st, kind)(text) 

    # --- Source evidence (citation drawer / source accordion) --- 

    with st.expander("    Source evidence"): 
        validated_chunks = result.get("validated_chunks", []) 
        if not validated_chunks: 
            st.write("No validated source chunks — this answer has no supporting evidence.") 
        else: 
            for c in validated_chunks: 
                st.markdown(f"**[{c.index}]** · {c.source_document}") 
                st.text(c.text) 
                st.divider() 

    # --- Query decomposition --- 

    with st.expander("    Query decomposition"): 
        st.write("**Sub-queries:**") 
        for sq in result.get("sub_queries", []): 
            st.write(f"- {sq}") 
        meta_cols = st.columns(3) 
        meta_cols[0].write(f"**Company:** {result.get('company_name') or '—'}") 
        meta_cols[1].write(f"**Report type:** {result.get('report_type') or '—'}") 
        date_from, date_to = result.get("date_from"), result.get("date_to") 
        date_range = f"{date_from.date()} to {date_to.date()}" if date_from and date_to else "—" 
        meta_cols[2].write(f"**Date range:** {date_range}") 

    # --- All candidates & citation scores --- 

    with st.expander("          All candidate answers & citation scores"): 
        candidates = result.get("candidate_answers", []) 
        scores_by_id = {s["candidate_id"]: s for s in result.get("citation_scores", [])} 
        if not candidates: 
            st.write("No candidates were generated for this query.") 
        else: 
            for cand in candidates: 
                score = scores_by_id.get(cand.candidate_id) 
                is_best = result.get("final_answer_candidate_id") == cand.candidate_id 
                st.markdown(f"**Candidate {cand.candidate_id}**" + ("  ⭐ selected" if is_best else "")) 
                if score: 
                    sc = st.columns(3) 
                    sc[0].write(f"Recall: {score['citation_recall']:.2f}") 
                    sc[1].write(f"Precision: {score['citation_precision']:.2f}") 
                    sc[2].write(f"F1: {score['citation_f1']:.2f}") 
                st.write(cand.raw_text) 
                st.divider() 


def render_chat_view(filters: dict) -> None: 
    # --- Replay history on every rerun (Streamlit re-executes top-to-bottom 
    # on every interaction, so past turns must be redrawn every time) --- 
    for idx, message in enumerate(st.session_state.messages): 
        with st.chat_message(message["role"]): 
            if message["role"] == "user": 
                st.write(message["content"]) 
            else: 
                _render_assistant_result(message["result"], idx) 

    # --- New query --- 
    
    query = st.chat_input("Ask a question about a company's financial filings or earnings calls") 
    if not query: 
        return 
    st.session_state.messages.append({"role": "user", "content": query}) 
    with st.chat_message("user"): 
        st.write(query) 
    with st.chat_message("assistant"): 
        status = st.status(NODE_STATUS_LABELS["planner"], expanded=False) 
        final_state = None 
        try: 
            for node_name, delta, accumulated in stream_query( 
                query, 
                max_retries=filters.get("max_retries"), 
                company_name=filters.get("company_name"), 
                report_type=filters.get("report_type"), 
                date_from=filters.get("date_from"), 
                date_to=filters.get("date_to"), 
            ): 
                status.update(label=NODE_STATUS_LABELS.get(node_name, node_name)) 
                final_state = accumulated 
            status.update(label="Done", state="complete") 
        except Exception as e: 
            status.update(label="Failed", state="error") 
            st.error(f"The pipeline raised an error: {e}") 
            with st.expander("Full error details"): 
                st.exception(e) 
            return 
        msg_idx = len(st.session_state.messages) 
        final_state["_ran_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        _render_assistant_result(final_state, msg_idx) 
        st.session_state.messages.append({"role": "assistant", "query": query, "result": final_state}) 