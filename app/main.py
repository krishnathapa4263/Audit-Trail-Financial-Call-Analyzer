"""
Stage 14 — Streamlit entry point.

Per the architecture doc: "app/main.py, app/components/ -- Streamlit chat
interface, file upload sidebar, citation display, feedback widgets ->
Langfuse."

Run with:
    streamlit run app/main.py

This file's job is exactly what the doc says main.py should do: initialize
session state, wire the sidebar to the chat view, and nothing else -- all
actual rendering logic lives in app/components/.
"""

import os
import sys

# main.py lives at <project_root>/app/main.py. Running `streamlit run
# app/main.py` sets sys.path[0] to app/, not <project_root> -- without this,
# neither `from config.settings import settings` (used by components/) nor
# `from app.components... import ...` below would resolve.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.chat_view import render_chat_view

st.set_page_config(
    page_title="Audit Trail Financial Call Analyzer",
    layout="wide",
)

# --- Session state ---
if "messages" not in st.session_state:
    # Each entry: {"role": "user", "content": str}
    #          or {"role": "assistant", "query": str, "result": AgentState-like dict}
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}     # message index -> "up" | "down"
if "feedback_messages" not in st.session_state:
    st.session_state.feedback_messages = {}  # message index -> (streamlit_fn_name, text)

st.title("Audit Trail Financial Call Analyzer")

filters = render_sidebar()
render_chat_view(filters)
