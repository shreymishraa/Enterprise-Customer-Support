"""
Streamlit Chat UI — Enterprise Customer Support Bot.

WhatsApp-style chat interface that calls the FastAPI backend.
Run: streamlit run ui/main.py
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SupportBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }

    /* User message bubble */
    .user-bubble {
        background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
        color: white;
        border-radius: 20px 20px 4px 20px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-left: 20%;
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
        animation: slideInRight 0.3s ease-out;
    }

    /* Bot message bubble */
    .bot-bubble {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        color: #e2e8f0;
        border-radius: 20px 20px 20px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-right: 15%;
        border: 1px solid rgba(255,255,255,0.1);
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: slideInLeft 0.3s ease-out;
    }

    /* Escalation bubble */
    .escalation-bubble {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
        border-radius: 20px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-right: 15%;
        animation: slideInLeft 0.3s ease-out;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: rgba(108, 99, 255, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }

    /* Meta info */
    .meta-info {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        margin-top: 6px;
    }

    /* Agent badge */
    .agent-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 1px 7px;
        font-size: 0.68rem;
        margin: 1px;
    }

    /* Confidence bar */
    .confidence-bar-wrapper {
        margin-top: 6px;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.5);
    }

    /* Sentiment indicator */
    .sentiment-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 6px;
    }

    /* Animations */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* Input area */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 12px 16px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.6) !important;
    }

    /* Header */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.5);
        font-size: 0.95rem;
        margin-top: 6px;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* Scrollable chat */
    .scrollable-chat {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_url = st.text_input("API URL", value=API_BASE)
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.metric("Queries Sent", user_msgs)

    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("""
    1. 🧠 **Query Understanding**
    2. 🔍 **RAG Retrieval** (Hybrid)
    3. 💬 **Sentiment Analysis**
    4. ✅ **Validator**
    5. 🚨 **Escalation** *(if needed)*
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        st.rerun()

    st.markdown("---")
    # Health check
    try:
        health = requests.get(f"{api_url}/health", timeout=3).json()
        idx_status = "✅ Loaded" if health.get("faiss_index_loaded") else "❌ Not found"
        st.success(f"API: {health.get('status', 'unknown').upper()}")
        st.caption(f"FAISS Index: {idx_status}")
    except Exception:
        st.error("API: OFFLINE")
        st.caption("Start the FastAPI server first.")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">🤖 SupportBot AI</div>
        <div class="hero-subtitle">Enterprise Customer Support · Powered by Multi-Agent RAG</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


def sentiment_color(sentiment: str) -> str:
    colors = {
        "positive": "#10b981",
        "neutral": "#6b7280",
        "negative": "#f59e0b",
        "very_negative": "#ef4444",
    }
    return colors.get(sentiment, "#6b7280")


def render_message(msg: dict, idx: int):
    """Render a single chat message."""
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">💬 {msg["content"]}'
            f'<div class="meta-info">{msg.get("time", "")}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        resp = msg.get("response", {})
        needs_escalation = resp.get("needs_escalation", False)
        bubble_class = "escalation-bubble" if needs_escalation else "bot-bubble"

        # Sources
        sources_html = "".join(
            f'<span class="source-chip">📄 {s}</span>' for s in resp.get("sources", [])
        )

        # Agents
        agents_html = "".join(
            f'<span class="agent-badge">{a}</span>' for a in resp.get("agents_invoked", [])
        )

        # Sentiment
        sentiment = resp.get("sentiment", "neutral") or "neutral"
        sent_color = sentiment_color(sentiment)
        sent_html = f'<span class="sentiment-badge" style="background: {sent_color}22; color: {sent_color}; border: 1px solid {sent_color}44;">{sentiment}</span>'

        # Confidence
        conf = resp.get("confidence", 0.5)
        conf_pct = int(conf * 100)
        conf_color = "#10b981" if conf >= 0.7 else ("#f59e0b" if conf >= 0.4 else "#ef4444")

        answer_text = msg["content"].replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="{bubble_class}">
                <b>{'🚨 Escalated to Human Support' if needs_escalation else '🤖 SupportBot'}</b>{sent_html}
                <br><br>
                {answer_text}
                <br><br>
                {f'<b>Sources:</b><br>{sources_html}' if sources_html else ''}
                <div class="confidence-bar-wrapper">
                    Confidence: <span style="color:{conf_color}; font-weight:600;">{conf_pct}%</span>
                    &nbsp;|&nbsp; Retrieval: <b>{resp.get('retrieval_method', 'hybrid')}</b>
                    &nbsp;|&nbsp; Intent: <b>{resp.get('intent', '—')}</b>
                </div>
                <div class="meta-info">Agents: {agents_html} &nbsp;|&nbsp; {msg.get('time', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feedback buttons
        if not st.session_state.feedback_given.get(idx):
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                if st.button("👍", key=f"up_{idx}"):
                    _submit_feedback(api_url, idx, "thumbs_up", msg)
            with col2:
                if st.button("👎", key=f"down_{idx}"):
                    _submit_feedback(api_url, idx, "thumbs_down", msg)
        else:
            st.caption(f"✅ Feedback: {st.session_state.feedback_given[idx]}")

        # Expandable context
        if resp.get("retrieved_context"):
            with st.expander("🔍 View Retrieved Context"):
                for i, ctx in enumerate(resp["retrieved_context"][:3], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.markdown(f"> {ctx[:400]}{'...' if len(ctx) > 400 else ''}")
                    st.divider()


def _submit_feedback(api_url: str, idx: int, rating: str, msg: dict):
    """Submit feedback to the API."""
    try:
        resp = msg.get("response", {})
        requests.post(
            f"{api_url}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "query": resp.get("query", ""),
                "answer": msg["content"],
                "rating": rating,
            },
            timeout=5,
        )
        st.session_state.feedback_given[idx] = rating
        st.rerun()
    except Exception:
        pass


# Render all messages
for i, msg in enumerate(st.session_state.messages):
    render_message(msg, i)

# ── Suggested questions ───────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    suggestions = [
        "How do I reset my password?",
        "What is your refund policy?",
        "How do I cancel my subscription?",
        "My package was lost, what should I do?",
        "How do I enable two-factor authentication?",
    ]
    cols = st.columns(len(suggestions))
    for col, sugg in zip(cols, suggestions):
        with col:
            if st.button(sugg, key=f"sugg_{sugg[:10]}"):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": sugg,
                        "time": datetime.now().strftime("%H:%M"),
                    }
                )
                st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────

st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            label="Message",
            placeholder="Type your support question here…",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤")

if submitted and user_input.strip():
    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "time": datetime.now().strftime("%H:%M"),
        }
    )

    # Call API
    with st.spinner("🤖 Agents are thinking…"):
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={
                    "session_id": st.session_state.session_id,
                    "query": user_input,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            data["query"] = user_input

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "response": data,
                    "time": datetime.now().strftime("%H:%M"),
                }
            )
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "⚠️ Cannot connect to the API. Please start the FastAPI server with `uvicorn api.main:app --reload`.",
                    "response": {},
                    "time": datetime.now().strftime("%H:%M"),
                }
            )
        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "response": {},
                    "time": datetime.now().strftime("%H:%M"),
                }
            )

    st.rerun()
