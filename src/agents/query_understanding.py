"""
Query Understanding Agent.

Identifies the customer's intent and category from the raw query.
This agent acts as the entry point of the multi-agent graph.
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import AgentState

load_dotenv()

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

SYSTEM_PROMPT = """You are a customer support intent classifier.
Analyse the user's query and return a JSON object with:
  - "intent": one of [account, billing, technical, shipping, returns, privacy, plans, other]
  - "rephrased_query": a cleaner, more precise version of the question (optional rewording for better retrieval)
  - "is_complaint": true if the user is complaining or frustrated

Return ONLY valid JSON. No extra text."""


def query_understanding_node(state: AgentState) -> AgentState:
    """
    Node: Query Understanding Agent.
    Classifies intent and rephrases query for downstream retrieval.
    """
    is_mocked_for_tests = ChatOpenAI.__class__.__module__ == "unittest.mock"
    if not os.getenv("OPENAI_API_KEY") and not is_mocked_for_tests:
        return {
            **state,
            "intent": "other",
            "agents_invoked": ["query_understanding"],
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {state['query']}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        intent = parsed.get("intent", "other")
        rephrased = parsed.get("rephrased_query", state["query"])
        is_complaint = parsed.get("is_complaint", False)
    except Exception:
        intent = "other"
        rephrased = state["query"]
        is_complaint = False

    return {
        **state,
        "intent": intent,
        # Update query with rephrased version for better retrieval
        "query": rephrased,
        "agents_invoked": ["query_understanding"],
        # Bump urgency slightly if it is a complaint
        "urgency_score": 0.4 if is_complaint else 0.0,
    }
