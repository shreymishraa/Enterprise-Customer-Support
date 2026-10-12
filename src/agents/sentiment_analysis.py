"""
Sentiment Analysis Agent.

Evaluates customer satisfaction and urgency from their query.
This feeds into the escalation decision.
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.graph.state import AgentState

load_dotenv()

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

SYSTEM_PROMPT = """You are a customer sentiment analyser.
Analyse the customer's message and return a JSON object with:
  - "sentiment": one of [positive, neutral, negative, very_negative]
  - "urgency_score": float 0.0 to 1.0 (1.0 = extremely urgent)
  - "emotions": list of detected emotions (e.g. ["frustrated", "angry", "confused"])
  - "key_concern": a one-sentence summary of the customer's main concern

Return ONLY valid JSON. No extra text."""


def sentiment_analysis_node(state: AgentState) -> AgentState:
    """
    Node: Sentiment Analysis Agent.
    Classifies sentiment and urgency, augments escalation signal.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Customer message: {state['query']}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        sentiment = parsed.get("sentiment", "neutral")
        urgency_score = float(parsed.get("urgency_score", 0.0))
    except Exception:
        sentiment = "neutral"
        urgency_score = 0.0

    # Escalate immediately for very negative sentiment with high urgency
    auto_escalate = sentiment == "very_negative" and urgency_score >= 0.8

    return {
        **state,
        "sentiment": sentiment,
        "urgency_score": urgency_score,
        "needs_escalation": state.get("needs_escalation", False) or auto_escalate,
        "agents_invoked": ["sentiment_analysis"],
    }


# ── Validation node ────────────────────────────────────────────────────────────

VALIDATOR_SYSTEM_PROMPT = """You are a quality validator for customer support answers.
Given a user query and a generated answer, decide:
  - "is_valid": true if the answer directly addresses the query using the context
  - "reason": brief explanation
  - "improved_answer": optional — only provide if the answer can be clearly improved; otherwise leave null

Return ONLY valid JSON."""


def validator_node(state: AgentState) -> AgentState:
    """
    Node: Validator Agent.
    Checks answer quality and grounding before sending to the customer.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    context_summary = "\n".join(state.get("retrieved_context", [])[:3])

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Query: {state['query']}\n\n"
                f"Retrieved Context:\n{context_summary}\n\n"
                f"Generated Answer: {state.get('answer', '')}"
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        is_valid = bool(parsed.get("is_valid", True))
        improved = parsed.get("improved_answer")
    except Exception:
        is_valid = True
        improved = None

    final_answer = improved if (improved and not is_valid) else state.get("answer", "")

    return {
        **state,
        "answer": final_answer,
        "validation_passed": is_valid,
        "agents_invoked": ["validator"],
    }
