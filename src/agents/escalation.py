"""
Escalation Agent.

Detects unresolved issues and routes them to human support.
Also invokes the RAG tool as a sub-tool (Agents as Tools pattern) to
check if there is any escalation policy information in the knowledge base.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.graph.state import AgentState
from src.tools.rag_tool import rag_tool

load_dotenv()

# Urgency threshold above which we escalate regardless of other signals
URGENCY_THRESHOLD = float(os.getenv("URGENCY_THRESHOLD", "0.7"))
# Confidence threshold below which we escalate (uncertain answers)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

ESCALATION_MESSAGE = (
    "I'm sorry, but your issue requires personal attention from our support team. "
    "A human agent will contact you within 2 business hours. "
    "Your reference number is: ESC-{session_id}. "
    "You can also reach us at support@company.com or call +1-800-SUPPORT."
)


def escalation_node(state: AgentState) -> AgentState:
    """
    Node: Escalation Agent.

    Decides whether to escalate based on:
    1. needs_escalation flag from RAG tool
    2. Urgency score from sentiment analysis
    3. Low confidence in generated answer
    4. Validation failure after retries

    If escalating, fetches escalation policy via RAG (Agents as Tools pattern)
    and composes a human-handoff message.
    """
    should_escalate = (
        state.get("needs_escalation", False)
        or state.get("urgency_score", 0.0) >= URGENCY_THRESHOLD
        or state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
        or (not state.get("validation_passed", True) and state.get("retry_count", 0) >= 2)
    )

    escalation_reason = None
    final_answer = state.get("answer", "")

    if should_escalate:
        # Agents as Tools: invoke RAG to fetch escalation procedures from KB
        rag_result = rag_tool(query="escalation procedure human agent support contact", k=2)
        escalation_policy_context = rag_result.get("retrieved_context", [])

        reasons = []
        if state.get("needs_escalation"):
            reasons.append("issue flagged for human review")
        if state.get("urgency_score", 0.0) >= URGENCY_THRESHOLD:
            reasons.append(f"high urgency ({state['urgency_score']:.2f})")
        if state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            reasons.append(f"low confidence ({state.get('confidence', 0):.2f})")
        if not state.get("validation_passed", True):
            reasons.append("answer validation failed")

        escalation_reason = "; ".join(reasons)
        final_answer = ESCALATION_MESSAGE.format(
            session_id=state.get("session_id", "N/A")
        )

    return {
        **state,
        "answer": final_answer,
        "needs_escalation": should_escalate,
        "escalation_reason": escalation_reason,
        "agents_invoked": ["escalation"],
    }
