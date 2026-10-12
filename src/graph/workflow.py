"""
LangGraph multi-agent workflow for Enterprise Customer Support Automation.

Graph topology:
  START
    └─► query_understanding
          └─► rag_retrieval
                └─► sentiment_analysis
                      └─► validator
                            ├─► escalation  (if escalation needed)
                            └─► END
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END, START

from src.graph.state import AgentState
from src.agents.query_understanding import query_understanding_node
from src.agents.rag_retrieval import rag_retrieval_node
from src.agents.sentiment_analysis import sentiment_analysis_node, validator_node
from src.agents.escalation import escalation_node


# ── Routing functions ─────────────────────────────────────────────────────────

def should_escalate(state: AgentState) -> Literal["escalation", "end"]:
    """Route to escalation node if needed, otherwise END."""
    if (
        state.get("needs_escalation", False)
        or state.get("urgency_score", 0.0) >= 0.7
        or state.get("confidence", 1.0) < 0.4
        or not state.get("validation_passed", True)
    ):
        return "escalation"
    return "end"


# ── Graph construction ────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("sentiment_analysis", sentiment_analysis_node)
    graph.add_node("validator", validator_node)
    graph.add_node("escalation", escalation_node)

    # Add edges (linear backbone)
    graph.add_edge(START, "query_understanding")
    graph.add_edge("query_understanding", "rag_retrieval")
    graph.add_edge("rag_retrieval", "sentiment_analysis")
    graph.add_edge("sentiment_analysis", "validator")

    # Conditional branching after validation
    graph.add_conditional_edges(
        "validator",
        should_escalate,
        {
            "escalation": "escalation",
            "end": END,
        },
    )
    graph.add_edge("escalation", END)

    return graph.compile()


# ── Default state factory ─────────────────────────────────────────────────────

def make_initial_state(query: str, session_id: str = "default") -> AgentState:
    """Create a fresh AgentState for a new query."""
    return AgentState(
        query=query,
        session_id=session_id,
        intent=None,
        sentiment=None,
        urgency_score=0.0,
        retrieved_context=[],
        sources=[],
        answer=None,
        confidence=0.0,
        needs_escalation=False,
        escalation_reason=None,
        is_relevant=False,
        retrieval_method="none",
        agents_invoked=[],
        retry_count=0,
        validation_passed=False,
        feedback=None,
        messages=[],
        error=None,
    )


# ── Public interface ──────────────────────────────────────────────────────────

# Compile once at import time
_workflow = None


def get_workflow():
    """Lazily build and cache the compiled workflow."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run_query(query: str, session_id: str = "default") -> AgentState:
    """
    Run the full multi-agent pipeline for a single query.

    Args:
        query: User's natural language question.
        session_id: Session/conversation identifier.

    Returns:
        Final AgentState with all fields populated.
    """
    workflow = get_workflow()
    initial_state = make_initial_state(query, session_id)
    result = workflow.invoke(initial_state)
    return result
