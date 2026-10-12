"""
RAG Retrieval Agent.

Calls the RAG tool to fetch relevant context and generate an initial answer.
Implements retry logic with query rewriting if the first retrieval is not relevant.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.graph.state import AgentState
from src.tools.rag_tool import rag_tool

load_dotenv()

MAX_RETRIES = 2
LARGER_K = int(os.getenv("TOP_K", 5)) + 3


def rag_retrieval_node(state: AgentState) -> AgentState:
    """
    Node: RAG Retrieval Agent.

    1. Calls rag_tool with the current query.
    2. If retrieval is not relevant and retries remain, rewrites the query and retries.
    3. Stores results back into state.
    """
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    # First attempt
    result = rag_tool(query=query, rewrite=(retry_count > 0))

    # Retry with rewriting if not relevant and retries remain
    while not result["is_relevant"] and retry_count < MAX_RETRIES:
        retry_count += 1
        result = rag_tool(
            query=query,
            k=LARGER_K,
            rewrite=True,
        )

    return {
        **state,
        "retrieved_context": result["retrieved_context"],
        "sources": result["sources"],
        "answer": result["answer"],
        "confidence": result["confidence"],
        "needs_escalation": result["needs_escalation"],
        "is_relevant": result["is_relevant"],
        "retrieval_method": result["retrieval_method"],
        "retry_count": retry_count,
        "agents_invoked": ["rag_retrieval"],
    }
