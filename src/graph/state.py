"""
LangGraph state definition for the multi-agent customer support workflow.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph workflow.

    Fields:
        query:              Original user question.
        session_id:         Conversation session identifier.
        intent:             Detected intent/category (e.g. billing, account, technical).
        sentiment:          Detected sentiment (positive, neutral, negative, urgent).
        urgency_score:      0.0–1.0 urgency level from sentiment agent.
        retrieved_context:  List of retrieved text chunks.
        sources:            List of source document references.
        answer:             Final generated answer.
        confidence:         Generation confidence score (0.0–1.0).
        needs_escalation:   Whether to escalate to human support.
        escalation_reason:  Reason for escalation (if any).
        is_relevant:        Whether retrieval was relevant to the query.
        retrieval_method:   hybrid | semantic | keyword | none.
        agents_invoked:     Log of agent names called during this run.
        retry_count:        Number of RAG retries performed.
        validation_passed:  Whether the validator approved the response.
        feedback:           Optional user feedback (thumbs up/down).
        messages:           Conversation history (list of {role, content} dicts).
        error:              Any error message encountered.
    """
    query: str
    session_id: str
    intent: Optional[str]
    sentiment: Optional[str]
    urgency_score: float
    retrieved_context: List[str]
    sources: List[str]
    answer: Optional[str]
    confidence: float
    needs_escalation: bool
    escalation_reason: Optional[str]
    is_relevant: bool
    retrieval_method: str
    agents_invoked: Annotated[List[str], operator.add]
    retry_count: int
    validation_passed: bool
    feedback: Optional[str]
    messages: List[Dict[str, Any]]
    error: Optional[str]
