"""Pydantic schemas for the FastAPI backend."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request payload."""
    session_id: str = Field(default="default", description="Session or conversation ID")
    query: str = Field(..., min_length=1, max_length=2000, description="User's support question")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123",
                    "query": "What is the penalty for early loan foreclosure?",
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response payload from the RAG pipeline."""
    session_id: str
    answer: str
    sources: List[str]
    retrieved_context: List[str]
    agents_invoked: List[str]
    confidence: float
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    urgency_score: float = 0.0
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None
    retrieval_method: str = "hybrid"
    retry_count: int = 0
    validation_passed: bool = True

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123",
                    "answer": "Foreclosure charges depend on the loan type...",
                    "sources": ["loan_policy.pdf#p4", "faq_loans.md"],
                    "retrieved_context": ["...policy text..."],
                    "agents_invoked": ["query_understanding", "rag_retrieval", "sentiment_analysis", "validator"],
                    "confidence": 0.87,
                    "intent": "billing",
                    "sentiment": "neutral",
                    "urgency_score": 0.1,
                    "needs_escalation": False,
                    "escalation_reason": None,
                    "retrieval_method": "hybrid",
                    "retry_count": 0,
                    "validation_passed": True,
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """User feedback on a response."""
    session_id: str
    query: str
    answer: str
    rating: str = Field(..., description="thumbs_up | thumbs_down | correction")
    correction: Optional[str] = Field(None, description="Corrected answer if rating is correction")


class FeedbackResponse(BaseModel):
    """Feedback acknowledgement."""
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    faiss_index_loaded: bool
