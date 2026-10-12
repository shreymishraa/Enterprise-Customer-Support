"""
FastAPI backend for the Enterprise Customer Support RAG system.

Endpoints:
  POST /query    — Run the multi-agent RAG pipeline
  GET  /health   — Health check
  POST /feedback — Submit user feedback
  GET  /docs     — Swagger UI (built-in)
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.graph.workflow import run_query

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Customer Support RAG API",
    description=(
        "A multi-agent Retrieval-Augmented Generation system for enterprise "
        "customer support automation. Built with LangGraph, FAISS, and OpenAI."
    ),
    version="1.0.0",
    contact={"name": "Support Team", "email": "support@company.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_LOG = Path(os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl"))
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))


# ── Lazy-loaded workflow ───────────────────────────────────────────────────────

_workflow_ready = False


def ensure_workflow():
    """Load the workflow (and retriever) on first use."""
    global _workflow_ready
    if not _workflow_ready:
        from src.graph.workflow import get_workflow  # noqa: F401 — triggers compilation
        _workflow_ready = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return API health status and index availability."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        faiss_index_loaded=FAISS_INDEX_PATH.exists(),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Run the multi-agent RAG pipeline for a customer support query.

    The pipeline runs: Query Understanding → RAG Retrieval → Sentiment Analysis
    → Validator → (optional) Escalation Agent.
    """
    try:
        ensure_workflow()
        state = run_query(
            query=request.query,
            session_id=request.session_id,
        )

        return QueryResponse(
            session_id=request.session_id,
            answer=state.get("answer") or "I could not generate an answer. Please try again.",
            sources=state.get("sources", []),
            retrieved_context=state.get("retrieved_context", []),
            agents_invoked=state.get("agents_invoked", []),
            confidence=state.get("confidence", 0.0),
            intent=state.get("intent"),
            sentiment=state.get("sentiment"),
            urgency_score=state.get("urgency_score", 0.0),
            needs_escalation=state.get("needs_escalation", False),
            escalation_reason=state.get("escalation_reason"),
            retrieval_method=state.get("retrieval_method", "hybrid"),
            retry_count=state.get("retry_count", 0),
            validation_passed=state.get("validation_passed", True),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"FAISS index not found. Run ingestion first: {e}",
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal pipeline error: {e}\n{tb}",
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def feedback_endpoint(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user feedback (thumbs up/down or correction) for a response.
    Feedback is logged to JSONL for future fine-tuning or evaluation.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": request.session_id,
        "query": request.query,
        "answer": request.answer,
        "rating": request.rating,
        "correction": request.correction,
    }
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")

    return FeedbackResponse(
        status="success",
        message="Thank you for your feedback! It helps us improve.",
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
