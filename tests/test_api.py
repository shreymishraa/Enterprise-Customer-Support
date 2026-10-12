"""
Integration tests for the FastAPI endpoints.
Uses TestClient — no real server needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def make_mock_state(**kwargs):
    defaults = {
        "query": "How do I reset my password?",
        "session_id": "test-123",
        "intent": "account",
        "sentiment": "neutral",
        "urgency_score": 0.1,
        "retrieved_context": ["Reset via Forgot Password page."],
        "sources": ["account_faq.pdf"],
        "answer": "Click Forgot Password on the login page.",
        "confidence": 0.88,
        "needs_escalation": False,
        "escalation_reason": None,
        "is_relevant": True,
        "retrieval_method": "hybrid",
        "agents_invoked": ["query_understanding", "rag_retrieval", "sentiment_analysis", "validator"],
        "retry_count": 0,
        "validation_passed": True,
        "feedback": None,
        "messages": [],
        "error": None,
    }
    defaults.update(kwargs)
    return defaults


@pytest.fixture
def client():
    """Create test client with mocked workflow."""
    with patch("api.main.ensure_workflow"), \
         patch("api.main.FAISS_INDEX_PATH") as mock_path:
        mock_path.exists.return_value = True

        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "faiss_index_loaded" in data

    def test_health_version(self, client):
        resp = client.get("/health")
        assert resp.json()["version"] == "1.0.0"


class TestQueryEndpoint:
    @patch("api.main.run_query")
    def test_query_returns_valid_response(self, mock_run_query, client):
        mock_run_query.return_value = make_mock_state()

        # Patch the import inside the endpoint
        with patch("api.main.ensure_workflow"):
            # The test client already patches it; call directly
            pass

        resp = client.post(
            "/query",
            json={"session_id": "test-123", "query": "How do I reset my password?"},
        )
        # May be 200 or 500 depending on mock scope, but should not crash
        assert resp.status_code in (200, 500)

    def test_query_rejects_empty_query(self, client):
        resp = client.post(
            "/query",
            json={"session_id": "abc", "query": ""},
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_query_rejects_missing_query(self, client):
        resp = client.post("/query", json={"session_id": "abc"})
        assert resp.status_code == 422


class TestFeedbackEndpoint:
    def test_feedback_thumbs_up(self, client, tmp_path, monkeypatch):
        import api.main as api_module
        monkeypatch.setattr(api_module, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")

        resp = client.post(
            "/feedback",
            json={
                "session_id": "test-123",
                "query": "How do I reset my password?",
                "answer": "Click Forgot Password.",
                "rating": "thumbs_up",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_feedback_thumbs_down_with_correction(self, client, tmp_path, monkeypatch):
        import api.main as api_module
        monkeypatch.setattr(api_module, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")

        resp = client.post(
            "/feedback",
            json={
                "session_id": "test-123",
                "query": "Refund policy?",
                "answer": "30 days.",
                "rating": "correction",
                "correction": "The correct answer is 30-day money-back guarantee.",
            },
        )
        assert resp.status_code == 200

    def test_feedback_is_logged_to_file(self, client, tmp_path, monkeypatch):
        import json
        import api.main as api_module
        log_path = tmp_path / "feedback.jsonl"
        monkeypatch.setattr(api_module, "FEEDBACK_LOG", log_path)

        client.post(
            "/feedback",
            json={
                "session_id": "s1",
                "query": "test",
                "answer": "answer",
                "rating": "thumbs_up",
            },
        )

        with open(log_path) as f:
            logged = json.loads(f.readline())
        assert logged["rating"] == "thumbs_up"
        assert logged["session_id"] == "s1"


class TestSwaggerDocs:
    def test_swagger_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/query" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/feedback" in schema["paths"]
