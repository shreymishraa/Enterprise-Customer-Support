"""
Unit tests for the RAG tool and agent nodes.
All LLM and vector store calls are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.retriever import RetrievedChunk


# ── RAG Tool Tests ────────────────────────────────────────────────────────────

class TestRagTool:
    def _make_chunk(self, text="Reset your password via Forgot Password."):
        return RetrievedChunk(
            text=text,
            score=0.85,
            source="account_faq.pdf",
            category="account",
        )

    @patch("src.tools.rag_tool.get_retriever")
    @patch("src.tools.rag_tool.ChatOpenAI")
    def test_rag_tool_returns_valid_dict(self, mock_llm_cls, mock_get_retriever):
        from src.tools.rag_tool import rag_tool
        import json

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [self._make_chunk()]
        mock_get_retriever.return_value = mock_retriever

        # Mock LLM
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({
            "answer": "Click Forgot Password on the login page.",
            "sources": ["account_faq.pdf"],
            "confidence": 0.9,
            "needs_escalation": False,
        })
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        result = rag_tool("How do I reset my password?")

        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "retrieved_context" in result
        assert "needs_escalation" in result
        assert isinstance(result["confidence"], float)

    @patch("src.tools.rag_tool.get_retriever")
    @patch("src.tools.rag_tool.ChatOpenAI")
    def test_rag_tool_handles_json_error(self, mock_llm_cls, mock_get_retriever):
        from src.tools.rag_tool import rag_tool

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [self._make_chunk()]
        mock_get_retriever.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "NOT VALID JSON AT ALL"
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        result = rag_tool("test query")
        # Should not crash; should return fallback
        assert "answer" in result
        assert result["needs_escalation"] is True

    def test_check_relevance_with_good_chunks(self):
        from src.tools.rag_tool import check_relevance

        chunks = [
            RetrievedChunk(
                text="Password reset steps: click Forgot Password",
                score=0.8,
                source="s.pdf",
                category="account",
            )
        ]
        assert check_relevance("reset password", chunks, threshold=0.1) is True

    def test_check_relevance_with_empty_chunks(self):
        from src.tools.rag_tool import check_relevance
        assert check_relevance("reset password", []) is False

    def test_format_context(self):
        from src.tools.rag_tool import format_context
        chunks = [
            RetrievedChunk("Some text", 0.9, "doc.pdf", "account", page=2)
        ]
        ctx = format_context(chunks)
        assert "doc.pdf" in ctx
        assert "Some text" in ctx
        assert "[1]" in ctx


# ── Agent Node Tests ──────────────────────────────────────────────────────────

def make_state(**kwargs):
    """Helper to create a minimal AgentState dict."""
    defaults = {
        "query": "How do I reset my password?",
        "session_id": "test-session",
        "intent": None,
        "sentiment": None,
        "urgency_score": 0.0,
        "retrieved_context": [],
        "sources": [],
        "answer": None,
        "confidence": 0.0,
        "needs_escalation": False,
        "escalation_reason": None,
        "is_relevant": False,
        "retrieval_method": "hybrid",
        "agents_invoked": [],
        "retry_count": 0,
        "validation_passed": False,
        "feedback": None,
        "messages": [],
        "error": None,
    }
    defaults.update(kwargs)
    return defaults


class TestQueryUnderstandingAgent:
    @patch("src.agents.query_understanding.ChatOpenAI")
    def test_sets_intent(self, mock_llm_cls):
        import json
        from src.agents.query_understanding import query_understanding_node

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = json.dumps({
            "intent": "account",
            "rephrased_query": "How to reset account password?",
            "is_complaint": False,
        })
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        state = make_state()
        result = query_understanding_node(state)

        assert result["intent"] == "account"
        assert "query_understanding" in result["agents_invoked"]

    @patch("src.agents.query_understanding.ChatOpenAI")
    def test_handles_llm_failure(self, mock_llm_cls):
        from src.agents.query_understanding import query_understanding_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")
        mock_llm_cls.return_value = mock_llm

        state = make_state()
        result = query_understanding_node(state)

        # Should fall back to "other" intent without crashing
        assert result["intent"] == "other"


class TestEscalationAgent:
    @patch("src.agents.escalation.rag_tool")
    def test_escalates_when_flagged(self, mock_rag_tool):
        from src.agents.escalation import escalation_node

        mock_rag_tool.return_value = {"retrieved_context": []}

        state = make_state(needs_escalation=True)
        result = escalation_node(state)

        assert result["needs_escalation"] is True
        assert "ESC-" in result["answer"]

    @patch("src.agents.escalation.rag_tool")
    def test_no_escalation_for_normal_query(self, mock_rag_tool):
        from src.agents.escalation import escalation_node

        mock_rag_tool.return_value = {"retrieved_context": []}

        state = make_state(
            needs_escalation=False,
            urgency_score=0.1,
            confidence=0.9,
            validation_passed=True,
        )
        result = escalation_node(state)

        assert result["needs_escalation"] is False
