"""
Unit tests for the HybridRetriever.
Tests use a mock FAISS index to avoid needing real embeddings.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.retriever import HybridRetriever, RetrievedChunk


@pytest.fixture
def mock_retriever():
    """Create a HybridRetriever with all external dependencies mocked."""
    with patch("src.retrieval.retriever.FAISS") as mock_faiss_cls, \
         patch("src.retrieval.retriever.OpenAIEmbeddings"):

        # Build a fake docstore
        fake_doc = MagicMock()
        fake_doc.page_content = "To reset your password, click Forgot Password on the login page."
        fake_doc.metadata = {
            "id": "faq_001",
            "category": "account",
            "source": "account_faq.pdf",
            "page": 1,
            "chunk_index": 0,
        }

        mock_faiss_instance = MagicMock()
        mock_faiss_instance.docstore._dict = {"doc_0": fake_doc}
        mock_faiss_instance.similarity_search_with_score.return_value = [
            (fake_doc, 0.15)
        ]
        mock_faiss_cls.load_local.return_value = mock_faiss_instance

        # Patch Path.exists to return True
        with patch("src.retrieval.retriever.Path.exists", return_value=True):
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever.top_k = 5
            retriever.embeddings = MagicMock()
            retriever.vector_store = mock_faiss_instance
            retriever.bm25_corpus = [fake_doc.page_content]
            retriever.bm25_metadatas = [fake_doc.metadata]

            from rank_bm25 import BM25Okapi
            retriever.bm25 = BM25Okapi([fake_doc.page_content.lower().split()])

            yield retriever


class TestRetrievedChunk:
    def test_dataclass_fields(self):
        chunk = RetrievedChunk(
            text="Hello world",
            score=0.8,
            source="test.pdf",
            category="account",
        )
        assert chunk.text == "Hello world"
        assert chunk.score == 0.8
        assert chunk.retrieval_method == "semantic"

    def test_default_values(self):
        chunk = RetrievedChunk(text="test", score=0.5, source="s.pdf", category="billing")
        assert chunk.page == 1
        assert chunk.chunk_index == 0


class TestHybridRetriever:
    def test_semantic_search_returns_chunks(self, mock_retriever):
        results = mock_retriever.semantic_search("reset password", k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], RetrievedChunk)

    def test_keyword_search_returns_chunks(self, mock_retriever):
        results = mock_retriever.keyword_search("password reset", k=3)
        assert isinstance(results, list)

    def test_hybrid_search_returns_chunks(self, mock_retriever):
        results = mock_retriever.hybrid_search("how do I reset my password", k=2)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_search_uses_rrf(self, mock_retriever):
        """RRF scores should be positive floats."""
        results = mock_retriever.hybrid_search("password", k=2)
        for chunk in results:
            assert chunk.score > 0

    def test_retrieve_delegates_to_hybrid(self, mock_retriever):
        """retrieve() should call hybrid_search."""
        mock_retriever.hybrid_search = MagicMock(return_value=[
            RetrievedChunk("text", 0.5, "src.pdf", "account")
        ])
        mock_retriever.retrieve("test query")
        mock_retriever.hybrid_search.assert_called_once()
