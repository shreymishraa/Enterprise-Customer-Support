"""
Retrieval module: FAISS similarity search + BM25 keyword search (hybrid retrieval).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
TOP_K = int(os.getenv("TOP_K", 5))


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""
    text: str
    score: float
    source: str
    category: str
    page: int = 1
    chunk_index: int = 0
    retrieval_method: str = "semantic"


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Semantic search via FAISS (dense)
    - Keyword search via BM25 (sparse)

    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, index_path: Path = FAISS_INDEX_PATH, top_k: int = TOP_K):
        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        self.vector_store: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[str] = []
        self.bm25_metadatas: List[dict] = []
        self._load_index(index_path)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_index(self, index_path: Path) -> None:
        """Load FAISS index and build BM25 corpus from stored docs."""
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run `python -m src.ingestion.ingest` first."
            )
        self.vector_store = FAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        # Build BM25 from the FAISS docstore
        docs = list(self.vector_store.docstore._dict.values())
        self.bm25_corpus = [d.page_content for d in docs]
        self.bm25_metadatas = [d.metadata for d in docs]
        tokenized = [doc.lower().split() for doc in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, k: int) -> List[RetrievedChunk]:
        """Dense vector search using FAISS."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        chunks = []
        for doc, score in results:
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    score=float(1 - score),   # cosine similarity (higher = better)
                    source=doc.metadata.get("source", "unknown"),
                    category=doc.metadata.get("category", "general"),
                    page=int(doc.metadata.get("page", 1)),
                    chunk_index=int(doc.metadata.get("chunk_index", 0)),
                    retrieval_method="semantic",
                )
            )
        return chunks

    def keyword_search(self, query: str, k: int) -> List[RetrievedChunk]:
        """Sparse keyword search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        chunks = []
        for idx in top_indices:
            meta = self.bm25_metadatas[idx]
            chunks.append(
                RetrievedChunk(
                    text=self.bm25_corpus[idx],
                    score=float(scores[idx]),
                    source=meta.get("source", "unknown"),
                    category=meta.get("category", "general"),
                    page=int(meta.get("page", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    retrieval_method="keyword",
                )
            )
        return chunks

    def hybrid_search(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion of semantic + keyword results.
        RRF score = sum(1 / (rank + 60)) across both lists.
        """
        k = k or self.top_k
        n_candidates = k * 3   # retrieve more, then fuse and truncate

        sem_results = self.semantic_search(query, n_candidates)
        kw_results = self.keyword_search(query, n_candidates)

        # Build RRF score map: text → score
        rrf_scores: dict[str, float] = {}
        text_to_chunk: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(sem_results):
            key = chunk.text[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 60)
            text_to_chunk[key] = chunk

        for rank, chunk in enumerate(kw_results):
            key = chunk.text[:200]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + 60)
            if key not in text_to_chunk:
                text_to_chunk[key] = chunk
            else:
                text_to_chunk[key].retrieval_method = "hybrid"

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
        fused = []
        for key in sorted_keys:
            chunk = text_to_chunk[key]
            chunk.score = rrf_scores[key]
            fused.append(chunk)
        return fused

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Public interface — always runs hybrid search."""
        return self.hybrid_search(query, k)


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Return (and lazily initialize) the global retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
