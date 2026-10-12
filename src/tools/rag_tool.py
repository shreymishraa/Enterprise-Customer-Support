"""
RAG Tool: exposes the retrieve-then-generate pipeline as a callable LangGraph tool.

Agents can call `rag_tool(query, session_id)` to get structured JSON output
containing the answer, retrieved context, sources and a confidence score.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.retrieval.retriever import RetrievedChunk, get_retriever

load_dotenv()

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

SYSTEM_PROMPT = """You are a helpful enterprise customer support assistant.
Your task is to answer the user's question ONLY using the provided context.

Rules:
1. If the context is relevant, answer precisely and cite sources.
2. If the context is NOT relevant, say you don't have enough information.
3. Always be polite and professional.
4. Output MUST be valid JSON with these keys:
   - "answer": string — your answer to the user's question
   - "sources": list of strings — source filenames referenced
   - "confidence": float between 0.0 and 1.0 — how confident you are
   - "needs_escalation": boolean — true if the issue needs human intervention
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

User Question: {query}

Respond in JSON format only."""


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk.source} (page {chunk.page})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def check_relevance(query: str, chunks: List[RetrievedChunk], threshold: float = 0.2) -> bool:
    """
    Basic relevance check: at least one chunk must have a score above threshold,
    and key query terms must appear in retrieved text.
    """
    if not chunks:
        return False
    max_score = max(c.score for c in chunks)
    if max_score < threshold:
        return False
    # Term overlap check
    query_terms = set(query.lower().split())
    combined_text = " ".join(c.text.lower() for c in chunks)
    overlap = sum(1 for term in query_terms if term in combined_text)
    return overlap >= max(1, len(query_terms) // 3)


def rag_tool(
    query: str,
    k: Optional[int] = None,
    rewrite: bool = False,
) -> dict:
    """
    Core RAG tool callable by agents.

    Args:
        query: The user's question.
        k: Number of documents to retrieve (defaults to TOP_K env var).
        rewrite: If True, expand the query for better recall.

    Returns:
        dict with keys: answer, sources, retrieved_context, confidence,
                        needs_escalation, is_relevant, retrieval_method.
    """
    retriever = get_retriever()
    effective_query = query

    # Optional query rewriting for retry scenarios
    if rewrite:
        effective_query = f"{query} support help information guide policy"

    chunks = retriever.retrieve(effective_query, k=k)

    is_relevant = check_relevance(query, chunks)
    context_str = format_context(chunks)
    sources = list({c.source for c in chunks})
    retrieved_texts = [c.text for c in chunks]
    retrieval_method = chunks[0].retrieval_method if chunks else "none"

    # Generate response
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(
                context=context_str,
                query=query,
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        result = {
            "answer": f"I encountered an issue generating a response. Please try again. (Error: {e})",
            "sources": sources,
            "confidence": 0.0,
            "needs_escalation": True,
        }

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", sources),
        "confidence": float(result.get("confidence", 0.5)),
        "needs_escalation": bool(result.get("needs_escalation", False)),
        "retrieved_context": retrieved_texts,
        "is_relevant": is_relevant,
        "retrieval_method": retrieval_method,
    }
