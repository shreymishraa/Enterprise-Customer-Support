"""Embedding selection with an offline-safe local fallback."""
from __future__ import annotations

import hashlib
import os
import re
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class LocalHashEmbeddings(Embeddings):
    """Small deterministic bag-of-words embedding for local demos and tests."""

    dimensions = 384

    def _embed(self, text: str) -> List[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings():
    """Use OpenAI when configured; otherwise keep the project runnable offline."""
    if os.getenv("OPENAI_API_KEY") and os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() != "true":
        return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
    return LocalHashEmbeddings()
