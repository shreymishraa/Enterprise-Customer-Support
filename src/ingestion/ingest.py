"""
Ingestion module: loads, chunks, embeds, and stores documents in FAISS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = Path("data/customer_support_faq.json")
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> List[Dict[str, Any]]:
    """Load the raw JSON corpus."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_raw_texts(corpus: List[Dict[str, Any]]) -> List[tuple[str, Dict]]:
    """
    Combine question + answer into a single text block per document,
    keeping source metadata.
    Returns list of (text, metadata) tuples.
    """
    texts_and_meta = []
    for doc in corpus:
        combined = (
            f"Category: {doc['category']}\n"
            f"Q: {doc['question']}\n"
            f"A: {doc['answer']}"
        )
        meta = {
            "id": doc["id"],
            "category": doc["category"],
            "source": doc["source"],
            "page": doc.get("page", 1),
        }
        texts_and_meta.append((combined, meta))
    return texts_and_meta


def chunk_documents(
    texts_and_meta: List[tuple[str, Dict]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[List[str], List[Dict]]:
    """
    Split each document into overlapping chunks.
    Returns (all_chunks, all_metadata).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks: List[str] = []
    all_meta: List[Dict] = []

    for text, meta in tqdm(texts_and_meta, desc="Chunking"):
        splits = splitter.split_text(text)
        for i, chunk in enumerate(splits):
            all_chunks.append(chunk)
            all_meta.append({**meta, "chunk_index": i})

    return all_chunks, all_meta


def build_and_save_faiss(
    chunks: List[str],
    metadatas: List[Dict],
    save_path: Path,
) -> FAISS:
    """
    Embed chunks with OpenAI and persist the FAISS index to disk.
    """
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
    print(f"Embedding {len(chunks)} chunks …")
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))
    print(f"✅ FAISS index saved to {save_path}")
    return vector_store


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    """End-to-end ingestion pipeline."""
    print("📂 Loading corpus …")
    corpus = load_corpus(DATA_PATH)
    print(f"   Loaded {len(corpus)} documents.")

    texts_and_meta = build_raw_texts(corpus)
    chunks, metadatas = chunk_documents(texts_and_meta)
    print(f"   Created {len(chunks)} chunks.")

    build_and_save_faiss(chunks, metadatas, FAISS_INDEX_PATH)
    print("🎉 Ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
