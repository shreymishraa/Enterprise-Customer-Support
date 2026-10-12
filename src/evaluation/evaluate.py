"""
Evaluation module: BLEU, ROUGE, and relevance scoring.
Compares baseline RAG vs. agentic RAG pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import nltk
import numpy as np
from dotenv import load_dotenv
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

load_dotenv()

# Download required NLTK data
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


# ── Test set ──────────────────────────────────────────────────────────────────

EVAL_PAIRS: List[Dict] = [
    {
        "query": "How do I reset my password?",
        "reference": (
            "To reset your password, click Forgot Password on the login page, "
            "enter your registered email, and follow the link sent to you."
        ),
    },
    {
        "query": "What is the refund policy?",
        "reference": (
            "We offer a 30-day money-back guarantee. Contact billing@company.com "
            "for a full refund within 5-7 business days."
        ),
    },
    {
        "query": "How do I cancel my subscription?",
        "reference": (
            "Go to Settings > Billing > Cancel Subscription. Your subscription "
            "remains active until the end of the billing period."
        ),
    },
    {
        "query": "How long does shipping take?",
        "reference": (
            "Standard shipping takes 5-7 business days and is free for orders over $50. "
            "Express takes 2-3 business days for $12.99."
        ),
    },
    {
        "query": "How do I enable two-factor authentication?",
        "reference": (
            "Go to Settings > Security > Two-Factor Authentication, enable it, "
            "and scan the QR code with an authenticator app."
        ),
    },
]


# ── Scoring utilities ─────────────────────────────────────────────────────────

def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus-level BLEU score."""
    refs = [[r.lower().split()] for r in references]
    hyps = [h.lower().split() for h in hypotheses]
    smoother = SmoothingFunction().method1
    return corpus_bleu(refs, hyps, smoothing_function=smoother)


def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """Compute average ROUGE-1, ROUGE-2 and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in agg:
            agg[key].append(scores[key].fmeasure)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def compute_relevance(retrieved_contexts: List[List[str]], references: List[str]) -> float:
    """
    Simple term-overlap relevance metric.
    Measures what fraction of reference terms appear in the retrieved context.
    """
    scores = []
    for ctx_list, ref in zip(retrieved_contexts, references):
        ref_terms = set(ref.lower().split())
        ctx_text = " ".join(ctx_list).lower()
        ctx_terms = set(ctx_text.split())
        if not ref_terms:
            continue
        overlap = len(ref_terms & ctx_terms) / len(ref_terms)
        scores.append(overlap)
    return float(np.mean(scores)) if scores else 0.0


# ── Evaluation runners ────────────────────────────────────────────────────────

def evaluate_baseline(pairs: List[Dict] = EVAL_PAIRS) -> Dict:
    """
    Baseline RAG: direct rag_tool call without the multi-agent workflow.
    """
    from src.tools.rag_tool import rag_tool

    references, hypotheses, contexts = [], [], []
    for pair in pairs:
        result = rag_tool(query=pair["query"])
        references.append(pair["reference"])
        hypotheses.append(result["answer"])
        contexts.append(result["retrieved_context"])

    return {
        "mode": "baseline_rag",
        "bleu": compute_bleu(references, hypotheses),
        "rouge": compute_rouge(references, hypotheses),
        "relevance": compute_relevance(contexts, references),
    }


def evaluate_agentic(pairs: List[Dict] = EVAL_PAIRS) -> Dict:
    """
    Agentic RAG: runs the full LangGraph multi-agent workflow.
    """
    from src.graph.workflow import run_query

    references, hypotheses, contexts = [], [], []
    for pair in pairs:
        state = run_query(pair["query"], session_id="eval")
        references.append(pair["reference"])
        hypotheses.append(state.get("answer") or "")
        contexts.append(state.get("retrieved_context", []))

    return {
        "mode": "agentic_rag",
        "bleu": compute_bleu(references, hypotheses),
        "rouge": compute_rouge(references, hypotheses),
        "relevance": compute_relevance(contexts, references),
    }


def run_evaluation(output_path: str = "data/eval_results.json") -> Tuple[Dict, Dict]:
    """Run both evaluations and save results."""
    print("🔍 Evaluating Baseline RAG …")
    baseline = evaluate_baseline()

    print("🤖 Evaluating Agentic RAG …")
    agentic = evaluate_agentic()

    results = {"baseline": baseline, "agentic": agentic}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n📊 Evaluation Results:")
    print("=" * 50)
    for mode, res in results.items():
        print(f"\n{mode.upper()}")
        print(f"  BLEU:       {res['bleu']:.4f}")
        print(f"  ROUGE-1:    {res['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2:    {res['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L:    {res['rouge']['rougeL']:.4f}")
        print(f"  Relevance:  {res['relevance']:.4f}")
    print(f"\n✅ Results saved to {output_path}")

    return baseline, agentic


if __name__ == "__main__":
    run_evaluation()
