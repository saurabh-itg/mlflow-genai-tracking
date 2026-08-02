"""Deterministic evaluation scorers.

LLM-as-judge is the usual choice, but a judge makes CI non-reproducible and costs money.
These are heuristic proxies for the same four dimensions, and every one of them is logged
to MLflow as a metric so you can compare configurations. Swap in mlflow.genai judges by
setting GENAI_JUDGE=1 (see scripts/03_evaluate.py).
"""
from __future__ import annotations

import re


def _terms(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\w+", text) if len(t) > 2}


def groundedness(answer: str, context: str) -> float:
    """Share of answer terms that actually appear in the retrieved context.

    Low groundedness is the fingerprint of hallucination.
    """
    a, c = _terms(answer), _terms(context)
    if not a:
        return 0.0
    return round(len(a & c) / len(a), 4)


def answer_relevance(answer: str, question: str) -> float:
    """Share of question terms the answer engages with."""
    q, a = _terms(question), _terms(answer)
    if not q:
        return 0.0
    return round(len(q & a) / len(q), 4)


def correctness(answer: str, expected_facts: list[str]) -> float:
    """Fraction of required facts present in the answer."""
    if not expected_facts:
        return 0.0
    low = answer.lower()
    hits = sum(1 for f in expected_facts if f.lower() in low)
    return round(hits / len(expected_facts), 4)


def retrieval_hit_rate(retrieved_doc_ids: list[str], relevant_doc_id: str) -> float:
    return 1.0 if relevant_doc_id in retrieved_doc_ids else 0.0


def retrieval_mrr(retrieved_doc_ids: list[str], relevant_doc_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == relevant_doc_id:
            return round(1.0 / rank, 4)
    return 0.0


def refusal(answer: str) -> float:
    markers = ("don't have enough information", "does not contain the answer",
               "not contain the answer", "cannot answer", "insufficient")
    return 1.0 if any(m in answer.lower() for m in markers) else 0.0
