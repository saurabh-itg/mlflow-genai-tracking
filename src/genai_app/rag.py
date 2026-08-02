"""The traced RAG pipeline.

Every stage is an MLflow span. When you open a trace in the MLflow UI you get the
retrieve -> build_prompt -> generate waterfall, with inputs, outputs, and per-span
latency, which is the thing that actually makes a GenAI regression debuggable.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
from mlflow.entities import SpanType

from . import prompts as prompt_lib
from .llm import BaseLLM, LLMResponse, estimate_cost_usd, get_llm
from .retriever import TfidfRetriever


@dataclass
class RagConfig:
    corpus_dir: str
    chunk_size: int = 120
    chunk_overlap: int = 20
    top_k: int = 4
    temperature: float = 0.0
    prompt_version: str = prompt_lib.DEFAULT_PROMPT
    max_tokens: int = 400

    def as_params(self) -> dict:
        d = asdict(self)
        d["corpus_dir"] = Path(self.corpus_dir).name
        return d


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    llm: LLMResponse | None = None
    cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    trace_id: str | None = None

    @property
    def context_blob(self) -> str:
        return "\n".join(self.contexts)


class RagPipeline:
    def __init__(self, config: RagConfig, llm: BaseLLM | None = None):
        self.config = config
        self.llm = llm or get_llm()
        self.retriever = TfidfRetriever.from_corpus(
            config.corpus_dir, config.chunk_size, config.chunk_overlap
        )

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def retrieve(self, question: str) -> list[dict]:
        hits = self.retriever.search(question, self.config.top_k)
        return [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": round(s, 4), "text": c.text}
            for c, s in hits
        ]

    @mlflow.trace(span_type=SpanType.PARSER)
    def build_prompt(self, question: str, hits: list[dict]) -> str:
        context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)
        return prompt_lib.render(self.config.prompt_version, context, question)

    @mlflow.trace(span_type=SpanType.LLM)
    def generate(self, prompt: str) -> dict:
        resp = self.llm.complete(prompt, self.config.temperature, self.config.max_tokens)
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_attributes(
                {
                    "llm.model": resp.model,
                    "llm.prompt_tokens": resp.prompt_tokens,
                    "llm.completion_tokens": resp.completion_tokens,
                    "llm.total_tokens": resp.total_tokens,
                    "llm.latency_ms": resp.latency_ms,
                    "llm.cost_usd": estimate_cost_usd(resp),
                }
            )
        return asdict(resp)

    @mlflow.trace(span_type=SpanType.CHAIN)
    def answer(self, question: str) -> RagResult:
        import time

        t0 = time.perf_counter()
        hits = self.retrieve(question)
        prompt = self.build_prompt(question, hits)
        raw = self.generate(prompt)
        resp = LLMResponse(**raw)

        result = RagResult(
            question=question,
            answer=resp.text,
            contexts=[h["text"] for h in hits],
            retrieved_doc_ids=[h["doc_id"] for h in hits],
            retrieval_scores=[h["score"] for h in hits],
            llm=resp,
            cost_usd=estimate_cost_usd(resp),
            total_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        try:
            result.trace_id = mlflow.get_current_active_span().request_id
        except Exception:
            result.trace_id = None
        return result


def build_pipeline(**overrides) -> RagPipeline:
    corpus = overrides.pop("corpus_dir", os.getenv("GENAI_CORPUS", "data/corpus"))
    return RagPipeline(RagConfig(corpus_dir=corpus, **overrides))
