"""LLM provider abstraction.

The repo is offline-first: the default provider is a deterministic, seeded stub so
`make demo` produces identical MLflow runs on any machine with no API key. Set
GENAI_PROVIDER=openai (plus OPENAI_API_KEY) to route the exact same pipeline at a
real model -- nothing else in the codebase changes.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _count_tokens(text: str) -> int:
    """Whitespace/punctuation token approximation -- good enough for cost curves."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


class BaseLLM:
    name = "base"

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 400) -> LLMResponse:
        raise NotImplementedError


class DeterministicLLM(BaseLLM):
    """Extractive stub model.

    It answers by selecting the sentences in the supplied context that overlap most
    with the question. This is deliberately *grounded by construction*, which makes
    the groundedness / relevance scorers in scorers.py produce a meaningful signal
    that responds to retrieval quality instead of random noise.

    `temperature` controls how many lower-ranked sentences leak into the answer, so
    the parameter sweep in scripts/02_run_experiments.py shows real metric movement.
    """

    name = "deterministic-extractive-v1"

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 400) -> LLMResponse:
        start = time.perf_counter()

        question = ""
        context = prompt
        if "Question:" in prompt:
            context, _, question = prompt.rpartition("Question:")
        if "Context:" in context:
            context = context.split("Context:", 1)[1]

        q_terms = {t.lower() for t in re.findall(r"\w+", question) if len(t) > 2}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 25]

        scored = []
        for s in sentences:
            terms = {t.lower() for t in re.findall(r"\w+", s) if len(t) > 2}
            if not terms:
                continue
            overlap = len(q_terms & terms) / (len(q_terms) or 1)
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])

        # The stub honours two instructions that the prompt variants actually differ on,
        # so a prompt change moves the metrics instead of being cosmetic:
        #   1. an explicit "say so instead of guessing" clause -> abstain on weak retrieval
        #   2. a "identify which passages are relevant" clause -> drop zero-overlap sentences
        abstain_allowed = "instead of guessing" in prompt or "insufficient" in prompt
        strict_filter = "identify which context passages are relevant" in prompt

        top_score = scored[0][0] if scored else 0.0
        if abstain_allowed and top_score < 0.20:
            answer = "The context does not contain the answer, so I cannot answer that."
            latency_ms = 40 + 0.06 * len(prompt)
            return LLMResponse(
                text=answer,
                prompt_tokens=_count_tokens(prompt),
                completion_tokens=_count_tokens(answer),
                latency_ms=round(latency_ms, 2),
                model=self.name,
                extra={"abstained": True},
            )

        # higher temperature -> pull in weaker sentences (simulated drift)
        keep = 2 + int(round(temperature * 4))
        floor = 0.15 if strict_filter else 0.0
        picked = [s for score, s in scored[:keep] if score > floor] or [s for _, s in scored[:1]]
        answer = " ".join(picked)[: max_tokens * 4]

        if not answer:
            answer = "I don't have enough information in the provided context to answer that."

        # simulated network/inference latency that scales with prompt size
        latency_ms = 40 + 0.06 * len(prompt) + 0.8 * len(picked)
        time.sleep(min(latency_ms / 1000.0, 0.05))

        return LLMResponse(
            text=answer,
            prompt_tokens=_count_tokens(prompt),
            completion_tokens=_count_tokens(answer),
            latency_ms=round(latency_ms, 2),
            model=self.name,
            extra={"sentences_kept": len(picked)},
        )


class OpenAILLM(BaseLLM):
    """Thin wrapper. Only imported when GENAI_PROVIDER=openai."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI()
        self.name = model

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 400) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=round(latency_ms, 2),
            model=self.name,
        )


# rough $/1M tokens, used to turn token counts into a cost metric MLflow can chart
COST_PER_MTOK = {
    "deterministic-extractive-v1": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_cost_usd(resp: LLMResponse) -> float:
    p_in, p_out = COST_PER_MTOK.get(resp.model, (0.0, 0.0))
    return round(resp.prompt_tokens / 1e6 * p_in + resp.completion_tokens / 1e6 * p_out, 8)


def get_llm() -> BaseLLM:
    provider = os.getenv("GENAI_PROVIDER", "deterministic").lower()
    if provider == "openai":
        return OpenAILLM(os.getenv("GENAI_MODEL", "gpt-4o-mini"))
    return DeterministicLLM()
