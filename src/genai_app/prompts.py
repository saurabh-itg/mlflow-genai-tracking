"""Prompt templates, versioned.

Each template is registered in the MLflow Prompt Registry by scripts/01_log_prompts.py,
so a run's prompt version is an auditable artifact rather than a string buried in code.
"""

PROMPTS: dict[str, str] = {
    "v1-terse": (
        "Answer the question using only the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
    "v2-grounded": (
        "You are an industrial engineering assistant. Answer strictly from the context.\n"
        "If the context does not contain the answer, say so instead of guessing.\n"
        "Cite the source id in square brackets after each claim.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
    "v3-grounded-cot": (
        "You are an industrial engineering assistant.\n"
        "Step 1: identify which context passages are relevant.\n"
        "Step 2: answer strictly from those passages, citing source ids in brackets.\n"
        "Step 3: if the context is insufficient, say so explicitly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ),
}

DEFAULT_PROMPT = "v2-grounded"


def render(version: str, context: str, question: str) -> str:
    return PROMPTS[version].format(context=context, question=question)
