# Responsible AI in Production

Fairness and explainability work best as instrumentation, not as a review gate at the end.

For NLP-based resume shortlisting we track three or more fairness metrics per model version:
demographic parity difference, equalised odds difference, and selection rate by group. The metrics
are logged alongside accuracy in the same experiment tracking system, so a model that improves recall
by degrading parity is visible in the same comparison view rather than in a separate audit document.

For time-series anomaly detection we attach SHAP and LIME explanations to each flagged event. An
operator who sees an alert also sees which channels drove it, which is the difference between an
alert that gets actioned and one that gets muted.

Synthetic data generation covers the long tail. A pipeline processes more than three thousand pages
of technical documentation into question-answer pairs, and RAG quality is evaluated against a
ground-truth benchmark rather than against vibes. Groundedness, answer relevance, and retrieval hit
rate are the three numbers that get watched.
