# Enterprise RAG: Ingestion Is Where It Fails

Document search usually fails at ingestion long before it fails at retrieval. Teams tune rerankers
for weeks while the index is quietly three months stale, or contains four copies of the same policy
document with different headers.

Knowledge Nexus solves the ingestion half first. A Temporal-based worker system absorbs
high-concurrency SharePoint fetching, with retries and backoff expressed as workflow logic rather
than as bespoke queue code. An SHA-256 incremental sync engine hashes content at the chunk level and
tracks change across thousands of files, so re-ingestion touches only what actually moved. The result
is zero redundant ingestion: the Qdrant index stays current without periodic full rebuilds.

Retrieval is hybrid. Sparse BM25 matching catches exact identifiers, part numbers, and acronyms that
dense embeddings blur together, while dense vectors handle paraphrase. Scores are fused before
reranking.

Every generated answer passes through Guardrails AI validation before it reaches a user or a SQL
engine. The validators check output schema, refusal behaviour on out-of-scope questions, and
grounding against the retrieved passages.
