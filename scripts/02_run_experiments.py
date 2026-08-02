"""Sweep RAG configurations and log one MLflow run per configuration.

Each run carries:
  params   - chunk_size, top_k, temperature, prompt_version, model
  metrics  - quality (groundedness, relevance, correctness), retrieval (hit rate, MRR),
             cost (usd per query, tokens), latency (p50/p95 ms)
  traces   - one span tree per eval question (retrieve -> build_prompt -> generate)
  artifact - the full per-question result table

That triple of quality/cost/latency on every run is the whole point: it makes a
"better" configuration argue for itself against what it costs.
"""
import itertools
import json
import statistics as stats

import _bootstrap
import mlflow

from genai_app import scorers
from genai_app.rag import RagConfig, RagPipeline

_bootstrap.init()
EVAL = _bootstrap.load_eval_set()

GRID = {
    "chunk_size": [80, 160],
    "top_k": [2, 4],
    "prompt_version": ["v1-terse", "v2-grounded", "v3-grounded-cot"],
    "temperature": [0.0, 0.6],
}


def evaluate_config(cfg: RagConfig) -> dict:
    pipe = RagPipeline(cfg)
    rows = []
    for item in EVAL:
        res = pipe.answer(item["question"])
        ctx = res.context_blob
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": res.answer,
                "retrieved_doc_ids": ",".join(res.retrieved_doc_ids),
                "groundedness": scorers.groundedness(res.answer, ctx),
                "answer_relevance": scorers.answer_relevance(res.answer, item["question"]),
                "correctness": scorers.correctness(res.answer, item["expected_facts"]),
                "hit_rate": scorers.retrieval_hit_rate(res.retrieved_doc_ids, item["relevant_doc_id"]),
                "mrr": scorers.retrieval_mrr(res.retrieved_doc_ids, item["relevant_doc_id"]),
                "refusal": scorers.refusal(res.answer),
                "latency_ms": res.total_latency_ms,
                "total_tokens": res.llm.total_tokens,
                "cost_usd": res.cost_usd,
                "trace_id": res.trace_id or "",
            }
        )
    return rows


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["id"] != "q15"]
    lat = sorted(r["latency_ms"] for r in rows)
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
    return {
        "groundedness": round(stats.mean(r["groundedness"] for r in answerable), 4),
        "answer_relevance": round(stats.mean(r["answer_relevance"] for r in answerable), 4),
        "correctness": round(stats.mean(r["correctness"] for r in answerable), 4),
        "retrieval_hit_rate": round(stats.mean(r["hit_rate"] for r in answerable), 4),
        "retrieval_mrr": round(stats.mean(r["mrr"] for r in answerable), 4),
        "refusal_rate": round(stats.mean(r["refusal"] for r in rows), 4),
        "latency_p50_ms": round(stats.median(lat), 2),
        "latency_p95_ms": round(p95, 2),
        "avg_total_tokens": round(stats.mean(r["total_tokens"] for r in rows), 2),
        "cost_usd_per_1k_queries": round(stats.mean(r["cost_usd"] for r in rows) * 1000, 6),
    }


def main():
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"running {len(combos)} configurations x {len(EVAL)} questions")

    best = (None, -1.0)
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        cfg = RagConfig(corpus_dir=_bootstrap.CORPUS, **params)
        name = f"k{params['top_k']}-c{params['chunk_size']}-{params['prompt_version']}-t{params['temperature']}"

        with mlflow.start_run(run_name=name) as run:
            rows = evaluate_config(cfg)
            agg = aggregate(rows)

            mlflow.log_params(cfg.as_params())
            mlflow.log_param("llm_model", RagPipeline(cfg).llm.name)
            mlflow.log_metrics(agg)
            mlflow.set_tags(
                {
                    "pipeline": "rag-qa",
                    "eval_set": "industrial-15q",
                    "sweep": "grid-v1",
                    "provider": "deterministic",
                }
            )
            mlflow.log_table({k: [r[k] for r in rows] for k in rows[0]}, "eval_results.json")
            mlflow.log_dict({"config": cfg.as_params(), "aggregate": agg}, "summary.json")

            # composite score: quality that has to pay for its latency
            score = agg["correctness"] * 0.5 + agg["groundedness"] * 0.3 + agg["retrieval_mrr"] * 0.2
            mlflow.log_metric("composite_score", round(score, 4))
            if score > best[1]:
                best = (run.info.run_id, score)
            print(f"  [{i:2d}/{len(combos)}] {name:44s} score={score:.4f} p95={agg['latency_p95_ms']}ms")

    print(f"\nbest run: {best[0]}  composite_score={best[1]:.4f}")
    (_bootstrap.ROOT / "best_run_id.txt").write_text(best[0])


if __name__ == "__main__":
    main()
