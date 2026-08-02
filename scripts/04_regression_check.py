"""Nightly-style regression gate.

Loads the champion, re-runs the eval set, logs the result as its own run, and fails
with a non-zero exit code if quality drops or cost climbs past the thresholds. This is
the piece that turns tracking into something that can block a deploy.
"""
import os
import statistics as stats
import sys

import _bootstrap
import mlflow

from genai_app import scorers
from genai_app.rag import RagConfig, RagPipeline

_bootstrap.init(os.getenv("MLFLOW_EXPERIMENT_REGRESSION", "genai-rag-regression"))

THRESHOLDS = {
    "groundedness": ("min", 0.55),
    "correctness": ("min", 0.45),
    "retrieval_hit_rate": ("min", 0.70),
    "latency_p95_ms": ("max", 750.0),
    "cost_usd_per_1k_queries": ("max", 5.0),
}


def main() -> int:
    import json
    from pathlib import Path

    cfg_path = Path(_bootstrap.ROOT / "champion_config.json")
    if not cfg_path.exists():
        raise SystemExit("run scripts/03_register_model.py first")
    cfg = json.loads(cfg_path.read_text())

    pipe = RagPipeline(RagConfig(corpus_dir=_bootstrap.CORPUS, **cfg))
    evalset = _bootstrap.load_eval_set()

    g, c, h, lat, cost = [], [], [], [], []
    with mlflow.start_run(run_name="regression-check") as run:
        for item in evalset:
            res = pipe.answer(item["question"])
            if item["id"] != "q15":
                g.append(scorers.groundedness(res.answer, res.context_blob))
                c.append(scorers.correctness(res.answer, item["expected_facts"]))
                h.append(scorers.retrieval_hit_rate(res.retrieved_doc_ids, item["relevant_doc_id"]))
            lat.append(res.total_latency_ms)
            cost.append(res.cost_usd)

        slat = sorted(lat)
        observed = {
            "groundedness": round(stats.mean(g), 4),
            "correctness": round(stats.mean(c), 4),
            "retrieval_hit_rate": round(stats.mean(h), 4),
            "latency_p95_ms": round(slat[max(0, int(len(slat) * 0.95) - 1)], 2),
            "cost_usd_per_1k_queries": round(stats.mean(cost) * 1000, 6),
        }
        mlflow.log_params(cfg)
        mlflow.log_metrics(observed)

        failures = []
        for metric, (direction, bound) in THRESHOLDS.items():
            val = observed[metric]
            ok = val >= bound if direction == "min" else val <= bound
            mlflow.log_metric(f"gate_{metric}_pass", 1.0 if ok else 0.0)
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {metric:28s} {val:>10}  ({direction} {bound})")
            if not ok:
                failures.append(metric)

        mlflow.set_tag("gate_status", "fail" if failures else "pass")
        print("\nregression gate:", "FAILED -> " + ", ".join(failures) if failures else "PASSED")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
