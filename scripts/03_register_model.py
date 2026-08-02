"""Wrap the winning configuration as an MLflow pyfunc model and register it.

The registered model is what a serving layer loads. Aliases ("champion", "challenger")
are how you promote without changing the calling code.
"""
import _bootstrap
import mlflow
import pandas as pd
from mlflow.models import infer_signature

from genai_app.rag import RagConfig, RagPipeline

_bootstrap.init()
MODEL_NAME = "genai-rag-industrial-qa"


class RagModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import json
        from pathlib import Path

        cfg = json.loads(Path(context.artifacts["config"]).read_text())
        cfg["corpus_dir"] = context.artifacts["corpus"]
        self._pipe = RagPipeline(RagConfig(**cfg))

    def predict(self, context, model_input, params=None):
        questions = model_input["question"].tolist()
        return pd.DataFrame(
            [
                {
                    "answer": r.answer,
                    "sources": ",".join(sorted(set(r.retrieved_doc_ids))),
                    "latency_ms": r.total_latency_ms,
                    "cost_usd": r.cost_usd,
                }
                for r in (self._pipe.answer(q) for q in questions)
            ]
        )


def main():
    import json

    best_file = _bootstrap.ROOT / "best_run_id.txt"
    if not best_file.exists():
        raise SystemExit("run scripts/02_run_experiments.py first")
    best_run_id = best_file.read_text().strip()

    client = mlflow.MlflowClient()
    best = client.get_run(best_run_id)
    cfg = {
        "chunk_size": int(best.data.params["chunk_size"]),
        "chunk_overlap": int(best.data.params["chunk_overlap"]),
        "top_k": int(best.data.params["top_k"]),
        "temperature": float(best.data.params["temperature"]),
        "prompt_version": best.data.params["prompt_version"],
    }

    cfg_path = _bootstrap.ROOT / "champion_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    example = pd.DataFrame({"question": ["What VRAM footprint does the TinyLlama fine-tune fit into?"]})
    pipe = RagPipeline(RagConfig(corpus_dir=_bootstrap.CORPUS, **cfg))
    preview = pd.DataFrame([{"answer": pipe.answer(example.iloc[0, 0]).answer,
                             "sources": "edge-finetuning", "latency_ms": 1.0, "cost_usd": 0.0}])

    with mlflow.start_run(run_name="register-champion") as run:
        mlflow.log_params(cfg)
        mlflow.set_tags({"promoted_from_run": best_run_id, "stage": "registry"})
        kwargs = dict(
            python_model=RagModel(),
            artifacts={"config": str(cfg_path), "corpus": _bootstrap.CORPUS},
            signature=infer_signature(example, preview),
            input_example=example,
            registered_model_name=MODEL_NAME,
            code_paths=[str(_bootstrap.ROOT / "src" / "genai_app")],
        )
        try:
            info = mlflow.pyfunc.log_model(name="rag_model", **kwargs)
        except TypeError:  # MLflow < 3
            info = mlflow.pyfunc.log_model(artifact_path="rag_model", **kwargs)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    client.set_model_version_tag(MODEL_NAME, latest.version, "promoted_from_run", best_run_id)
    print(f"registered {MODEL_NAME} v{latest.version} -> alias 'champion'  ({info.model_uri})")


if __name__ == "__main__":
    main()
