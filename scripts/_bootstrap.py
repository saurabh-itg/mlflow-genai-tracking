"""Shared setup: path wiring + tracking URI + experiment naming."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlruns.db'}")
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", str(ROOT / "mlartifacts"))
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "genai-rag-observability")
CORPUS = str(ROOT / "data" / "corpus")
EVAL_SET = ROOT / "data" / "eval_set.jsonl"


def init(experiment: str = EXPERIMENT) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)
    return experiment


def load_eval_set() -> list[dict]:
    import json

    return [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]
