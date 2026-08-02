"""Screenshot the running MLflow UI into docs/screenshots/.

Assumes `make ui` is already serving on http://127.0.0.1:5000. Every image in the
README was produced by this script against the runs created by scripts 01-04, so the
docs cannot drift from the code without someone noticing.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

BASE = os.getenv("MLFLOW_UI", "http://127.0.0.1:5000")
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
VIEWPORT = {"width": 1680, "height": 1050}


def api(path: str, payload: dict) -> dict:
    r = requests.post(f"{BASE}/api/2.0/mlflow/{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def discover() -> dict:
    exps = {e["name"]: e["experiment_id"] for e in api("experiments/search", {"max_results": 50})["experiments"]}
    main_id = exps["genai-rag-observability"]
    runs = api("runs/search", {"experiment_ids": [main_id], "max_results": 200,
                               "order_by": ["metrics.composite_score DESC"]})["runs"]
    best = runs[0]["info"]["run_id"]
    worst = runs[-1]["info"]["run_id"]

    traces = requests.get(f"{BASE}/api/3.0/mlflow/traces",
                          params={"locations": main_id, "max_results": 5}, timeout=30)
    trace_id = ""
    if traces.ok:
        items = traces.json().get("traces", [])
        if items:
            trace_id = items[0].get("trace_id") or items[0].get("request_id", "")
    return {"exp": main_id, "regression_exp": exps.get("genai-rag-regression", ""),
            "best": best, "worst": worst, "trace": trace_id,
            "compare": [r["info"]["run_id"] for r in runs[:6]]}


SHOTS = lambda c: [
    ("01-experiment-runs-table",
     f"{BASE}/#/experiments/{c['exp']}",
     "Run table: 24 sweep configurations, one row each"),
    ("02-run-comparison-charts",
     f"{BASE}/#/experiments/{c['exp']}?compareRunsMode=CHART",
     "Metric charts across the sweep"),
    ("03-traces-list",
     f"{BASE}/#/experiments/{c['exp']}/traces",
     "Trace list: one trace per question per config"),
    ("04-trace-waterfall",
     f"{BASE}/#/experiments/{c['exp']}/traces?selectedTraceId={c['trace']}" if c["trace"]
     else f"{BASE}/#/experiments/{c['exp']}/traces",
     "Span waterfall: retrieve -> build_prompt -> generate"),
    ("05-best-run-detail",
     f"{BASE}/#/experiments/{c['exp']}/runs/{c['best']}",
     "Champion run: params, metrics, artifacts"),
    ("06-run-comparison-table",
     f"{BASE}/#/compare-runs?runs=" + str(c["compare"]).replace("'", '"').replace(" ", "") + f"&experiments=[\"{c['exp']}\"]",
     "Side-by-side comparison of the top configurations"),
    ("07-model-registry",
     f"{BASE}/#/models/genai-rag-industrial-qa",
     "Registered model with the champion alias"),
    ("08-prompt-registry",
     f"{BASE}/#/prompts",
     "Versioned prompt templates"),
    ("09-regression-gate",
     f"{BASE}/#/experiments/{c['regression_exp']}",
     "Regression gate run with pass/fail metrics"),
]


def _settle(page, timeout_ms: int = 25_000) -> None:
    """Wait until MLflow's React shell has swapped its skeleton loaders for real content."""
    try:
        page.wait_for_function(
            "() => document.querySelectorAll('[class*=skeleton i],[class*=Skeleton]').length === 0",
            timeout=timeout_ms,
        )
    except Exception:
        pass
    page.wait_for_timeout(3500)


def main():
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    ctx = discover()
    print("discovered:", {k: (v[:8] if isinstance(v, str) else v) for k, v in ctx.items()})
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        for name, url, caption in SHOTS(ctx):
            if only and not any(name.startswith(o) for o in only):
                continue
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            _settle(page)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(f"  captured {name}.png  <- {caption}")
        browser.close()
    print(f"\nwrote {len(list(OUT.glob('*.png')))} screenshots to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
