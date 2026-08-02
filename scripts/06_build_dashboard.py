"""Render a shareable static dashboard from the MLflow tracking store.

The MLflow UI is the interactive tool; this is the artifact you paste into a PR or a
weekly report for people who don't have access to the tracking server. Everything here
is read back out of MLflow -- no numbers are hardcoded.

Output: docs/dashboard.html
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import _bootstrap
import mlflow

_bootstrap.init()
OUT = Path(__file__).resolve().parents[1] / "docs" / "dashboard.html"
client = mlflow.MlflowClient()

CSS = """
:root{--bg:#0b0d10;--card:#151a21;--soft:#12151a;--line:#232a34;--fg:#e8ecf1;--dim:#9aa6b4;
--faint:#6b7684;--accent:#5eead4;--accent2:#2dd4bf;--amber:#fbbf24;--red:#f87171;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:14px;line-height:1.6}
.wrap{max-width:1240px;margin:0 auto;padding:40px 32px 64px}
h1{font-size:26px;letter-spacing:-.02em;margin-bottom:4px}
.sub{color:var(--dim);font-family:var(--mono);font-size:12.5px;margin-bottom:32px}
h2{font-size:12px;font-family:var(--mono);color:var(--accent);text-transform:uppercase;
letter-spacing:.10em;margin:38px 0 14px;font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.kpi .v{font-size:24px;font-weight:650;letter-spacing:-.02em;font-family:var(--mono)}
.kpi .l{color:var(--faint);font-size:11px;font-family:var(--mono);text-transform:uppercase;
letter-spacing:.06em;margin-top:4px}
.kpi .d{color:var(--dim);font-size:11.5px;margin-top:6px}
.grid2{display:grid;grid-template-columns:1.35fr 1fr;gap:16px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px}
th{text-align:left;color:var(--faint);font-weight:500;text-transform:uppercase;
letter-spacing:.06em;font-size:10px;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:5px 8px;border-bottom:1px solid #1a2029;white-space:nowrap}
tr:hover td{background:#1a2029}
.best td{background:rgba(94,234,212,.07)}
.num{text-align:right}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:10px;
border:1px solid var(--line);color:var(--dim)}
.pass{color:var(--accent);border-color:rgba(94,234,212,.35);background:rgba(94,234,212,.08)}
.fail{color:var(--red);border-color:rgba(248,113,113,.35);background:rgba(248,113,113,.08)}
.bar{height:9px;border-radius:4px;background:linear-gradient(90deg,var(--accent2),var(--accent))}
.legend{color:var(--faint);font-size:11px;font-family:var(--mono);margin-top:10px}
.span-row{display:grid;grid-template-columns:150px 1fr 76px;align-items:center;gap:10px;margin:7px 0}
.span-name{font-family:var(--mono);font-size:11.5px;color:var(--dim)}
.span-track{background:#12151a;border:1px solid var(--line);border-radius:5px;height:20px;position:relative}
.span-fill{position:absolute;top:2px;bottom:2px;border-radius:4px}
.span-ms{font-family:var(--mono);font-size:11px;color:var(--faint);text-align:right}
.foot{color:var(--faint);font-size:11.5px;font-family:var(--mono);margin-top:40px;
border-top:1px solid var(--line);padding-top:16px}
"""

SPAN_COLOR = {"CHAIN": "#5eead4", "RETRIEVER": "#fbbf24", "PARSER": "#8a63bf", "LLM": "#2dd4bf"}


def esc(x) -> str:
    return html.escape(str(x))


def collect():
    exp = client.get_experiment_by_name("genai-rag-observability")
    runs = [r for r in client.search_runs([exp.experiment_id], max_results=500)
            if "composite_score" in r.data.metrics]
    runs.sort(key=lambda r: -r.data.metrics["composite_score"])

    reg_exp = client.get_experiment_by_name("genai-rag-regression")
    reg = client.search_runs([reg_exp.experiment_id], max_results=1,
                             order_by=["attributes.start_time DESC"]) if reg_exp else []

    traces = client.search_traces(locations=[exp.experiment_id], max_results=1)
    spans = []
    if traces:
        t = traces[0]
        t0 = min(s.start_time_ns for s in t.data.spans)
        for s in t.data.spans:
            spans.append({
                "name": s.name,
                "type": str(s.span_type),
                "offset_ms": (s.start_time_ns - t0) / 1e6,
                "dur_ms": (s.end_time_ns - s.start_time_ns) / 1e6,
            })
        spans.sort(key=lambda s: s["offset_ms"])

    models = []
    try:
        rm = client.get_registered_model("genai-rag-industrial-qa")
        alias_by_version: dict[str, list[str]] = {}
        for alias, version in (rm.aliases or {}).items():
            alias_by_version.setdefault(str(version), []).append(alias)
    except Exception:
        alias_by_version = {}
    for mv in client.search_model_versions("name='genai-rag-industrial-qa'"):
        models.append({"version": mv.version,
                       "aliases": alias_by_version.get(str(mv.version), list(mv.aliases or [])),
                       "run_id": mv.run_id[:12]})
    return runs, (reg[0] if reg else None), spans, models


def scatter(runs) -> str:
    W, H, P = 560, 260, 44
    xs = [r.data.metrics["latency_p95_ms"] for r in runs]
    ys = [r.data.metrics["composite_score"] for r in runs]
    x0, x1 = min(xs) * 0.98, max(xs) * 1.02
    y0, y1 = min(ys) * 0.98, max(ys) * 1.01
    sx = lambda v: P + (v - x0) / (x1 - x0 or 1) * (W - P - 16)
    sy = lambda v: H - P - (v - y0) / (y1 - y0 or 1) * (H - P - 18)
    pts = []
    for i, r in enumerate(runs):
        c = "#5eead4" if i == 0 else "#3b7a72"
        rad = 6 if i == 0 else 4
        pts.append(f'<circle cx="{sx(r.data.metrics["latency_p95_ms"]):.1f}" '
                   f'cy="{sy(r.data.metrics["composite_score"]):.1f}" r="{rad}" fill="{c}" '
                   f'fill-opacity="{0.95 if i == 0 else 0.6}"/>')
    grid = "".join(
        f'<line x1="{P}" y1="{sy(y0 + (y1 - y0) * f):.1f}" x2="{W-16}" y2="{sy(y0+(y1-y0)*f):.1f}" '
        f'stroke="#232a34" stroke-width="1"/>'
        f'<text x="{P-8}" y="{sy(y0+(y1-y0)*f)+4:.1f}" fill="#6b7684" font-size="9" '
        f'font-family="monospace" text-anchor="end">{y0+(y1-y0)*f:.2f}</text>'
        for f in (0, .25, .5, .75, 1))
    xlab = "".join(
        f'<text x="{sx(x0+(x1-x0)*f):.1f}" y="{H-P+18}" fill="#6b7684" font-size="9" '
        f'font-family="monospace" text-anchor="middle">{x0+(x1-x0)*f:.0f}</text>'
        for f in (0, .5, 1))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%">{grid}{"".join(pts)}{xlab}'
            f'<text x="{W//2}" y="{H-6}" fill="#6b7684" font-size="10" font-family="monospace" '
            f'text-anchor="middle">p95 latency (ms) →</text></svg>')


def effect_panel(runs, param: str) -> str:
    groups: dict[str, list[float]] = {}
    for r in runs:
        groups.setdefault(r.data.params[param], []).append(r.data.metrics["composite_score"])
    rows = []
    hi = max(sum(v) / len(v) for v in groups.values())
    for k, v in sorted(groups.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        m = sum(v) / len(v)
        rows.append(
            f'<div style="display:grid;grid-template-columns:130px 1fr 48px;gap:10px;'
            f'align-items:center;margin:6px 0"><span class="span-name">{esc(k)}</span>'
            f'<div style="background:#12151a;border-radius:4px"><div class="bar" '
            f'style="width:{m/hi*100:.1f}%"></div></div>'
            f'<span class="span-ms">{m:.3f}</span></div>')
    return "".join(rows)


def main():
    runs, reg, spans, models = collect()
    best = runs[0]
    m = best.data.metrics

    kpis = [
        (f'{m["composite_score"]:.3f}', "composite score", "0.5·correct + 0.3·ground + 0.2·mrr"),
        (f'{m["groundedness"]:.3f}', "groundedness", "answer terms present in context"),
        (f'{m["correctness"]:.3f}', "correctness", "required facts recovered"),
        (f'{m["latency_p95_ms"]:.0f} ms', "p95 latency", "end to end, per query"),
        (f'{len(runs)}', "configurations", "swept in one experiment"),
    ]
    kpi_html = "".join(
        f'<div class="card kpi"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div>'
        f'<div class="d">{esc(d)}</div></div>' for v, l, d in kpis)

    hdr = ["run", "prompt", "top_k", "chunk", "temp", "composite", "ground", "correct",
           "hit@k", "mrr", "p95 ms", "tokens"]
    body = []
    for i, r in enumerate(runs):
        p, d = r.data.params, r.data.metrics
        body.append(
            f'<tr class="{"best" if i == 0 else ""}"><td>{esc(r.info.run_name)}</td>'
            f'<td>{esc(p["prompt_version"])}</td><td class="num">{esc(p["top_k"])}</td>'
            f'<td class="num">{esc(p["chunk_size"])}</td><td class="num">{esc(p["temperature"])}</td>'
            f'<td class="num" style="color:var(--accent)">{d["composite_score"]:.4f}</td>'
            f'<td class="num">{d["groundedness"]:.3f}</td><td class="num">{d["correctness"]:.3f}</td>'
            f'<td class="num">{d["retrieval_hit_rate"]:.2f}</td><td class="num">{d["retrieval_mrr"]:.3f}</td>'
            f'<td class="num">{d["latency_p95_ms"]:.1f}</td>'
            f'<td class="num">{d["avg_total_tokens"]:.0f}</td></tr>')

    total = max((s["offset_ms"] + s["dur_ms"]) for s in spans) if spans else 1
    span_html = "".join(
        f'<div class="span-row"><span class="span-name">{esc(s["name"])} '
        f'<span class="pill">{esc(s["type"])}</span></span>'
        f'<div class="span-track"><div class="span-fill" style="left:{s["offset_ms"]/total*100:.2f}%;'
        f'width:{max(s["dur_ms"]/total*100,0.6):.2f}%;background:{SPAN_COLOR.get(s["type"],"#5eead4")}">'
        f'</div></div><span class="span-ms">{s["dur_ms"]:.2f}</span></div>' for s in spans)

    gate_html = ""
    if reg:
        thresholds = {"groundedness": ("min", 0.55), "correctness": ("min", 0.45),
                      "retrieval_hit_rate": ("min", 0.70), "latency_p95_ms": ("max", 750.0),
                      "cost_usd_per_1k_queries": ("max", 5.0)}
        rows = []
        for k, (direction, bound) in thresholds.items():
            v = reg.data.metrics.get(k, 0.0)
            ok = v >= bound if direction == "min" else v <= bound
            rows.append(f'<tr><td>{esc(k)}</td><td class="num">{v:.4g}</td>'
                        f'<td class="num">{direction} {bound}</td>'
                        f'<td><span class="pill {"pass" if ok else "fail"}">'
                        f'{"PASS" if ok else "FAIL"}</span></td></tr>')
        gate_html = ('<table><tr><th>metric</th><th class="num">observed</th>'
                     '<th class="num">threshold</th><th>gate</th></tr>' + "".join(rows) + "</table>")

    def _aliases(mv):
        return "".join('<span class="pill pass">%s</span> ' % esc(a) for a in mv["aliases"]) or "-"

    model_html = "".join(
        '<tr><td>genai-rag-industrial-qa</td><td class="num">v%s</td><td>%s</td><td>%s</td></tr>'
        % (esc(mv["version"]), _aliases(mv), esc(mv["run_id"])) for mv in models)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>GenAI RAG - MLflow tracking dashboard</title><style>{CSS}</style></head><body><div class="wrap">
<h1>GenAI RAG &mdash; MLflow tracking dashboard</h1>
<div class="sub">generated by scripts/06_build_dashboard.py &middot; read directly from the MLflow tracking store &middot; experiment <b>genai-rag-observability</b></div>

<h2>Champion configuration</h2>
<div class="kpis">{kpi_html}</div>

<h2>Quality vs latency &mdash; every configuration in the sweep</h2>
<div class="grid2">
  <div class="card">{scatter(runs)}
    <div class="legend">each dot is one run; the bright dot is the champion. A config that buys
    quality with latency has to justify itself here.</div></div>
  <div class="card">
    <div class="span-name" style="margin-bottom:10px">mean composite score by prompt version</div>
    {effect_panel(runs, "prompt_version")}
    <div class="span-name" style="margin:16px 0 10px">by top_k</div>
    {effect_panel(runs, "top_k")}
    <div class="span-name" style="margin:16px 0 10px">by chunk_size</div>
    {effect_panel(runs, "chunk_size")}
  </div>
</div>

<h2>All runs</h2>
<div class="card"><table><tr>{"".join(f'<th class="{"num" if h not in ("run","prompt") else ""}">{esc(h)}</th>' for h in hdr)}</tr>{"".join(body)}</table></div>

<h2>Trace waterfall &mdash; one request</h2>
<div class="grid2">
  <div class="card">{span_html}
    <div class="legend">span tree captured by @mlflow.trace; total {total:.2f} ms.
    Generation dominates, retrieval is sub-millisecond &mdash; the shape you want before you
    start optimising the wrong stage.</div></div>
  <div class="card">
    <div class="span-name" style="margin-bottom:10px">regression gate (latest)</div>
    {gate_html}
    <div class="span-name" style="margin:18px 0 10px">model registry</div>
    <table><tr><th>model</th><th class="num">version</th><th>alias</th><th>run</th></tr>{model_html}</table>
  </div>
</div>

<div class="foot">MLflow {esc(mlflow.__version__)} &middot; {len(runs)} runs &middot;
tracking store {esc(_bootstrap.TRACKING_URI)}</div>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT}  ({len(doc)//1024} KB, {len(runs)} runs)")


if __name__ == "__main__":
    main()
