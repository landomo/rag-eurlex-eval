"""Turn the per-run JSON blobs into results/RESULTS.md + results/summary.csv."""
from __future__ import annotations

import json

import pandas as pd

from .config import RESULTS, RUNS

# Ragas column name -> the label used in the report.
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_recall": "Context Recall",
    "factual_correctness(mode=f1)": "Factual Correctness",
    "factual_correctness": "Factual Correctness",
    "noise_sensitivity(mode=relevant)": "Noise Sensitivity",
    "noise_sensitivity_relevant": "Noise Sensitivity",
}
# Lower is better for this one.
LOWER_IS_BETTER = {"Noise Sensitivity"}


def load_runs() -> pd.DataFrame:
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        spec = d.get("spec", {})
        row = {
            "run_id": d["run_id"],
            "chunker": spec.get("chunker", d["run_id"].split("__")[0]),
            "retrieval": spec.get("mode", "?"),
            "rerank": bool(spec.get("rerank", "rerank" in d["run_id"])),
            "n": d.get("n_samples"),
            "seconds": d.get("seconds"),
        }
        for k, v in d["aggregate"].items():
            label = METRIC_LABELS.get(k)
            if label:
                row[label] = v
        rows.append(row)
    if not rows:
        raise SystemExit("No runs in results/runs/. Run: python scripts/04_run_experiments.py")
    return pd.DataFrame(rows)


def composite(df: pd.DataFrame) -> pd.Series:
    """Single ranking number: mean of the four core metrics, noise inverted.

    Not a claim that these weights are principled - it just gives the ablation a
    defensible ordering. The per-metric columns are what you should actually read.
    """
    cols = [c for c in ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"] if c in df]
    s = df[cols].mean(axis=1)
    if "Noise Sensitivity" in df:
        s = (s * len(cols) + (1 - df["Noise Sensitivity"])) / (len(cols) + 1)
    return s.round(4)


def build_report() -> str:
    df = load_runs()
    df["Composite"] = composite(df)
    df = df.sort_values("Composite", ascending=False).reset_index(drop=True)
    df.to_csv(RESULTS / "summary.csv", index=False)

    metric_cols = [c for c in METRIC_LABELS.values() if c in df.columns]
    metric_cols = list(dict.fromkeys(metric_cols))
    show = df[["chunker", "retrieval", "rerank"] + metric_cols + ["Composite"]]

    best = df.iloc[0]
    lines = [
        "# Results",
        "",
        f"`{len(df)}` configurations evaluated on `{int(df['n'].iloc[0])}` gold questions "
        f"over the EU AI Act + GDPR corpus.",
        "",
        "## Ablation table",
        "",
        show.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Noise Sensitivity: lower is better. All other metrics: higher is better.",
        "",
        "## Best configuration",
        "",
        f"**{best['chunker']} + {best['retrieval']}"
        f"{' + cross-encoder rerank' if best['rerank'] else ''}**",
        "",
    ]
    for c in metric_cols:
        lines.append(f"- {c}: **{best[c]:.3f}**")
    lines += ["", "## Deltas worth reading", ""]

    # chunker effect, holding retrieval mode fixed at the best one
    by_chunker = df.groupby("chunker")["Composite"].max().sort_values(ascending=False)
    lines.append("Best composite per chunking strategy:")
    lines.append("")
    for k, v in by_chunker.items():
        lines.append(f"- `{k}`: {v:.3f}")
    lines.append("")

    by_mode = df[~df["rerank"]].groupby("retrieval")["Composite"].mean()
    if len(by_mode) == 2:
        delta = by_mode.get("hybrid", 0) - by_mode.get("dense", 0)
        lines.append(
            f"Hybrid vs dense, averaged across chunkers: **{delta:+.3f}** composite."
        )
        lines.append("")

    rr = df[df["rerank"]]
    if len(rr):
        for _, r in rr.iterrows():
            baseline = df[
                (df["chunker"] == r["chunker"]) & (df["retrieval"] == r["retrieval"]) & (~df["rerank"])
            ]
            if len(baseline):
                b = baseline.iloc[0]
                lines.append(
                    f"Cross-encoder rerank on `{r['chunker']} + {r['retrieval']}`: "
                    f"Context Precision {b.get('Context Precision', float('nan')):.3f} -> "
                    f"{r.get('Context Precision', float('nan')):.3f}, "
                    f"Composite {b['Composite']:.3f} -> {r['Composite']:.3f}."
                )
        lines.append("")

    return "\n".join(lines)


def write_report() -> str:
    text = build_report()
    (RESULTS / "RESULTS.md").write_text(text, encoding="utf-8")
    return text
