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


def by_category() -> pd.DataFrame | None:
    """Score breakdown per question category.

    This is where the interesting claims live: hybrid retrieval should help most
    on `lexical` questions (exact statutory tokens), structural chunking most on
    `multi_hop` (severed cross-references), and `negative` questions test whether
    the system abstains rather than inventing an answer.
    """
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for s in d.get("samples", []):
            r = {"run_id": d["run_id"], "category": s.get("_category", "generated")}
            for k, v in (s.get("_scores") or {}).items():
                label = METRIC_LABELS.get(k)
                if label:
                    r[label] = v
            rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    metric_cols = [c for c in dict.fromkeys(METRIC_LABELS.values()) if c in df.columns]
    return df.groupby(["run_id", "category"])[metric_cols].mean().round(3).reset_index()


def build_report() -> str:
    df = load_runs()

    # Runs on different gold-set sizes are NOT comparable - different questions,
    # different difficulty. Group them rather than ranking them against each other.
    if df["n"].nunique() > 1:
        groups = sorted(df["n"].unique(), reverse=True)
        parts = ["# Results", "",
                 "Runs are grouped by gold-set size. **Scores from different groups are "
                 "not comparable** - they were measured on different questions.", ""]
        for n in groups:
            sub = df[df["n"] == n].copy()
            sub["Composite"] = composite(sub)
            sub = sub.sort_values("Composite", ascending=False)
            cols = [c for c in dict.fromkeys(METRIC_LABELS.values()) if c in sub.columns]
            parts += [f"## Gold set: {n} questions", "",
                      sub[["chunker", "retrieval", "rerank"] + cols].to_markdown(
                          index=False, floatfmt=".3f"), ""]
        df.to_csv(RESULTS / "summary.csv", index=False)
        cat = by_category()
        if cat is not None and not cat.empty:
            cat.to_csv(RESULTS / "by_category.csv", index=False)
            parts += ["## Per-category breakdown", "",
                      cat.to_markdown(index=False, floatfmt=".3f"), ""]
        parts += ["See `docs/EVALUATION.md` for the analysis, caveats and cost breakdown.", ""]
        return "\n".join(parts)

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

    cat = by_category()
    if cat is not None and not cat.empty:
        best_id = str(best["run_id"]) if "run_id" in best else None
        sub = cat[cat["run_id"] == best_id] if best_id else cat
        if not sub.empty:
            lines += ["## Best configuration, broken down by question type", "",
                      sub.drop(columns=["run_id"]).to_markdown(index=False, floatfmt=".3f"), ""]
        cat.to_csv(RESULTS / "by_category.csv", index=False)
        lines += ["Full per-category data: `results/by_category.csv`.", ""]

    return "\n".join(lines)


def write_report() -> str:
    text = build_report()
    (RESULTS / "RESULTS.md").write_text(text, encoding="utf-8")
    return text
