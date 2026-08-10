"""Cheapest possible check that the judge actually works before spending on a grid.

Runs ONE question through ONE retrieval configuration and scores it with every
metric, reporting which produced a real number and which produced NaN. Roughly
20 judge calls, i.e. cents.

Why this exists: with the LangChain judge backend, answer_relevancy returned NaN
on 100% of samples because Claude's free-text JSON did not parse. That was
invisible in the aggregates until someone looked at per-sample scores. This
makes it visible for the price of one question.
"""
import argparse

import _bootstrap  # noqa: F401

from ragbench.config import RunSpec, require_api_key
from ragbench.evaluate import METRIC_SETS, collect_predictions, judge_components, score
from ragbench.pipeline import RagPipeline
from ragbench.providers import describe
from ragbench.retrieval import build_retriever
from ragbench.testset import load

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="structural_article")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    require_api_key()
    print(f"providers: {describe()}")
    import os

    print(f"judge backend: {os.getenv('RAGBENCH_JUDGE_BACKEND', 'instructor')}\n")

    testset = load()[: args.n]
    spec = RunSpec(args.chunker, args.mode)
    retriever = build_retriever(spec.chunker, mode=spec.mode, top_k=spec.top_k)
    rows = collect_predictions(RagPipeline(retriever), testset, verbose=False)

    print(f"question : {rows[0]['user_input'][:90]}")
    print(f"answer   : {rows[0]['response'][:90]}")
    print(f"contexts : {len(rows[0]['retrieved_contexts'])} retrieved\n")

    payload = score(rows, "diagnostic", metric_set="full")

    print("\nmetric                                    result")
    print("-" * 56)
    ok = broken = 0
    for name, val in payload["aggregate"].items():
        if val is None or val != val:
            print(f"  {name:<40} NaN   <-- BROKEN")
            broken += 1
        else:
            print(f"  {name:<40} {val:.3f}")
            ok += 1
    print(f"\n{ok} metrics working, {broken} returning NaN.")
    if broken:
        print("Do NOT run the full grid until this is 0 - you would pay for NaN columns.")
        print("Try: RAGBENCH_JUDGE_BACKEND=langchain python scripts/00_diagnose_metrics.py")
    else:
        print("All metrics produce real values. Safe to proceed.")
