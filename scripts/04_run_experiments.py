"""Run the ablation grid, then add cross-encoder reranking on the winner."""
import argparse

import _bootstrap  # noqa: F401

from ragbench.config import RunSpec, default_grid, require_openai_key
from ragbench.evaluate import evaluate_spec
from ragbench.testset import load

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    args = ap.parse_args()

    require_openai_key()
    testset = load()
    if args.limit:
        testset = testset[: args.limit]
    print(f"{len(testset)} gold questions\n")

    grid = default_grid()
    print(f"Stage 1: {len(grid)} chunker x retrieval configurations")
    results = [evaluate_spec(s, testset, skip_existing=not args.force) for s in grid]

    if not args.no_rerank:
        def key(p):
            a = p["aggregate"]
            return (
                a.get("faithfulness", 0)
                + a.get("answer_relevancy", 0)
                + a.get("llm_context_precision_with_reference", 0)
                + a.get("context_recall", 0)
            )

        best = max(results, key=key)
        spec = RunSpec(**best["spec"])
        print(f"\nStage 2: cross-encoder rerank on winner ({spec.run_id})")
        evaluate_spec(
            RunSpec(chunker=spec.chunker, mode=spec.mode, rerank=True),
            testset,
            skip_existing=not args.force,
        )

    print("\nDone. Now run: python scripts/05_report.py")
