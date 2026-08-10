"""Run the ablation grid, then add cross-encoder reranking on the winner."""
import argparse

import _bootstrap  # noqa: F401

from ragbench.config import RunSpec, default_grid, require_api_key
from ragbench.evaluate import evaluate_spec
from ragbench.testset import load

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    ap.add_argument("--metrics", default="core", choices=["core", "standard", "full"],
                    help="core: faithfulness + context precision + recall. "
                         "standard: + factual correctness. full: + noise sensitivity. "
                         "answer_relevancy is excluded everywhere - it returns NaN "
                         "with a Claude judge on ragas 0.3.9.")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    require_api_key()
    from ragbench.providers import describe

    print(f"providers: {describe()}")
    testset = load()
    if args.limit:
        testset = testset[: args.limit]
    print(f"{len(testset)} gold questions\n")

    grid = default_grid()

    from ragbench.evaluate import estimate_calls

    est = estimate_calls(args.metrics, len(testset), len(grid) + (0 if args.no_rerank else 1))
    print(f"metric set: {args.metrics}")
    print(f"estimated API calls: {est['generation_calls']:,} generation + "
          f"{est['judge_calls']:,} judge = {est['total']:,} total")
    print("(cached configurations are skipped and cost nothing)\n")
    if not args.yes:
        try:
            if input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
                raise SystemExit("aborted - nothing spent")
        except EOFError:
            raise SystemExit("no tty; re-run with --yes if you accept the cost")

    print(f"Stage 1: {len(grid)} chunker x retrieval configurations")
    results = [
        evaluate_spec(s, testset, skip_existing=not args.force, metric_set=args.metrics)
        for s in grid
    ]

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
            metric_set=args.metrics,
        )

    print("\nDone. Now run: python scripts/05_report.py")
