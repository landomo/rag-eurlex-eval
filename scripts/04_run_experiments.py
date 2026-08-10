"""Run the ablation grid, then add cross-encoder reranking on the winner."""
import argparse

import _bootstrap  # noqa: F401

from ragbench.config import RunSpec, default_grid, require_api_key
from ragbench.evaluate import evaluate_spec
from ragbench.testset import load

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--chunkers", nargs="*", default=None,
                    help="restrict the grid to these chunking strategies")
    ap.add_argument("--rerank", action="store_true",
                    help="after the grid, run cross-encoder reranking on the winner "
                         "(one extra configuration - opt in, so it cannot surprise you)")
    ap.add_argument("--no-rerank", action="store_true", help=argparse.SUPPRESS)
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
    if args.chunkers:
        grid = [s for s in grid if s.chunker in args.chunkers]
        if not grid:
            raise SystemExit(f"No configurations match --chunkers {args.chunkers}")

    from ragbench.evaluate import already_done, estimate_calls

    # Count only what will actually run. Quoting the whole grid when most of it is
    # cached is how a "confirm the estimate" prompt stops meaning anything.
    todo = [s for s in grid if args.force or not already_done(s.run_key(len(testset)))]
    cached = len(grid) - len(todo)
    n_billed = len(todo) + (1 if args.rerank else 0)

    est = estimate_calls(args.metrics, len(testset), n_billed)
    print(f"metric set : {args.metrics}")
    print(f"gold set   : {len(testset)} questions")
    print(f"configs    : {len(grid)} requested, {cached} already cached (free), "
          f"{len(todo)} to run{' + 1 rerank' if args.rerank else ''}")
    print(f"ESTIMATED API CALLS: {est['total']:,} "
          f"({est['generation_calls']:,} generation + {est['judge_calls']:,} judge)")
    if not args.rerank:
        print("(reranking stage not included - pass --rerank to add it)")
    print()
    if not todo and not args.rerank:
        print("Everything requested is already cached. Nothing to spend.")
        raise SystemExit(0)
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

    if args.rerank:
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
