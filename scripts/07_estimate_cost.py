"""Estimate spend in DOLLARS before running anything.

Call counts are not a budget. This prices a plan using the measured chunk sizes
of each strategy, since context length is what actually drives cost.
"""
import argparse

import _bootstrap  # noqa: F401

from ragbench.evaluate import METRIC_SETS
from ragbench.index import load_chunks

CHARS_PER_TOKEN = 3.6
# Claude Haiku 4.5 list price, USD per token.
IN_RATE, OUT_RATE = 1.0 / 1e6, 5.0 / 1e6

# Fixed overheads per question, beyond the retrieved context itself.
OVERHEAD_IN = {"core": 6_000, "standard": 7_900, "full": 12_000}
OUT_TOK = {"core": 1_850, "standard": 2_350, "full": 3_100}


def context_tokens(strategy: str, k: int) -> float:
    docs = load_chunks(strategy)
    mean_chars = sum(len(d.page_content) for d in docs) / len(docs)
    return k * mean_chars / CHARS_PER_TOKEN


def cost_per_question(strategy: str, metric_set: str, k: int) -> float:
    c = context_tokens(strategy, k)
    # context is re-read by generation, faithfulness (x2), precision and recall
    tokens_in = 5 * c + OVERHEAD_IN[metric_set]
    return tokens_in * IN_RATE + OUT_TOK[metric_set] * OUT_RATE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunkers", nargs="*",
                    default=["fixed_512", "recursive_1000", "structural_article"])
    ap.add_argument("--modes", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget", type=float, default=5.0)
    args = ap.parse_args()

    print(f"{'strategy':<22}{'ctx tokens':>12}{'  $/question (core)':>20}{'  (standard)':>14}")
    print("-" * 70)
    per_q = {}
    for s in args.chunkers:
        c = context_tokens(s, args.k)
        core = cost_per_question(s, "core", args.k)
        std = cost_per_question(s, "standard", args.k)
        per_q[s] = (core, std)
        print(f"{s:<22}{c:>12,.0f}{core:>20.4f}{std:>14.4f}")

    print()
    print(f"complete {len(args.chunkers)}x{args.modes} table, by gold-set size:")
    print(f"{'n':>5}{'core':>12}{'standard':>12}   fits ${args.budget:.0f}?")
    print("-" * 46)
    for n in (15, 20, 25, 27, 35, 50, 70):
        core = sum(v[0] for v in per_q.values()) * args.modes * n
        std = sum(v[1] for v in per_q.values()) * args.modes * n
        fit = "core" if core <= args.budget else ""
        if std <= args.budget:
            fit = "both"
        elif core > args.budget:
            fit = "neither"
        print(f"{n:>5}{core:>12.2f}{std:>12.2f}   {fit}")
