"""Chunk the corpus four ways and build one Chroma collection per strategy."""
import argparse

import _bootstrap  # noqa: F401

from ragbench.config import CHUNKERS
from ragbench.index import build_index
from ragbench.ingest import load_sections

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=CHUNKERS)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    import time

    sections = load_sections()
    print(f"{len(sections)} sections loaded\n")
    print(f"{'strategy':<22} {'chunks':>7} {'mean':>8} {'p50':>7} {'p95':>7} {'max':>7}")
    print("-" * 62)
    for strategy in args.strategies:
        print(f"  building {strategy} ...", flush=True)
        t0 = time.time()
        _, st = build_index(strategy, sections=sections, rebuild=args.rebuild)
        print(
            f"{strategy:<22} {st['n_chunks']:>7} {st['chars_mean']:>8.0f} "
            f"{st['chars_p50']:>7} {st['chars_p95']:>7} {st['chars_max']:>7}"
            f"   [{time.time() - t0:.0f}s]",
            flush=True,
        )
    print("\nIndexes written to data/indexes/")
