"""Build the gold evaluation set: hand-written seeds + LLM-generated questions."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from ragbench.config import ROOT, TESTSET_PATH
from ragbench.ingest import load_sections
from ragbench.testset import build_seed_items, generate_from_sections, save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-generated", type=int, default=45)
    ap.add_argument("--no-seed", action="store_true")
    args = ap.parse_args()

    sections = load_sections()
    items = []

    if not args.no_seed:
        seed_file = ROOT / "config" / "seed_questions.yaml"
        seeds = yaml.safe_load(seed_file.read_text(encoding="utf-8"))["questions"]
        print(f"Building references for {len(seeds)} hand-written questions...")
        items += build_seed_items(seeds, sections)

    if args.n_generated:
        print(f"Generating {args.n_generated} questions from sampled Articles...")
        items += generate_from_sections(sections, n=args.n_generated)

    save(items)
    origins = {}
    for it in items:
        origins[it.origin] = origins.get(it.origin, 0) + 1
    print(f"\n{len(items)} gold items written to {TESTSET_PATH.relative_to(ROOT)}: {origins}")
    print("Seed items are flagged needs_review=true - read them before trusting the numbers.")
