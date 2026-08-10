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
    ap.add_argument("--seeds-only", action="store_true",
                    help="rebuild only the hand-written seed items, keep existing generated ones")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if a testset already exists (costs money, "
                         "and invalidates comparability with existing runs)")
    args = ap.parse_args()

    if TESTSET_PATH.exists() and not args.force and not args.seeds_only:
        n = sum(1 for _ in TESTSET_PATH.open())
        raise SystemExit(
            f"A gold set already exists ({n} items at {TESTSET_PATH.relative_to(ROOT)}).\n"
            "Regenerating costs API calls AND produces different questions, which would\n"
            "make new runs incomparable with the ones already in results/runs/.\n"
            "Pass --force if that is really what you want."
        )

    sections = load_sections()
    items = []

    kept_generated = []
    if args.seeds_only and TESTSET_PATH.exists():
        from ragbench.testset import load as load_testset

        from ragbench.testset import _is_refusal

        existing = [i for i in load_testset() if i.origin == "generated"]
        kept_generated = [i for i in existing if not _is_refusal(i.reference)]
        dropped = len(existing) - len(kept_generated)
        print(f"Keeping {len(kept_generated)} existing generated items (no new cost for those).")
        if dropped:
            print(f"Dropped {dropped} generated items whose reference was a hedged refusal "
                  f"- those cannot serve as ground truth.")
        args.n_generated = 0

    if not args.no_seed:
        seed_file = ROOT / "config" / "seed_questions.yaml"
        seeds = yaml.safe_load(seed_file.read_text(encoding="utf-8"))["questions"]
        n_neg = sum(1 for s_ in seeds if isinstance(s_, dict) and s_.get("category") == "negative")
        print(f"Building references for {len(seeds)} hand-written questions "
              f"({n_neg} negatives need no LLM call)...")
        items += build_seed_items(seeds, sections)

    if args.n_generated:
        print(f"Generating {args.n_generated} questions from sampled Articles...")
        items += generate_from_sections(sections, n=args.n_generated)

    items += kept_generated
    save(items)
    origins = {}
    for it in items:
        origins[it.category] = origins.get(it.category, 0) + 1
    print(f"\n{len(items)} gold items written to {TESTSET_PATH.relative_to(ROOT)}: {origins}")
    print("Seed items are flagged needs_review=true - read them before trusting the numbers.")
