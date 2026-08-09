"""Download the EU AI Act + GDPR from EUR-Lex and segment them into sections."""
import argparse

import _bootstrap  # noqa: F401

from ragbench.ingest import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    print("Ingesting EUR-Lex corpus...")
    sections = build(force=args.force)
    kinds = {}
    for s in sections:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    print(f"\n{len(sections)} sections total: {kinds}")
    print("Wrote data/processed/sections.jsonl")
