"""Validate the gold set without spending a cent.

Every metric is computed against these references. A malformed or hedged
reference is not a small problem - it is ground truth that says the wrong thing,
and no amount of correct plumbing downstream will reveal it.
"""
import _bootstrap  # noqa: F401

import collections

from ragbench.testset import _is_refusal, load

if __name__ == "__main__":
    items = load()
    print(f"{len(items)} items")
    print("  categories:", dict(collections.Counter(i.category for i in items)))

    problems = []
    for i in items:
        if i.category == "negative":
            if "does not contain enough information" not in i.reference:
                problems.append((i.id, "negative lacks the expected refusal reference"))
            continue
        if not i.reference.strip():
            problems.append((i.id, "empty reference"))
        elif _is_refusal(i.reference):
            problems.append((i.id, "reference is a hedged refusal - unusable as ground truth"))
        elif len(i.reference) < 45:
            problems.append((i.id, f"reference suspiciously short ({len(i.reference)} chars)"))
        if not i.reference_contexts:
            problems.append((i.id, "no reference_contexts - context recall cannot be computed"))

    print()
    if problems:
        print(f"{len(problems)} PROBLEMS:")
        for pid, why in problems:
            print(f"  {pid}: {why}")
        raise SystemExit(1)
    print("Gold set is clean. Safe to run the grid.")
