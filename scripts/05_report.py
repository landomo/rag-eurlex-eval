"""Aggregate runs into results/RESULTS.md and results/summary.csv."""
import _bootstrap  # noqa: F401

from ragbench.report import write_report

if __name__ == "__main__":
    print(write_report())
