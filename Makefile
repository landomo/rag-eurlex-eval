.PHONY: install ingest index testset run report all test clean

install:
	python -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

ingest:
	.venv/bin/python scripts/01_ingest.py

index:
	.venv/bin/python scripts/02_build_indexes.py

testset:
	.venv/bin/python scripts/03_make_testset.py --n-generated 45

run:
	.venv/bin/python scripts/04_run_experiments.py

report:
	.venv/bin/python scripts/05_report.py

all: ingest index testset run report

test:
	.venv/bin/python -m pytest tests/ -q

clean:
	rm -rf data/indexes results/runs
