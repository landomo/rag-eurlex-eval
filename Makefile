# macOS ships python3, not python. Override with: make install PYTHON=python3.12
PYTHON ?= python3
VENV := .venv/bin/python

.PHONY: install ingest index testset run report all test smoke clean

install:
	$(PYTHON) -m venv .venv
	$(VENV) -m pip install -U pip
	$(VENV) -m pip install -r requirements.txt
	@echo ""
	@echo "Installed. Next: make smoke"

ingest:
	$(VENV) scripts/01_ingest.py

index:
	$(VENV) scripts/02_build_indexes.py

testset:
	$(VENV) scripts/03_make_testset.py --n-generated 45

# Cheap end-to-end check: 5 questions across all 9 configurations.
smoke:
	$(VENV) scripts/04_run_experiments.py --limit 5

run:
	$(VENV) scripts/04_run_experiments.py

report:
	$(VENV) scripts/05_report.py

all: ingest index testset run report

test:
	$(VENV) -m pytest tests/ -q

clean:
	rm -rf data/indexes results/runs
