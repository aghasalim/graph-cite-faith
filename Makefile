.PHONY: setup counterfactual test
PY := .venv/bin/python
setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt
counterfactual:   ## the 2x2: does the narrative track the subgraph or the label?
	$(PY) -m experiments.counterfactual
test:
	$(PY) -m pytest tests/ -q
