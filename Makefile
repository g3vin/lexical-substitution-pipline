SHELL := /bin/bash

PYTHON ?= python
VENV_DIR ?= .venv
TOP_K ?= 10
REPORT_DIR ?= report_plots_model
SUBSTITUTE_MODEL ?= ./substitute-roberta
RANKER_CONFIG ?= ./ranker_config.json
LEARNED_RANKER_MODEL ?= ./learned_ranker.pkl
SWORDS_SPLIT ?= dev

.DEFAULT_GOAL := help

.PHONY: help venv install install-editable resources demo demo-fast batch eval tune train-substitute train-ranker plots plots-model clean clean-pyc

help:
	@echo "Available targets:"
	@echo "  make venv              Create a local virtual environment in $(VENV_DIR)"
	@echo "  make install           Install locked dependencies and the package"
	@echo "  make install-editable  Install locked dependencies and the package in editable mode"
	@echo "  make resources         Download spaCy and NLTK language resources"
	@echo "  make demo              Run the default lexical substitution demo"
	@echo "  make demo-fast         Run a faster demo configuration"
	@echo "  make batch             Generate example_outputs.json from built-in examples"
	@echo "  make eval              Evaluate on SWORDS using TOP_K=$(TOP_K)"
	@echo "  make tune              Tune the ranker config on SWORDS"
	@echo "  make train-substitute  Train the substitute-quality model"
	@echo "  make train-ranker      Train the learned deep ranker"
	@echo "  make plots             Generate dataset report plots"
	@echo "  make plots-model       Generate dataset and model report plots"
	@echo "  make clean             Remove Python cache files"

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install .

install-editable:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

resources:
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m nltk.downloader wordnet omw-1.4

demo:
	$(PYTHON) -m pipeline

demo-fast:
	$(PYTHON) -m pipeline "The dog chased the cat." "chased" --generation-mode fast --no-reranker

batch:
	LEARNED_RANKER_MODEL=$(LEARNED_RANKER_MODEL) ENABLE_RERANKER=1 bash run_100_examples.sh

eval:
	$(PYTHON) evaluate_swords.py --split $(SWORDS_SPLIT) --top-k $(TOP_K) --substitute-model $(SUBSTITUTE_MODEL) --learned-ranker-model $(LEARNED_RANKER_MODEL)

tune:
	$(PYTHON) evaluate_swords.py --split $(SWORDS_SPLIT) --top-k $(TOP_K) --substitute-model $(SUBSTITUTE_MODEL) --tune --output-config $(RANKER_CONFIG)

train-substitute:
	$(PYTHON) train_swords_substitute_classifier.py --output-dir $(SUBSTITUTE_MODEL)

train-ranker:
	$(PYTHON) train_swords_ranker.py --ranker-config $(RANKER_CONFIG) --substitute-model $(SUBSTITUTE_MODEL)

plots:
	$(PYTHON) plot_swords_report.py --split $(SWORDS_SPLIT) --output-dir $(REPORT_DIR)

plots-model:
	$(PYTHON) plot_swords_report.py --split $(SWORDS_SPLIT) --include-model-plots --substitute-model $(SUBSTITUTE_MODEL) --ranker-config $(RANKER_CONFIG) --learned-ranker-model $(LEARNED_RANKER_MODEL) --top-k $(TOP_K) --output-dir $(REPORT_DIR)

clean: clean-pyc

clean-pyc:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
