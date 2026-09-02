# Fair Code - contributor task runner.  See CONTRIBUTING.md.
# Reproduces locally what CI runs (.github/workflows: audits.yml, lint.yml,
# build-explainers.yml) so you can catch failures before you push.

.DEFAULT_GOAL := help
PY := python3

.PHONY: help setup test coverage build-explainers favicons fix-explainer-count lint check

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS = ":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Install the package plus the dev tools (pytest, pytest-cov, pre-commit, ruff)
	$(PY) -m pip install -e ".[excel,parquet,proxy,mcp]" pytest pytest-cov pre-commit ruff

test:  ## Run the full test suite (mirrors CI)
	$(PY) -m pytest tests/ -q

coverage:  ## Run the full test suite with a faircode/ coverage report (informational only)
	$(PY) -m pytest tests/ -q --cov=faircode --cov-report=term-missing

build-explainers:  ## Regenerate explainer pages, data.js, sitemap, and OG images (dark + light)
	$(PY) scripts/build_explainers.py
	$(PY) scripts/generate_og_images.py

favicons:  ## Regenerate favicon.ico/PNGs and apple-touch-icon.png from logo.svg
	$(PY) scripts/generate_favicons.py

fix-explainer-count:  ## Correct stale "N explainers" mentions in README/CONTRIBUTORS/METRICS/ROADMAP
	$(PY) scripts/check_explainer_count.py --fix

lint:  ## Enforce the em-dash-free rule + check for broken doc links + ruff (mirrors the lint workflow)
	$(PY) scripts/check_em_dash.py
	$(PY) scripts/check_broken_links.py
	ruff check faircode scripts tests

check: lint test  ## Run everything CI runs (lint + full test suite)
	@echo "All checks passed."
