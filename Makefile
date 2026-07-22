.PHONY: help dev-install lint lint-fix format format-check typecheck audit test test-cov build clean check all

.DEFAULT_GOAL := help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

dev-install:  ## Install development dependencies (linting, coverage, type-checking)
	uv pip install -e ".[dev]"

lint:  ## Run ruff linter
	uv run ruff check .

lint-fix:  ## Auto-fix ruff lint violations where possible
	uv run ruff check --fix .

format:  ## Run ruff formatter (in-place)
	uv run ruff format .

format-check:  ## Check formatting without changing files
	uv run ruff format --check .

typecheck:  ## Run pyright static type checker (src/ only)
	uv run pyright src/

audit:  ## Scan dependencies for known CVEs (requires network)
	uvx pip-audit

test:  ## Run test suite
	uv run pytest

test-cov: dev-install  ## Run tests with coverage report (XML + terminal)
	uv run pytest \
		--cov=src/pysharpe \
		--cov-report=term \
		--cov-report=xml:coverage.xml

build:  ## Build distribution packages (wheel + sdist)
	uv build

clean:  ## Remove build artifacts, caches, and coverage files
	@rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@rm -f coverage.xml .coverage pytest_results.log
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

check: lint format-check  ## Quick CI check (lint + format only)

all: check test-cov build  ## Full CI pipeline (lint, format-check, tests, coverage, build)
