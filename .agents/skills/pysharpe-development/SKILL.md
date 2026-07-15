---
name: pysharpe-development
description: >-
  PySharpe architecture, development commands, and test-targeting guidance.
  Invoke when writing or modifying any PySharpe code, adding new public
  symbols, choosing which test subset to run, or understanding module
  boundaries. Covers: uv/pytest/ruff commands, lazy __getattr__ export
  registration, settings singleton (get_settings LRU-cached), data pipeline
  stages, optimization subpackage structure, execution allocator/rebalance
  flow, Streamlit app layout, and the quick-reference test-to-module mapping
  table for targeted test runs.
---

# PySharpe Development Guide

Invoke this skill whenever you start work on PySharpe code. It provides the
architecture overview, development commands, and test-targeting guidance needed
to make changes safely.

## Quick-Start Commands

```bash
# Install for development (all extras + linting/testing tools)
uv pip install -e .[dev]

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_metrics.py

# Run a single test by name
uv run pytest tests/test_metrics.py::test_sharpe_ratio

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Run the Streamlit dashboard
uv run streamlit run app.py

# Run the CLI
uv run pysharpe --help
```

## Key Conventions

- `ruff` is the sole formatter and linter (88-char line length, double quotes).
  `black` is listed in dev deps but ruff-format is authoritative.
- Tests use only synthetic data (no network calls). Fixtures live in
  `tests/conftest.py`. The `data/` directory under `tests/` holds fixture CSVs.
- `get_settings()` is LRU-cached; call `get_settings.cache_clear()` in tests
  that need to vary env vars.
- `portfolio_config.json` in the working directory is auto-loaded by the CLI
  for MER/geo constraints. Pass `--config` to override.
- `proxy_map.json` maps tickers to proxy tickers with optional FX and weight
  adjustments. Loaded by `build_settings()`.

## Public API Registration

The package uses lazy `__getattr__` in `src/pysharpe/__init__.py`. Heavyweight
submodules (PyMC, statsmodels, etc.) are imported on first access only.

**When adding a new public symbol, you MUST register it in three places:**

1. **`_EXPORT_MAP` dict** — Maps attribute name to `(module_path, symbol_name)`.
2. **`TYPE_CHECKING` block** — Static import so type-checkers can resolve it.
3. **`__all__` list** — So `from pysharpe import *` works correctly.

Missing any of these three means the symbol is inaccessible at runtime or
invisible to tooling.

## Architecture Overview

For the full module-level architecture with data flow diagrams, load:

- `references/architecture.md`

## Test Targeting

For the complete test-to-module mapping table (which test files cover which
source modules), load:

- `references/test-map.md`

Use this to run only the relevant test subset instead of the full suite.
