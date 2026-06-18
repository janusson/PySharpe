# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

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

## Architecture

The package lives in `src/pysharpe/`. The public API is exposed via lazy `__getattr__` in `__init__.py` — heavyweight submodules (PyMC, statsmodels, etc.) are imported on first access only to avoid slow startup. Always add new public symbols to both the `_EXPORT_MAP` dict and the `TYPE_CHECKING` block in `__init__.py`.

**Settings** (`config.py`) — A frozen `PySharpeSettings` dataclass is the single source of truth for directory layout (`data/`, `data/portfolio/`, `data/price_hist/`, `data/exports/`, `data/cache/`). Always access it via `get_settings()` (LRU-cached singleton). Override the root with the `PYSHARPE_DATA_DIR` env var. `proxy_map.json` in the working directory is loaded automatically at startup to resolve ticker aliases.

**Data pipeline** (two stages):
1. `data/fetcher.py` — Abstract `PriceFetcher` with two implementations: `YFinancePriceFetcher` (live yfinance calls) and `DuckDBCachedPriceFetcher` (write-through DuckDB cache at `data/cache/pysharpe_cache.db`). FX conversion happens here when `base_currency` is not the native currency.
2. `data/collation.py` — Joins per-ticker price CSVs into a single collated DataFrame. Used by the download workflow.

**Workflows** (`workflows.py`) — Two top-level orchestrators: `download_portfolios()` and `optimise_portfolios()`. The CLI (`cli.py`) calls these. Portfolio definitions are CSV files in `data/portfolio/`; collated data and optimisation artefacts go to `data/exports/`.

**Optimisation** (`optimization/`):
- `sharpe_optimizer.py` — EfficientFrontier wrapper (PyPortfolioOpt) with optional MER and geographic constraints.
- `bayesian.py` — PyMC-based posterior estimation of asset returns; produces expected returns and covariance for use in place of historical estimates.
- `models.py` — Shared frozen dataclasses: `PortfolioWeights`, `OptimisationPerformance`, `OptimisationResult`.
- `portfolio_optimization.py` (top-level) — `optimise_portfolio()` ties collation → optimizer → artefact writes together.

**Execution** (`execution/`):
- `allocator.py` — `score_opportunities()` computes a blended opportunity score (drift 60% + valuation 40% by default, configurable via `AllocationConfig`). `allocate_contribution()` converts scores to dollar amounts.
- `rebalance.py` — `build_rebalance_plan()` loads saved artefacts (`<name>_weights.txt`, `<name>_collated.csv`), merges with current holdings, then calls the allocator.

**Analysis** (`analysis/`):
- `time_series.py` — ADF stationarity, GARCH volatility forecasting, VAR modeling.
- `backtest_engine.py` — Calendar and drift-based rebalancing backtests.
- `categorization.py` — Groups correlated tickers by category before optimisation.
- `scoring.py` — Shared scoring utilities used by backtests and benchmarks.

**App** (`app/`) — Streamlit pages split across `analytics.py`, `charts.py`, `data.py`, `dca.py`. The entry point is `app.py` in the repo root.

## Key conventions

- `portfolio_config.json` in the working directory is auto-loaded by the CLI for MER/geo constraints. Pass `--config` to override.
- `proxy_map.json` maps tickers to proxy tickers with optional FX and weight adjustments. Loaded by `build_settings()`.
- Tests use only synthetic data (no network calls). Fixtures live in `tests/conftest.py`. The `data/` directory under `tests/` holds fixture CSVs.
- `ruff` is the sole formatter and linter (88-char line length, double quotes). `black` is listed in dev deps but ruff-format is authoritative.
- `get_settings()` is LRU-cached; call `get_settings.cache_clear()` in tests that need to vary env vars.
