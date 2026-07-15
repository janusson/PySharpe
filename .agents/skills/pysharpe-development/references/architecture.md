# PySharpe Architecture Reference

## Package Layout

The package lives in `src/pysharpe/`. The public API is exposed via lazy
`__getattr__` in `__init__.py`.

## Settings (`config.py`)

A frozen `PySharpeSettings` dataclass is the single source of truth for
directory layout:
- `data/`
- `data/portfolio/`
- `data/price_hist/`
- `data/exports/`
- `data/cache/`

Always access it via `get_settings()` (LRU-cached singleton).
Override the root with the `PYSHARPE_DATA_DIR` env var.
`proxy_map.json` in the working directory is loaded automatically at startup
to resolve ticker aliases.

## Data Pipeline (Two Stages)

1. **`data/fetcher.py`** — Abstract `PriceFetcher` with two implementations:
   - `YFinancePriceFetcher` — Live yfinance calls.
   - `DuckDBCachedPriceFetcher` — Write-through DuckDB cache at
     `data/cache/pysharpe_cache.db`. FX conversion happens here when
     `base_currency` is not the native currency.

2. **`data/collation.py`** — Joins per-ticker price CSVs into a single collated
   DataFrame. Used by the download workflow.

## Workflows (`workflows.py`)

Two top-level orchestrators:
- `download_portfolios()` — Fetches and collates price data.
- `optimise_portfolios()` — Runs optimization on collated data.

The CLI (`cli.py`) calls these. Portfolio definitions are CSV files in
`data/portfolio/`; collated data and optimization artefacts go to
`data/exports/`.

## Optimization (`optimization/`)

- **`sharpe_optimizer.py`** — EfficientFrontier wrapper (PyPortfolioOpt) with
  optional MER and geographic constraints.
- **`bayesian.py`** — PyMC-based posterior estimation of asset returns;
  produces expected returns and covariance for use in place of historical
  estimates.
- **`models.py`** — Shared frozen dataclasses: `PortfolioWeights`,
  `OptimisationPerformance`, `OptimisationResult`.
- **`portfolio_optimization.py`** (top-level) — `optimise_portfolio()` ties
  collation → optimizer → artefact writes together.

## Execution (`execution/`)

- **`allocator.py`** — `score_opportunities()` computes a blended opportunity
  score (drift 60% + valuation 40% by default, configurable via
  `AllocationConfig`). `allocate_contribution()` converts scores to dollar
  amounts.
- **`rebalance.py`** — `build_rebalance_plan()` loads saved artefacts
  (`<name>_weights.txt`, `<name>_collated.csv`), merges with current holdings,
  then calls the allocator.

## Analysis (`analysis/`)

- **`time_series.py`** — ADF stationarity, GARCH volatility forecasting, VAR
  modeling.
- **`backtest_engine.py`** — Calendar and drift-based rebalancing backtests.
- **`categorization.py`** — Groups correlated tickers by category before
  optimization.
- **`comparison.py`** — Stateless head-to-head fund comparison engine that
  evaluates two assets side-by-side using vectorised metrics (CAGR, volatility,
  drawdown depth/duration, Sharpe, Sortino, Calmar, rolling tracking error,
  and return correlation) without invoking multi-asset optimization or VA
  allocation pipelines.
- **`scoring.py`** — Shared scoring utilities used by backtests and benchmarks.

## App (`app/`)

Streamlit pages split across `analytics.py`, `charts.py`, `data.py`, `dca.py`.
The entry point is `app.py` in the repo root.
