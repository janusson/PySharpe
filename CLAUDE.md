# PySharpe Architecture & Development Guide

## Quick-Start Commands

```bash
# Install (recommended: uv)
uv pip install -e .[all]

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_metrics.py
uv run pytest tests/test_metrics.py::test_sharpe_ratio

# Lint & format (ruff is authoritative)
uv run ruff check .
uv run ruff format .

# Run the Streamlit dashboard
uv run streamlit run app.py

# CLI entry point
uv run pysharpe --help
```

## Architecture Overview

PySharpe follows a layered pipeline from data ingestion through computation to
execution and presentation.

```
Config Layer   →  config.py (LRU-cached singleton), portfolio_config.json, proxy_map.json
Data Pipeline  →  YFinance → DuckDB cache → FX (CAD, no .bfill()) → CSV collation → DuckDB linkage
Computation    →  metrics.py (stateless) + optimization/ (pypfopt + PyMC) + analysis/ (backtests, GARCH, VAR)
Execution      →  allocator.py (60/40 VA) + rebalance.py + tax_tracker.py + cash_flow_rebalance.py
Presentation   →  cli.py (5 subcommands) + app.py (Streamlit, 4 tabs) + visualization/
```

See `docs/flowchart.md` for the full mermaid diagram with data-flow connections.

## Package Layout

```
src/pysharpe/
├── __init__.py              # Lazy __getattr__ public API
├── config.py                # PySharpeSettings, get_settings()
├── metrics.py               # Sharpe, Sortino, CAGR, max drawdown, etc.
├── portfolio_optimization.py # Top-level: collate → optimize → export
├── workflows.py             # download_portfolios(), optimise_portfolios()
├── cli.py                   # 5 subcommands: optimise, rebalance, allocate, simulate-dca, plot
├── data/
│   ├── fetcher.py           # YFinancePriceFetcher + DuckDBCachedPriceFetcher
│   ├── collation.py         # CSV collation into unified DataFrames
│   ├── linkage.py           # DuckDB cross-dataset joins + proxy stitching
│   ├── portfolio.py         # PortfolioDefinition, PortfolioRepository
│   └── workflows.py         # PortfolioDownloadWorkflow
├── optimization/
│   ├── sharpe_optimizer.py  # PyPortfolioOpt EfficientFrontier wrapper
│   ├── bayesian.py          # PyMC posterior return/covariance estimation
│   ├── black_litterman.py   # Black-Litterman model with investor views
│   ├── expected_returns.py  # EMA, mean, shrinkage, constant-return
│   ├── tax_location.py      # 2-D Asset Location Matrix (TFSA/RRSP/NON_REG)
│   ├── models.py            # PortfolioWeights, OptimisationResult
│   └── weights.py           # Weight normalization utilities
├── execution/
│   ├── allocator.py         # score_opportunities(), allocate_contribution()
│   ├── rebalance.py         # build_rebalance_plan(), format_rebalance_plan()
│   ├── cash_flow_rebalance.py # Multi-account contribution routing
│   ├── tax_tracker.py       # ACB tracking (CRA weighted-average method)
│   └── brokerage.py         # Whole-share rounding, commissions, slippage
├── analysis/
│   ├── backtest_engine.py   # Calendar + drift-band rebalancing backtests
│   ├── time_series.py       # ADF, GARCH, VAR
│   ├── benchmarks.py        # Canadian ETF baselines (VEQT, XEQT, etc.)
│   ├── comparison.py        # Head-to-head fund comparison
│   ├── scoring.py           # Strategy scoring utilities
│   └── visualization.py     # Backtest result visualization
├── validation/
│   ├── friction.py          # Transaction cost stress-testing
│   ├── ledger.py            # PBO computation
│   ├── resampling.py        # Purged cross-validation
│   └── metrics.py           # Statistical validation metrics
├── guardrails/
│   └── tax_compliance.py    # CRA rule enforcement
├── visualization/
│   ├── frontier.py          # Efficient frontier plots
│   ├── dca.py               # DCA projection plots
│   ├── equity_curve.py      # Portfolio equity curves
│   └── correlation.py       # Correlation heatmaps
└── app/
    ├── analytics.py         # Streamlit analytics page
    ├── backtest.py          # Streamlit backtest page
    ├── charts.py            # Shared chart helpers
    ├── data.py              # Data inspection page
    ├── dca.py               # DCA projection page
    └── rebalance_ui.py      # Rebalance UI helpers
```

## Key Conventions

- **ruff** is the sole formatter and linter (88-char line length, double quotes).
- **Tests** use synthetic data only with fixed seeds — no network calls.
- **`get_settings()`** is LRU-cached; call `get_settings.cache_clear()` in tests
  that vary env vars.
- **`portfolio_config.json`** in the working directory is auto-loaded for MER/
  geo constraints. Pass `--config` to override.
- **`proxy_map.json`** maps tickers to proxy tickers with optional FX and weight
  adjustments.

## Public API Registration

The package uses lazy `__getattr__` in `src/pysharpe/__init__.py`. Heavyweight
submodules (PyMC, statsmodels, etc.) are imported on first access only.

**When adding a new public symbol, register it in three places:**

1. **`_EXPORT_MAP` dict** — Maps attribute name to `(module_path, symbol_name)`.
2. **`TYPE_CHECKING` block** — Static import so type-checkers can resolve it.
3. **`__all__` list** — So `from pysharpe import *` works correctly.

Missing any of these three means the symbol is inaccessible at runtime or
invisible to tooling.

## Asset Universe

PySharpe is tuned for **broad-market, CAD-denominated index ETFs**. Example
tickers: VFV.TO, VCN.TO, QQC.TO, VDY.TO, VIU.TO, VEE.TO, VMO.TO, VVL.TO.

Prohibited: single-stock models, sentiment analysis, options-pricing logic,
predictive ML, day-trading algorithms, gamified UI.

## Investment Philosophy

- **Primary objective:** Deterministic Value Averaging (VA), not Markowitz MPT.
- **Opportunity score:** 60% path drift + 40% valuation/mean-reversion (by default).
- **Tax-advantaged:** TFSA assumed; no tax-loss harvesting (TFSA prohibits).
- **Foreign withholding tax:** Modeled as strict yield reduction on US dividends.
- **MER values:** Always decimal fractions (< 0.10), never percentage points.

## Test Targeting

See `docs/TEST_MAP.md` for the complete test-to-module mapping table to run
only the relevant test subset instead of the full suite.
