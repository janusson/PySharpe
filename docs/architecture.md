# PySharpe architecture

This document outlines the guiding principles and planned components for the
project. It serves as a living document to help contributors align on design
decisions before implementation work begins.

## Core objectives

- Fetch historical market data from Yahoo Finance using the `yfinance` package.
- Optimise portfolio allocations for the maximum Sharpe ratio using
  `pyportfolioopt`.
- Provide convenient visualisations of the efficient frontier and optimized
  portfolio results.

## Module layout

- `pysharpe.config`: Centralised configuration (paths, logging, MER/geo constraints), the LRU-cached `get_settings()` singleton, and account/tax profiles (`TaxProfile`, `AssetTaxProfile`).
- `pysharpe.exceptions`: Domain error hierarchy (`PySharpeError`, `DataIngestionError`, `DataValidationError`, `ExecutionConfigError`).
- `pysharpe.data`: Price fetchers (yfinance with DuckDB write-through cache), FX conversion (CAD, no `.bfill()`), CSV collation, and DuckDB-based linkage (`DataLinker`, `HistoryLinker`).
- `pysharpe.data_collector`: Legacy CLI-era data collection helpers (`PortfolioTickerReader`, `download_portfolio_prices`, `collate_prices`).
- `pysharpe.metrics`: Stateless, vectorized performance metrics (Sharpe, Sortino, Calmar, CAGR, max drawdown, tracking error).
- `pysharpe.optimization`: Estimators (analytical Ledoit-Wolf linear/nonlinear shrinkage), Max-Sharpe optimizer on PyPortfolioOpt, Bayesian (PyMC), Black-Litterman, Hierarchical Risk Parity, tax-location engine, and result dataclasses (`PortfolioWeights`, `OptimisationResult`).
- `pysharpe.portfolio_optimization`: Orchestrates collate → optimise → export artefact workflows.
- `pysharpe.analysis`: Time-series modelling (ADF, GARCH, VAR), backtest engine (calendar/drift rebalancing with transaction costs), Canadian ETF benchmarks, categorisation, scoring, and head-to-head fund comparison.
- `pysharpe.execution`: Value-averaging allocator (60/40 scoring blend), rebalance planning, cash-flow rebalancing (`cash_flow_rebalance.py`), brokerage export, and ACB tax tracking.
- `pysharpe.validation`: Deflated Sharpe Ratio, PBO ledger, purged cross-validation with autocorrelation-aware embargoes, execution-friction stress-testing, and sample-size adequacy.
- `pysharpe.guardrails`: CRA tax-compliance enforcement (superficial-loss interlock, multi-account validation).
- `pysharpe.visualization`: Plotting helpers (efficient frontier, DCA projections, equity curves, correlation heatmaps) with optional-dependency wrappers (`visualization/utils.py`).
- `pysharpe.workflows`: High-level orchestration for CLI, notebooks, or future UIs.
- `pysharpe.app`: Streamlit dashboard (metrics & comparison, efficient frontier, backtesting, DCA, rebalancing, raw data & logs).

## Status

As of v1.0.0 the CLI (`optimise`, `allocate`, `rebalance`, `simulate-dca`, `plot`), the Streamlit dashboard, and the 990+ test suite are shipped. This page is a high-level orientation; `CLAUDE.md` is the authoritative, always-current architecture reference, and `docs/flowchart.md` diagrams the full data flow end to end.
