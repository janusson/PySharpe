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

- `pysharpe.config`: Centralised configuration points (paths, logging, artefact versioning).
- `pysharpe.data`: Portfolio repositories, price fetchers, and collation workflows.
- `pysharpe.optimization`: Dataclasses describing optimisation outcomes.
- `pysharpe.portfolio_optimization`: Max-Sharpe optimiser built on PyPortfolioOpt.
- `pysharpe.visualization`: Plotting helpers (including DCA projections) and reusable chart styles.
- `pysharpe.workflows`: High-level orchestration for CLI, notebooks, or future UIs.

## Next steps

The CLI outlined above now ships with the project. Upcoming milestones focus on
rounding out developer ergonomics and user-facing documentation:

1. Expand automated test coverage for data ingestion, CLI flows, and
   visualisation utilities.
2. Create example Jupyter notebooks demonstrating the end-to-end workflow.
3. Prepare deployment logistics (release automation and packaging guidance).
