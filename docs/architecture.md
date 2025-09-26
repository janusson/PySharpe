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

- `pysharpe.data`: APIs for data ingestion and preprocessing.
- `pysharpe.optimization`: Portfolio construction and rebalancing utilities.
- `pysharpe.visualization`: Plotting helpers and reusable chart styles.

## Next steps

1. Expand automated test coverage for data ingestion edge cases.
2. Add a command-line interface for running optimisations from the terminal.
3. Create example Jupyter notebooks demonstrating the end-to-end workflow.
