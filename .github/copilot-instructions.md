# Copilot Instructions for PySharpe

## Project Overview
PySharpe is a Python toolkit for building, analyzing, and visualizing investment portfolios that maximize the Sharpe ratio. It integrates data ingestion (from Yahoo Finance via `yfinance`), portfolio optimization (using `PyPortfolioOpt`), and visualization utilities.

## Architecture & Key Components
- **src/pysharpe/**: Main package (src-layout)
  - `config.py`: Centralized settings and path management
  - `data/`: Market data ingestion and collation helpers
  - `optimization/`: Portfolio optimization logic (max-Sharpe, dataclasses)
  - `visualization/`: Plotting and DCA simulation utilities
  - `portfolio_optimization.py`: Main optimization engine
  - `workflows.py`: High-level orchestration for CLI and notebooks
  - `cli.py`: Command-line interface entry point
- **data/**: Input/output data (portfolios, price histories, exports)
- **tests/**: Automated test suite (pytest)
- **drafts/**: Exploratory scripts and plotting experiments

## Developer Workflows
- **Environment**: Use a virtualenv (see `environment/` or create `.venv`).
- **Install**: `pip install -e .[dev]` (editable + dev dependencies)
- **Test**: `python3 -m pytest` (runs all tests in `tests/`)
- **Lint**: `ruff src/ tests/` (enforced in CI)
- **CLI**: Use `pysharpe` or `python -m pysharpe.cli` for workflows:
  - `pysharpe download ...` (fetches and collates price data)
  - `pysharpe optimize ...` (runs portfolio optimization)
- **Artifacts**: Data and results are written to `data/price_hist/`, `data/exports/`, and `logs/` by default. Override with CLI flags as needed.

## Project-Specific Patterns
- **Portfolios**: Defined as newline-delimited ticker lists in `data/portfolio/`.
- **Data Flow**: Downloaded price data → collated per-portfolio CSVs → optimization → weights/performance/plots.
- **Workflows**: High-level functions in `pysharpe.workflows` mirror CLI commands and are importable for scripting/notebooks.
- **Visualization**: DCA simulation and plotting in `pysharpe.visualization` (see README for usage).
- **Configuration**: Paths and settings are managed centrally in `config.py`.

## Integration & Dependencies
- **External**: `yfinance`, `PyPortfolioOpt`, `matplotlib` (for plots)
- **Testing**: `pytest`, `ruff` (see `requirements-dev.txt`)

## Examples
- Download and optimize from Python:
  ```python
  from pysharpe.workflows import download_portfolios, optimise_portfolios
  download_portfolios(start="2020-01-01")
  optimise_portfolios(make_plot=False)
  ```
- CLI usage:
  ```bash
  pysharpe download --interval 1d --period 1y
  pysharpe optimize --no-plot
  ```

## Conventions
- Use the src-layout for imports: `from pysharpe...`
- Keep exploratory scripts in `drafts/`, not in main package
- All new features should include tests in `tests/`
- Run `ruff` before submitting PRs

Refer to `README.md` and `docs/` for more details on workflows and architecture.
