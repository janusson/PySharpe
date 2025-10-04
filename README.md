# PySharpe

PySharpe is a lightweight portfolio analytics and optimisation toolkit. It helps you download pricing data, collate custom portfolios, compute diagnostic metrics (returns, volatility, Sharpe ratios), and run maximum-Sharpe optimisations without leaving Python.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The editable install keeps the library and CLI in sync while you develop. The `dev` extras pull in pytest, coverage, ruff, and the other tooling used in this repository.

## Quickstart (Python REPL or notebook)

```python
import pysharpe

# Load the tickers listed in tests/data/sample_portfolio.csv without hitting any APIs.
prices = pysharpe.download_portfolio_prices(
    ["AAPL", "MSFT", "GOOG"],
    price_history_dir="tests/data",  # reuse local fixtures for a deterministic run
    export_dir="tests/data",
)
returns = pysharpe.metrics.compute_returns(prices["AAPL"])
compute_sharpe_ratio = pysharpe.metrics.sharpe_ratio
sharpe = compute_sharpe_ratio(returns)
print(f"Sample Sharpe ratio: {sharpe:.2f}")
```

The `pysharpe.metrics` module also exposes helpers for annualised returns, volatility, and mean estimates so you can extend the analysis in a single workflow.

## Command-line usage

PySharpe ships with a single entry point that is being refactored into clear subcommands. The examples below show the intended UX.

```bash
# Optimise the tickers from a CSV (writes weights, performance, and plots)
pysharpe optimise --portfolio tests/data/sample_portfolio.csv --export-dir outputs/

# Run an offline DCA projection and optionally emit a chart
pysharpe simulate-dca --initial 1000 --monthly 200 --rate 0.08 --months 36 --plot

# Plot any metric saved by the optimiser
pysharpe plot --input outputs/demo_performance.txt --metric sharpe_ratio
```

Run `pysharpe --help`, `pysharpe optimise --help`, etc. for the canonical flag list once the subcommands are enabled.

## Plotting examples

### Dollar-cost averaging

```python
from pysharpe.visualization import plot_dca_projection, simulate_dca

projection = simulate_dca(
    months=36,
    initial_investment=5_000,
    monthly_contribution=500,
    annual_return_rate=0.12,
)

ax = plot_dca_projection(projection)
ax.figure.savefig("dca_projection.png")
```

### Portfolio performance diagnostics

```python
from pysharpe import metrics

# Assume "collated" is a DataFrame with daily prices for your chosen tickers.
returns = metrics.compute_returns(collated)
expected = metrics.expected_return(returns)
volatility = metrics.annualize_volatility(returns)
sharpe = metrics.sharpe_ratio(returns)
```

## Metrics API highlights

| Function | Description |
| --- | --- |
| `compute_returns` | Convert price levels to simple or log returns. |
| `expected_return` | Compute annualised arithmetic returns. |
| `annualize_return` | Produce a geometric annualised rate. |
| `annualize_volatility` | Scale standard deviation to the desired frequency. |
| `sharpe_ratio` | Evaluate risk-adjusted performance with an optional risk-free rate. |

All helpers accept pandas Series/DataFrames, return matching shapes, and guard against infinities, NaNs, and zero-volatility cases with friendly errors.

## Contributing & Testing

1. Install the development extras: `pip install -e .[dev]`.
2. Run the automated tests with coverage: `python -m pytest`.
3. Lint and format before opening a PR: `ruff check .` and `ruff format .`.
4. If you add functionality, include a brief example or doctest in the corresponding docstring.

The test suite aims for 80% line coverage across the core modules.

## Changelog & Roadmap

- **0.1.0** – Initial packaging with data ingestion, optimisation, and DCA tools.
- **Upcoming** – CLI subcommands (`optimise`, `simulate-dca`, `plot`), richer documentation (tutorial notebooks, API reference), and performance benchmarks.

See `CHANGELOG.md` for detailed release notes as they land.
