# PySharpe

PySharpe is a Python toolkit for building, analyzing, and visualizing investment
portfolios that maximize the Sharpe ratio. The project combines data pulled from
Yahoo Finance (via [yfinance](https://pypi.org/project/yfinance/)) with the optimization tooling provided by
[PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) to help investors and
researchers evaluate asset allocations.

## Project structure

```text
PySharpe/
├── docs/                 # Design notes and supplementary documentation
├── src/pysharpe/         # Library source code (packaged with the src-layout)
│   ├── data/             # Market data ingestion helpers
│   ├── optimization/     # Portfolio optimization logic
│   └── visualization/    # Plotting utilities for analytics
├── tests/                # Automated tests
├── pyproject.toml        # Packaging and tooling configuration
└── README.md             # This file
```

## Getting started

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the package in editable mode along with dev dependencies:**

   ```bash
   pip install -e .[dev]
   ```

3. **Run the test suite** to verify the environment:

   ```bash
   pytest
   ```

## Command line interface

Once installed, the `pysharpe` command becomes available with subcommands that
coordinate data downloads and portfolio optimisation:

```bash
# Download price history for portfolios defined under data/portfolio/
pysharpe download --start 2020-01-01 --interval 1d

# Optimise those portfolios, writing collated prices, weights, and performance
pysharpe optimize
```

By default the commands mirror the original scripts by reading from
`data/portfolio/`, writing per-ticker histories to `data/price_hist/`, and
saving collated/optimisation artefacts in `data/exports/`. Pass
`--portfolio-dir`, `--price-dir`, `--collated-dir`, or `--output` to work with
custom locations. Run `pysharpe --help` for the full set of options.

## Usage example

```python
from datetime import datetime

from pysharpe.data import data_collector
from pysharpe.optimization import portfolio_optimization

# Collect price histories, collate them, and persist the CSV files
data_collector.process_all_portfolios(start=datetime(2020, 1, 1))

# Optimise the collated portfolios and export weights/performance summaries
portfolio_optimization.optimize_all_portfolios()
```

More documentation and notebook examples will be added as the project evolves.

## Contributing

Please open an issue or submit a pull request with proposed changes. Make sure
that any added code paths are covered by tests where possible and that linters
(`ruff`, or any additional linters you rely on) has been run locally.
