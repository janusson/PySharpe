# PySharpe

<<<<<<< HEAD
PySharpe is a Python toolkit for building, analyzing, and visualizing investment
portfolios that maximize the Sharpe ratio. The project combines data pulled from
Yahoo Finance (via [yfinance](https://pypi.org/project/yfinance/)) with the optimization tooling provided by
=======
PySharpe is a Python toolkit for building, analyzing, and visualizing optimized investment
portfolios. The project combines data pulled from
Yahoo Finance with the optimization tooling provided by
>>>>>>> 70b35dd43fbb272b8fce3aaf2a971c2b80eb402e
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

# Optimise those portfolios, writing weights and performance metrics
pysharpe optimize --output data/exports --collated-dir data/collated
```

Both commands accept `--portfolio-dir` and `--price-dir` switches if you keep
data in custom locations. Run `pysharpe --help` for the full set of options.

## Usage example

```python
from datetime import datetime

from pysharpe import PortfolioOptimizer, fetch_price_history

prices = fetch_price_history(["AAPL", "MSFT", "GOOG"], start=datetime(2020, 1, 1))
optimizer = PortfolioOptimizer(prices)
result = optimizer.max_sharpe()
print(result.allocation.weights)
print(result.performance.as_dict())
```

More documentation and notebook examples will be added as the project evolves.

## Contributing

Please open an issue or submit a pull request with proposed changes. Make sure
that any added code paths are covered by tests where possible and that linters
(`pylint`, `flake8`, `isort`) have been run locally.
