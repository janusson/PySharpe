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
