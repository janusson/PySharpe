# PySharpe

PySharpe is a portfolio research toolkit that wraps data collection, return analytics, optimisation, and simulation workflows into a single Python package. The repository ships a Streamlit dashboard for exploratory analysis, a CLI for scripted runs, and a modular library for notebooks or downstream automation.

## Features

- Download and clean market data from Yahoo Finance or local CSV files.
- Compute annualised return, volatility, and Sharpe ratio statistics.
- Optimise weights with `pypfopt` using an Efficient Frontier model.
- Run dollar-cost averaging (DCA) projections and export the results.
- Interact through an opinionated Streamlit UI or the command-line utilities.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The editable install keeps the CLI, dashboard, and library in sync. The `dev` extras bring in pytest, coverage, and linting tools that match the CI configuration.

## Usage

### Streamlit Dashboard

```bash
streamlit run app.py
```

Upload your own CSV or enter a comma-separated list of tickers. The dashboard will download prices, compute metrics, optimise weights, and plot DCA projections with downloadable CSV exports.

### CLI Example

```bash
pysharpe optimise \
  --portfolio tests/data/sample_portfolio.csv \
  --export-dir outputs/ \
  --risk-free-rate 0.02
```

The command writes optimised weights, summary metrics, and diagnostic plots to the specified export directory. Run `pysharpe --help` for the full list of subcommands and options.

### Library Example

```python
import pandas as pd
from pysharpe import metrics
from pysharpe.optimization import estimate_max_sharpe_weights

prices = pd.read_csv("tests/data/sample_prices.csv", index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()
mu = metrics.expected_return(returns)
cov = returns.cov() * 252
weights = estimate_max_sharpe_weights(mu=mu, cov_matrix=cov)
print("Optimised allocations:", weights.clean_weights())
```

All analytics functions accept pandas Series/DataFrames and return aligned structures so they compose naturally with your existing research workflow.

## Contributing

1. Create an isolated environment and install the project with dev extras: `pip install -e .[dev]`.
2. Format and lint prior to committing: `ruff format . && ruff check .`.
3. Write or update tests alongside any behavioural change.
4. Document new public APIs or workflows in docstrings and, when appropriate, in the README.

## Testing

```bash
pytest
```

The suite includes unit tests for metrics, optimisation, data collation, and workflow helpers. Add regression tests whenever you touch analytics code, and prefer fixtures over network requests.

## License

PySharpe is distributed under the MIT License. See `LICENSE` for details.
