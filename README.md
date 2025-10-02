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

## Graphical interface

For an interactive experience PySharpe includes a Tkinter-based desktop
application. Launch it after installation with:

```bash
pysharpe-gui
```

The GUI allows you to:

* Paste or load a CSV containing ticker symbols, then download their price
  histories to `data/price_hist/` (or a directory of your choice).
* Collate the downloaded histories into a single portfolio CSV.
* Optimise an existing collated portfolio and export the resulting weights and
  performance summaries.

If you prefer not to install the package, the interface can also be started via
`python -m pysharpe.gui` from the repository root. Tkinter is bundled with most
Python distributions; if it is missing, install the platform-specific packages
that provide it before launching the GUI.

## Command line interface

PySharpe also ships with a CLI that wraps the two major workflows: collecting
price histories and optimising the resulting portfolios. Once installed
(for example via `pip install -e .[dev]` during development), the `pysharpe`
command becomes available and exposes the following subcommands:

```text
pysharpe
├── download  # fetch ticker history and collate per-portfolio CSV files
└── optimize  # run max-Sharpe optimisation against collated data
```

### Downloading market data

```bash
# Pull price history for the default portfolios defined under data/portfolio/
pysharpe download --interval 1d --period 1y

# Constrain the window explicitly
pysharpe download --start 2020-01-01 --end 2024-01-01

# Target specific portfolio files (name or path) and enable logging
pysharpe download growth.csv income --log-dir logs
```

`--start` and `--end` accept ISO-8601 dates. When neither is supplied the CLI
forwards `--period` as-is to `yfinance`, mirroring the original script
behaviour. The downloaded ticker CSV files land in `data/price_hist/` (or the
directory supplied via `--price-dir`), and collated portfolio CSVs are written
to `data/exports/` unless overridden with `--export-dir`.

### Optimising portfolios

```bash
# Optimise every collated CSV under data/exports/
pysharpe optimize

# Skip plots and only optimise assets from 1995 onwards
pysharpe optimize --start 1995-01-01 --no-plot

# Focus on specific portfolios and write artefacts elsewhere
pysharpe optimize tech_dividend --collated-dir data/exports --output-dir results
```

The optimisation command always expects collated CSV files (produced by the
download step). It generates weight allocations (`*_weights.txt`), performance
summaries (`*_performance.txt`), and optional allocation plots for each
portfolio.

### Alternative invocation methods

You can always inspect additional options with `pysharpe download --help` or
`pysharpe optimize --help`.

If you prefer not to install the package, invoke the CLI module directly:

```bash
python -m pysharpe.cli download
python -m pysharpe.cli optimize --no-plot
```

From a checked-out repository you can also target the file path explicitly:

```bash
python src/pysharpe/cli.py download --portfolio-dir data/portfolio
```

By default the commands mirror the original scripts by reading from
`data/portfolio/`, writing per-ticker histories to `data/price_hist/`, and
saving collated/optimisation artefacts in `data/exports/`. Pass
`--portfolio-dir`, `--price-dir`, `--collated-dir`, or `--output` to work with
custom locations, and add `--start`/`--end` to constrain the time window.
Run `pysharpe --help` for the full set of options.

## Workflow overview

1. Define portfolios as newline-delimited ticker lists under `data/portfolio/`.
2. Run `pysharpe download` (or call
   `pysharpe.data_collector.process_all_portfolios`) to download individual
   ticker histories and collate them per portfolio.
3. Run `pysharpe optimize` (or call
   `pysharpe.portfolio_optimization.optimise_all_portfolios`) to compute
   Sharpe-maximising weights and performance metrics.
4. Review artefacts under `data/price_hist/`, `data/exports/`, and `logs/` as
   needed.

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
