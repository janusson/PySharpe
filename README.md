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
│   ├── config.py         # Centralised settings and path management
│   ├── data/             # Market data ingestion helpers
│   ├── optimization/     # Dataclasses and optimisation primitives
│   ├── portfolio_optimization.py  # Max-Sharpe engine built on PyPortfolioOpt
│   ├── visualization/    # Plotting utilities (e.g. DCA projections) for analytics
│   └── workflows.py      # High-level orchestration for CLI and notebooks
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
   python3 -m pytest
   ```

## Command line interface

PySharpe ships with a single `pysharpe` command that performs portfolio
discovery, data collection, and optimisation in one pass. Run it from the
project root (or after installing the package):

```bash
pysharpe
```

The CLI reports the directory that holds portfolio definitions, lists the
available portfolios (one CSV per portfolio, with one ticker per line),
downloads the necessary price history, and then optimises every portfolio it
finds. All artefacts are written under the configured `data/` directory by
default:

- `data/portfolio/` – source CSV definitions.
- `data/price_hist/` – raw per-ticker downloads from Yahoo Finance.
- `data/exports/` – collated price history, optimisation weights, performance
  summaries, and plots.

Commonly used flags:

```bash
pysharpe --period 1y --interval 1d           # override download window
pysharpe --portfolio core income             # limit to specific portfolios
pysharpe --skip-download                     # reuse previously downloaded data
pysharpe --skip-optimize                     # only refresh the price history
pysharpe --no-plot                           # skip allocation pie charts
pysharpe --portfolio-dir custom/portfolio    # change source directory
pysharpe --export-dir results                # write artefacts elsewhere
```

When neither `--start` nor `--end` is provided the downloader requests the
longest history available (`period=max`), so optimisation always considers the
full span of the collated data by default.

> **Note:** Plot generation requires `matplotlib`. Install it via
> `pip install matplotlib` or pass `--no-plot` to skip figure creation on
> minimal environments.

### Typical CLI session

1. Create one CSV per portfolio under `data/portfolio/` (or point
   `--portfolio-dir` at another folder). Use one ticker per line and prefix
   comments with `#` to keep notes out of the imports.
2. Run `pysharpe` to download fresh price history and collate it into
   portfolio-level CSV files. Add `--start`/`--end` when you want a bounded
   window, or `--skip-download` if you only need to re-optimise.
3. Inspect artefacts under `data/price_hist/` and `data/exports/`. The exports
   folder holds collated CSVs, optimisation weights, performance summaries, and
   optional allocation plots.
4. Review the generated metadata JSON files (one per portfolio) to confirm
   which tickers were dropped due to missing data, then iterate on the CSVs or
   rerun the command with adjusted parameters.

### Dollar-cost averaging projections

PySharpe also ships a reusable simulator for dollar-cost averaging scenarios
under `pysharpe.visualization`. You can generate projections and optional plots
directly from Python:

```python
from pysharpe.visualization import simulate_dca, plot_dca_projection

projection = simulate_dca(
    months=24 * 12,
    initial_investment=20_000,
    monthly_contribution=1_750,
    annual_return_rate=0.17,
)

ax = plot_dca_projection(projection)
ax.figure.savefig("dca_projection.png")
```

This mirrors the exploratory scripts that previously lived under `drafts/` and
keeps the logic importable for notebooks or other analysis pipelines.

### Alternative invocation methods

Run `pysharpe --help` to see every option. If you would rather call the module
directly (for example before installing the package), execute:

```bash
python -m pysharpe.cli
```

You can always point the CLI at alternative directories with
`--portfolio-dir`, `--price-dir`, and `--export-dir`. Supplying `--start` or
`--end` narrows the download window; omitting both keeps Yahoo Finance's full
history and therefore optimises across the longest possible span.

## Workflow overview

1. Define portfolios as newline-delimited ticker lists under `data/portfolio/`.
2. Use the CLI or the programmatic helpers below to download prices and collate
   them into `data/exports/<portfolio>_collated.csv`.
3. Optimise the collated data to produce allocation weights, performance
   summaries, and optional plots. Reuse `--skip-download` when only the
   optimisation needs updating.
4. Review artefacts under `data/price_hist/`, `data/exports/`, and `logs/` as
   needed.

## Programmatic workflows

These helpers mirror the CLI pipeline and are importable for notebooks or
automation scripts:

- `pysharpe.data_collector.download_portfolio_prices()` – fetch raw ticker
  histories and write one CSV per symbol.
- `pysharpe.data_collector.collate_prices()` /
  `pysharpe.data_collector.process_portfolio()` – combine ticker CSVs into a
  portfolio-level file while emitting metadata about dropped tickers.
- `pysharpe.data_collector.process_all_portfolios()` – batch the above for every
  portfolio file in a directory.
- `pysharpe.workflows.download_portfolios()` and
  `pysharpe.workflows.optimise_portfolios()` – higher-level orchestration used
  by the CLI, returning in-memory data frames and optimisation results.
- `pysharpe.data_collector.PortfolioTickerReader` and
  `pysharpe.data_collector.SecurityDataCollector` – conveniences for inspecting
  available tickers or looking up single-symbol fundamentals.

Example end-to-end run from Python:

```python
from datetime import date

from pysharpe.workflows import download_portfolios, optimise_portfolios

# Collect price histories, collate them, and persist the CSV files
download_portfolios(start=date(2020, 1, 1).isoformat())

# Optimise the collated portfolios and export weights/performance summaries
optimise_portfolios(make_plot=False)
```

## Contributing

Please open an issue or submit a pull request with proposed changes. Ensure
new code paths include tests where practical, then run:

```bash
python3 -m pytest
```

before opening the PR. Finally, format and lint using your preferred tooling
(`ruff`, `black`, etc.) to keep the codebase consistent.
<!-- Change: Expanded contribution checklist during documentation sweep. -->
