# PySharpe

PySharpe is a portfolio research toolkit that wraps data collection, return analytics, optimisation, and simulation workflows into a single Python package. The repository ships a Streamlit dashboard for exploratory analysis, a CLI for scripted runs, and a modular library for notebooks or downstream automation.

## Features

- Download and clean market data from Yahoo Finance or local CSV files.
- Compute annualised return, volatility, and Sharpe ratio statistics.
- **Bayesian Portfolio Optimization**: Estimate posterior distributions of asset returns using PyMC to build robust, Black-Litterman compatible models.
- **Time-Series Analysis**: Test for stationarity (ADF), forecast volatility clusters using GARCH models, and capture asset interdependencies with Vector Autoregression (VAR).
- **Causal Inference & Data Linkage**: Use an embedded DuckDB database to perform high-performance SQL window functions, lagging, and joining of market data with external macro-economic datasets.
- Optimise weights with `pypfopt` using an Efficient Frontier model, applying optional MER and geographic constraints.
- Run historical portfolio backtests with calendar and drift-based rebalancing logic.
- Calculate smart cash allocations to correct portfolio drift while factoring in fundamental valuation.
- Run dollar-cost averaging (DCA) projections and export the results.
- Interact through an opinionated Streamlit UI or the command-line utilities.

## Installation

PySharpe's dependencies are modular, so you only install what you need:

- **Core Library** (minimal install for data and math): `pip install -e .`
- **CLI Tools** (adds visualization helpers like `matplotlib`): `pip install -e .[cli]`
- **Web Dashboard** (adds `streamlit` and `altair`): `pip install -e .[gui]`
- **CLI and GUI (default)** (everything needed for regular use): `pip install -e .[all]`

To set up a complete environment for development (includes tests and linters):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart (The "Golden Path")

PySharpe's most powerful workflow bridges theory and execution in two steps:

1. **Research & Target Setting**: Generate optimal weights based on historical performance.
   ```bash
   pysharpe optimise --portfolio demo --export-dir outputs/
   ```
   *Produces: `outputs/demo_weights.txt` and `outputs/demo_collated.csv`.*

2. **Smart Execution**: Generate a buy plan for new capital using your current holdings.
   ```bash
   pysharpe rebalance \
     --portfolio demo \
     --holdings-json '{"AAPL": 2, "MSFT": 1}' \
     --new-cash 1000 \
     --export-dir outputs/
   ```

*PySharpe calculates how much you are "drifting" from your targets and blends it with fundamental valuation to tell you exactly how many shares to buy.*

## Professional Usage & Best Practices

### 1. The Collaborative Research Workflow
For best results, use the **Streamlit Dashboard** (`app.py`) for visual exploration and "Category Grouping" to handle correlated assets (e.g., grouping VOO and VFV). Once you are satisfied with a portfolio mix, transition to the **CLI** for recurring rebalancing and scripted updates.

### 2. Constraints and MER Management
Serious investors should use a `portfolio_config.json` to enforce structural discipline:
- **MER Caps**: Prevent your portfolio from becoming "expensive" by capping the weighted average expense ratio.
- **Geo Limits**: Ensure you aren't over-exposed to a single region (e.g., max 60% US).
- **Valuation Overlay**: Use the `--config` flag during `allocate` or `rebalance` to weight buys toward assets with lower P/E or higher Dividend Yield.

## Interpreting Your Results

### Portfolio Analytics
- **Sharpe Ratio**: The primary efficiency metric. A higher Sharpe means better risk-adjusted returns. Target > 1.0 where possible.
- **Annual Volatility**: A measure of the "bumpiness" of the ride. Use this to ensure the portfolio aligns with your risk tolerance.
- **Expected Return (EMA)**: By default, PySharpe uses an Exponential Moving Average for returns, which weights recent history more heavily. This makes the model more responsive to current market regimes.

### Rebalancing Metrics
- **Drift (Underweight %)**: The percentage points an asset is below its target. Higher drift indicates a stronger "need" to buy.
- **Valuation Score (0-1)**: A multi-factor score blending P/E, P/B, Dividend Yield, and Momentum. A score of 1.0 represents a "perfect" fundamental setup.
- **Opportunity Score**: The final ranking tool. It blends **Drift** (60%) and **Valuation** (40%).
    - **Score > 0.8**: Strong Buy. The asset is significantly underweight and fundamentals are attractive.
    - **Score < 0.3**: Low Priority. The asset is likely near its target or fundamentals are poor.

## Usage

### Streamlit Dashboard

To use the interactive web interface, ensure you have installed the package with the `gui` (or `all`) dependencies:

```bash
# If using standard pip:
pip install -e .[gui]
streamlit run app.py

# If using uv:
uv pip install -e .[gui]
uv run streamlit run app.py
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

### Category Grouping

Highly correlated tickers can be collapsed into broader economic exposures before optimisation. Create a JSON mapping of ticker to category (for example `{"VOO": "US Large Cap", "IEF": "Bonds"}`) and store it at `data/info/asset_categories.json` or pass it to the CLI via `--category-map`. The Streamlit dashboard exposes the same feature under **Category Grouping** in the sidebar. Tickers without an explicit category can either be kept as standalone exposures or dropped entirely.

### Portfolio Constraints & Mathematical Models

You can enforce MER (Management Expense Ratio) caps and geographic exposure limits by providing a JSON configuration file. If a file named `portfolio_config.json` exists in the directory where you run `pysharpe`, the CLI will detect and load it automatically. You can also explicitly pass a file via `--config`.

Example `portfolio_config.json`:

```json
{
  "mer_mapping": {
    "VFV.TO": 0.0009,
    "VCN.TO": 0.0005
  },
  "geo_mapping": {
    "VFV.TO": "US",
    "VCN.TO": "CA"
  },
  "constraints": {
    "max_portfolio_mer": 0.0015,
    "geo_upper_bounds": {
      "US": 0.6,
      "CA": 0.4
    },
    "geo_lower_bounds": {
      "US": 0.1
    }
  }
}
```

*Note on constraints:* The optimiser is intelligent enough to skip geographic lower-bound constraints for portfolios that do not contain any assets mapped to that region, avoiding "infeasible solver" errors.

#### Expected Return Models
By default, PySharpe uses an **Exponential Moving Average (EMA)** to calculate expected returns, making the optimisation model more responsive to recent market trends. You can override this to use standard arithmetic means via the CLI:

```bash
pysharpe optimise --portfolio my_portfolio --return-model mean
```

### Smart Contribution Allocation

If you have fresh capital to invest, PySharpe can recommend how to deploy it. The logic prioritizes assets that have drifted below their target weights, and can optionally factor in fundamental valuation (e.g. buying the "cheaper" underweight assets first).

```bash
pysharpe allocate --portfolio data/portfolio/current_state.csv --amount 1000.0
```

The CSV must already contain the current state plus `ticker`, `current_value`, and `target_weight` columns, so this path is best when you are hand-writing or exporting your own state sheet. Optional columns `pe_ratio`, `pb_ratio`, `div_yield`, and `momentum_6m` deepen the valuation signal.

If you don’t yet have a CSV, the bundled script `scripts/export_current_state.py` combines your holdings and saved weights, fetches the latest prices from Yahoo Finance, and writes the required `ticker,current_value,target_weight` table for you:

```bash
scripts/export_current_state.py \
  --holdings-json '{"AAPL": 2, "MSFT": 1}' \
  --weights outputs/demo_weights.txt \
  --output current_state.csv
pysharpe allocate --portfolio current_state.csv --amount 1000
```

If instead you want a CLI that runs straight from saved optimisation outputs plus your real holdings, use `pysharpe rebalance` (see below Quickstart). That command loads `<portfolio>_weights.txt` and `<portfolio>_collated.csv`, merges them with your CSV or JSON holdings, and then calls the same allocator to display buy dollars and shares.

### Rebalance CLI

Use `pysharpe rebalance` when you want a user-facing buy plan that starts from saved optimisation artefacts instead of manually building a `target_weight` CSV. The command:

1. Loads the latest optimiser weights from `<portfolio>_weights.txt`.
2. Loads the latest prices from `<portfolio>_collated.csv`.
3. Merges those artefacts with your current real-world holdings.
4. Computes the opportunity score with the existing allocator logic.
5. Prints exactly how many dollars and estimated shares to buy for each ticker.

Run the optimiser first so the export directory contains the required artefacts:

```bash
pysharpe optimise --portfolio demo --export-dir outputs/
```

Then provide your current holdings as either a CSV or a JSON mapping.

CSV example:

```csv
ticker,shares
AAPL,2
MSFT,1
```

```bash
pysharpe rebalance \
  --portfolio demo \
  --holdings-csv holdings.csv \
  --new-cash 1000 \
  --export-dir outputs/
```

Inline JSON example:

```bash
pysharpe rebalance \
  --portfolio demo \
  --holdings-json '{"AAPL": 2, "MSFT": 1}' \
  --holdings-kind shares \
  --new-cash 1000 \
  --export-dir outputs/
```

If your holdings are already in dollars instead of shares, either use a CSV with a `current_value` or `total_value` column, or pass JSON with `--holdings-kind value`.

Optional inputs:

- `--config portfolio_config.json` loads `allocation_weights` and optional per-ticker `fundamentals`.
- `--include-zero-buys` shows the full merged portfolio state instead of only positive buy recommendations.

Expected files in `--export-dir`:

```text
<portfolio>_weights.txt
<portfolio>_collated.csv
```

The terminal output includes the latest price, target weight, current weight, opportunity score, recommended dollar buy, and estimated share count for each recommended purchase.

### Advanced Statistical Analysis (Library Only)

PySharpe includes advanced statistical tools designed for use in Jupyter notebooks or automated data pipelines.

**Bayesian Optimization:**
```python
from pysharpe.optimization import BayesianOptimizer

# Fit a PyMC model to historical returns
optimizer = BayesianOptimizer(random_seed=42)
trace = optimizer.fit_returns_model(returns_df, draws=1000, tune=1000)

# Extract expected returns and covariance from the posterior distribution
expected_returns, expected_cov = optimizer.get_posterior_estimates()
```

**Time-Series & Volatility Modeling:**
```python
from pysharpe.analysis import check_stationarity, GARCHVolatilityForecaster, VARModeler

# Check if a return series is stationary (ADF Test)
adf_result = check_stationarity(returns_df["AAPL"])
print(f"Stationary: {adf_result['is_stationary']}")

# Forecast Volatility using GARCH(1,1)
forecaster = GARCHVolatilityForecaster(p=1, q=1).fit(returns_df["AAPL"] * 100)
projected_variance = forecaster.forecast(horizon=5)

# Model interdependencies using Vector Autoregression
var_model = VARModeler(maxlags=5).fit(returns_df)
var_forecast = var_model.forecast(steps=5)
```

**Data Linkage via DuckDB:**
```python
from pysharpe.data import DataLinker

linker = DataLinker()
linker.register_data("market", market_df)
linker.register_data("macro", macro_df)

# Use SQL window functions and lagging to build causal features
enhanced_data = linker.get_enhanced_market_data(
    market_table="market",
    macro_table="macro",
    rolling_window=7
)
linker.close()
```

### Library Example

```python
import pandas as pd
from pysharpe import metrics
from pysharpe.optimization.weights import normalize_weights

prices = pd.read_csv("my_prices.csv", index_col=0, parse_dates=True)
returns = metrics.compute_returns(prices)

summary = pd.DataFrame(
    {
        "expected_return": metrics.expected_return(returns),
        "annual_volatility": metrics.annualize_volatility(returns),
        "sharpe_ratio": metrics.sharpe_ratio(returns),
    }
)

weights = normalize_weights({"AAPL": 0.55, "MSFT": 0.45})
print(summary.round(4))
print("Normalised weights:", weights)
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
