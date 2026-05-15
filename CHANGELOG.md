# Changelog

## v0.3.0 (unreleased)

- Added dynamic Foreign Exchange (FX) adjustment helper to convert multi-currency portfolio data into a common base currency (e.g., USD to CAD) before optimization.
- Added Bayesian Portfolio Optimization using `PyMC` (`BayesianOptimizer`) for estimating robust posterior distributions of asset returns.
- Added Time-Series Analysis tools (`statsmodels`, `arch`) including ADF stationarity testing (`check_stationarity`), GARCH volatility forecasting (`GARCHVolatilityForecaster`), and Vector Autoregression (`VARModeler`).
- Added Causal Inference and Data Linkage module using an embedded `DuckDB` database (`DataLinker`) to perform high-performance SQL window functions and join market data with external macro datasets.
- Added `pymc`, `statsmodels`, `arch`, and `duckdb` to the project's core dependencies.
- Split dependencies into modular `cli`, `gui`, `dev`, and `all` groups in `pyproject.toml` to support minimal core installations.
- Removed legacy `requirements.txt` and `requirements-dev.txt` in favor of `pyproject.toml` and `uv.lock`.
- Upgraded the portfolio optimiser to use **Exponential Moving Average (EMA)** for expected returns by default, making allocations more responsive to recent market trends.
- Enforced `scikit-learn` as a core dependency to guarantee the use of **Ledoit-Wolf Covariance Shrinkage**, which dramatically improves out-of-sample portfolio stability.
- Added `--return-model` CLI flag (choices: `ema`, `mean`) to allow toggling between the new EMA math and the legacy arithmetic mean.
- The `optimise` command now automatically detects and loads `portfolio_config.json` if it exists in the current working directory.
- The optimiser now gracefully drops geographic `lower_bound` constraints for portfolios lacking assets in those sectors, preventing "infeasible solver" crashes when processing multiple distinct portfolios.
- Fixed a pandas 2.2+ deprecation warning/crash related to `groupby(axis=1)`.
- Fixed a timezone deprecation warning (`datetime.utcnow()`) across internal logging utilities.
- Fixed a bug where `read_tickers` failed to process structured CSV files (like `current_state.csv`) by treating entire rows as single ticker symbols.
- Added `HistoricalBacktester` to simulate chronological portfolio drift and calendar/tolerance-based rebalancing.
- Added `cagr` and `maximum_drawdown` performance metrics.
- Added "Smart Contribution Allocation" tool (`pysharpe allocate`) to recommend cash deployment based on target drift and fundamental valuation (PE, PB, Yield, Momentum).
- Added `pysharpe rebalance`, a user-facing CLI subcommand that loads saved optimisation weights, merges them with current holdings, and prints per-ticker buy amounts and estimated share counts for new cash contributions.
- Documented the rebalance workflow in NumPy-style docstrings and expanded the README with end-to-end usage instructions for CSV and JSON holdings input.
- Added `scripts/export_current_state.py` so users can pull live prices, merge holdings with optimiser weights, and write the CSV required by `pysharpe allocate`.
- Integrated `valueizer.py` into core `execution/allocator.py`.
- Added `CANADIAN_BENCHMARKS` and `fetch_benchmark_metrics` to provide standard all-in-one ETF baselines (VEQT, XGRO, VBAL, etc.) for performance comparison.
- Implemented `generate_efficient_frontier` and `plot_portfolio_comparison` to map and visualize the Efficient Frontier curve overlaid with user portfolios, optimized targets, and benchmarks.
- Added an interactive **Weight Tweak Engine** to the Streamlit dashboard (`app.py`), allowing real-time portfolio manipulation and visualization on the Efficient Frontier.
- Refactored FX conversion logic into a shared utility `apply_fx_conversion` in `src/pysharpe/data/fetcher.py`.

## v0.2.0

- Added argparse-based CLI with `optimise`, `simulate-dca`, and `plot` subcommands.
- Expanded packaging metadata and development extras for ruff/black/pytest.
- Introduced automated CI (GitHub Actions) running lint checks and coverage tests.
- Documented contribution workflow and seeded offline quickstart notebook.
- Implemented MER and geographic exposure constraints for portfolio optimization.
- Added `--config` argument to CLI for loading constraints from JSON.
- Enhanced reporting with Portfolio MER, optimization time window, and limiting ticker identification.
