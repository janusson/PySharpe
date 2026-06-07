# Changelog

## v0.3.0 (unreleased)

### Bug Fixes

- **MER unit consistency**: Removed a double-division bug where MER values supplied as decimal fractions (e.g. `0.0017` for 0.17%) were being divided by 100 a second time inside `SharpeOptimizer` and `optimise_portfolio_for_sharpe`, causing reported costs to be 100× smaller than intended. The default VEQT MER in `config.py` has also been corrected from `0.17` (percentage point) to `0.0017` (decimal fraction) to match the documented convention.
- **FX lookahead bias**: `apply_fx_conversion` was calling `.bfill()` after `.ffill()` on aligned exchange-rate series, which applied future FX rates to historical prices and introduced lookahead bias. Back-fill has been removed. Leading rows where FX data is not yet available are now detected, logged as a warning, and excluded from the price history to preserve temporal integrity. If all rows are excluded (FX history does not overlap the price window at all) a `ValueError` is raised with a clear message. A tz-normalisation guard was also added so callers can pass tz-aware price DataFrames without triggering a `TypeError`.
- **Stale LRU cache**: `_cached_collated_prices` was keyed on `(portfolio_name, collated_dir, time_constraint)` only, so re-downloading a collated CSV within the same Python process silently returned the pre-download snapshot. File modification time is now included in the cache key: `_load_collated_prices` reads `csv_path.stat().st_mtime` (via `try/except FileNotFoundError`) and passes it as an extra `mtime: float` argument, causing a cache miss whenever the file is overwritten.
- **DuckDB wrapping of custom fetchers**: `CollationService` was wrapping every non-`DuckDBCachedPriceFetcher` fetcher in the write-through DuckDB cache, including test stubs and custom implementations. This meant test stubs were bypassed when the global cache already held data for the requested ticker. DuckDB wrapping is now applied only to `YFinancePriceFetcher` instances, where network-call caching is appropriate.

### Features

- **Backtesting tab** (`src/pysharpe/app/backtest.py`): Added a "Backtest" tab to the Streamlit dashboard. Users can simulate historical portfolio performance with configurable rebalancing strategies (monthly/quarterly/annual calendar, absolute drift band, relative drift band, or none), transaction fees, and slippage. Results display CAGR, Max Drawdown, Sharpe Ratio, and rebalance count, alongside an interactive equity curve with optional Canadian ETF benchmark overlay (VEQT, XEQT, VGRO, XGRO, VBAL, XBAL) and a stacked area weight-drift chart. Portfolio values can be downloaded as CSV.
- **Backtest engine hardening** (`src/pysharpe/analysis/backtest_engine.py`): Added `_PERIOD_ALIAS_MAP` to translate newer pandas offset aliases (`ME`/`QE`/`YE`) to period aliases (`M`/`Q`/`Y`) required by `DatetimeIndex.to_period()` on pandas ≥ 2.2. Added a zero-portfolio-value guard to break the simulation loop cleanly if all asset prices reach zero, preventing a division-by-zero.

### Security

- Upgraded `starlette` from 1.0.0 to 1.2.1 (transitive dependency via `streamlit`) to address a missing Host header validation vulnerability (Dependabot #29).

---

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
