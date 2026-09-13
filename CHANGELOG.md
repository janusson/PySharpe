# Changelog

## v1.0.0 (2026-08-19)

First production release.

### Developer Infrastructure

- **PEP 735 dependency groups**: dev tooling moved from the legacy `dev` extra to `[dependency-groups]` — uv ≥ 0.10 no longer installs a `dev` extra during `uv run`/`uv sync`, so `make install` is now `uv sync` (project + dev group from `uv.lock`). Published extras (`cli`/`gui`/`all`) are unchanged for consumers.
- **Makefile**: `install`, `lint` (ruff check + format), `typecheck` (pyright with warnings fatal), `test` (pytest with a 75% coverage floor), `build_docs` (MkDocs `--strict`), `clean` (artefacts, caches, and local data directories — `.gitkeep` preserved), and `check` (full pre-commit gate).
- **CI** (`.github/workflows/ci.yml`): three parallel jobs — lint+typecheck, tests with coverage, and a strict docs build — on every push and pull request.
- **Strict typing**: pyright configured with warnings-as-errors; the codebase holds 0 errors / 0 warnings.
- **Docs**: MkDocs Material + mkdocstrings API reference (`docs/api.md`), test map, gotchas log, and the Astro showcase article (`docs/articles/pysharpe_showcase.mdx`).

### Mathematical Core Hardening

- **Covariance estimators** (`optimization/estimators.py`): analytical Ledoit-Wolf (2004) linear shrinkage and nonlinear shrinkage (2017/2020), listwise missing-data handling, and `ensure_strictly_psd` eigen-clipping so every output is strictly positive definite; structural failures raise `DataValidationError`.
- **HRP** (`optimization/hrp.py`): zero-variance assets handled explicitly in recursive bisection (variance floors) — no `ZeroDivisionError` possible; covariance inputs validated for symmetry, finiteness, and PSD.
- **Black-Litterman & Bayesian** (`optimization/black_litterman.py`, `optimization/bayesian.py`): posterior covariances hardened to strict PSD; `BayesianOptimizer.optimize_efficient_frontier()` feeds PyPortfolioOpt the posterior (shrunk) covariance — never the sample — with solver failures surfaced instead of silently falling back.

### Validation & Backtesting

- **PurgedKFold** (`validation/resampling.py`): strict embargo separation between every consecutive test-fold pair (trailing remainders excluded, never clamped); autocorrelation-decay gap sizing via `autocorrelation_decay_lag` and `PurgedKFold.from_returns`.
- **DSR** (`validation/metrics.py`): `compute_dsr(..., theta=...)` deflates the observed Sharpe by Lo's √θ autocorrelation factor before deflation; `compute_validation_metrics` computes θ from the returns automatically.
- **Backtest engine** (`analysis/backtest_engine.py`): explicit bid-ask spread (half-spread per side), slippage, and per-order commissions; walk-forward backtests forward cost parameters to every window; missing prices handled by dropping fully-missing columns with a warning then listwise row deletion; costs never use future prices.

### Test Suite

- 995 tests (3 MCMC integration tests skip-gated on compiler availability), all synthetic data with fixed seeds — no network calls. PyMC samplers isolated with `pytest.MonkeyPatch`.
- Edge-case coverage: perfectly/near-perfectly correlated shrinkage, NaN-in-fold walk-forwards, same-day VFV↔VOO superficial-loss trades across TFSA/NON_REG, and ±30-day window boundaries.

### Presentation

- **Harmonized comparison pipeline**: the dashboard's Performance Comparison now evaluates every row — 1/N equal-weight baseline, Custom Mix, PySharpe Optimized, and the six Canadian benchmarks — with a single estimator: Bayes-Stein shrunk expected returns (benchmarks shrunk *jointly with the asset universe*) reduced by MER and account-specific tax drag via `AssetLocationEngine`, with Sharpe ratios recomputed at the 2 % risk-free rate. Adds `compute_adjusted_metrics`/`evaluate_adjusted_performance` (`src/pysharpe/app/analytics.py`) and `BENCHMARK_MERS`/`build_benchmark_characteristics` (`src/pysharpe/analysis/benchmarks.py`); benchmark rows default to Non-Registered placement, stated in the UI caption. The `tax_profile=None` path of `fetch_benchmark_metrics` keeps the legacy raw-metric behavior for external callers.
- **Streamlit dashboard rewrite**: four-tab workspace (Portfolio Metrics & Comparison, Efficient Frontier, DCA Simulation, Raw Data & Logs). Full price history is fetched first and the maximum overlapping date range across tickers becomes the UI defaults (`compute_overlapping_date_range` in `src/pysharpe/app/data.py`); every date/weight widget is keyed by the ticker-set signature so stale state cannot leak between selections; weights default to a strict 1/N allocation with a zero-sum fallback; backtesting and execution/rebalancing are preserved as collapsed expanders under the metrics tab. Fixed the missing `__main__` guard that rendered a blank page on `streamlit run app.py`.
- README and docs overhaul with CI badges, engineering-rigor section, uv-first quickstart, and the `.agents/skills/` agentic-development framework.

## v0.3.0 (development log — superseded by v1.0.0)

### Features

- **Backtesting tab** (`src/pysharpe/app/backtest.py`): Added a "Backtest" tab to the Streamlit dashboard. Users can simulate historical portfolio performance with configurable rebalancing strategies (monthly/quarterly/annual calendar, absolute drift band, relative drift band, or none), transaction fees, and slippage. Results display CAGR, Max Drawdown, Sharpe Ratio, and rebalance count, alongside an interactive equity curve with optional Canadian ETF benchmark overlay (VEQT, XEQT, VGRO, XGRO, VBAL, XBAL) and a stacked area weight-drift chart. Portfolio values can be downloaded as CSV.
- **Backtest engine hardening** (`src/pysharpe/analysis/backtest_engine.py`): Added `_PERIOD_ALIAS_MAP` to translate newer pandas offset aliases (`ME`/`QE`/`YE`) to period aliases (`M`/`Q`/`Y`) required by `DatetimeIndex.to_period()` on pandas ≥ 2.2. Added a zero-portfolio-value guard to break the simulation loop cleanly if all asset prices reach zero, preventing a division-by-zero.
- **FX conversion utility** (`src/pysharpe/data/fetcher.py`): Refactored FX adjustment into a shared `apply_fx_conversion` helper that converts multi-currency portfolio prices to a common base currency (e.g., USD to CAD) before optimisation.
- **Bayesian Portfolio Optimization** (`src/pysharpe/optimization/bayesian.py`): Added `BayesianOptimizer` using PyMC to estimate posterior distributions of asset returns, producing Black-Litterman compatible expected returns and covariance.
- **Time-Series Analysis** (`src/pysharpe/analysis/time_series.py`): Added ADF stationarity testing (`check_stationarity`), GARCH volatility forecasting (`GARCHVolatilityForecaster`), and Vector Autoregression (`VARModeler`) via `statsmodels` and `arch`.
- **Causal Inference & Data Linkage** (`src/pysharpe/data/`): Added `DataLinker` using an embedded DuckDB database for high-performance SQL window functions, lagging, and joining of market data with external macro datasets.
- **Smart Contribution Allocation** (`src/pysharpe/execution/allocator.py`): Added `pysharpe allocate` CLI subcommand to recommend cash deployment based on target drift and fundamental valuation (PE, PB, Yield, Momentum).
- **Rebalance CLI** (`src/pysharpe/execution/rebalance.py`): Added `pysharpe rebalance` subcommand that loads saved optimisation weights, merges them with current holdings (CSV or JSON), and prints per-ticker buy amounts and estimated share counts for new capital.
- **Export current state script** (`scripts/export_current_state.py`): Fetches live prices, merges holdings with optimiser weights, and writes the CSV required by `pysharpe allocate`.
- **Canadian ETF benchmarks** (`src/pysharpe/analysis/benchmarks.py`): Added `CANADIAN_BENCHMARKS` dict and `fetch_benchmark_metrics` to provide VEQT, XEQT, VGRO, XGRO, VBAL, and XBAL baselines for performance comparison.
- **Efficient Frontier visualisation** (`src/pysharpe/visualization/`): Added `generate_efficient_frontier` and `plot_portfolio_comparison` to map and render the Efficient Frontier overlaid with user portfolios, optimised targets, and benchmarks.
- **Weight Tweak Engine** (`app.py`): Added interactive sidebar sliders to the Streamlit dashboard for real-time portfolio manipulation visualised on the Efficient Frontier.
- **EMA expected returns**: Upgraded the portfolio optimiser to use Exponential Moving Average for expected returns by default, making allocations more responsive to recent market regimes. Override via `--return-model mean`.
- **Ledoit-Wolf covariance shrinkage**: Enforced `scikit-learn` as a core dependency to guarantee Ledoit-Wolf shrinkage, which improves out-of-sample portfolio stability.
- **Modular dependency groups**: Split dependencies into `cli`, `gui`, `dev`, and `all` extras in `pyproject.toml`. Removed legacy `requirements.txt` and `requirements-dev.txt`.
- **CAGR and Maximum Drawdown metrics** (`src/pysharpe/metrics.py`): Added compound annual growth rate, maximum drawdown, Sortino ratio, Calmar ratio, tracking error, and max drawdown duration to the metrics library.
- **`--return-model` CLI flag**: Added `ema`/`mean` toggle to `pysharpe optimise` for switching expected return models.
- **Auto-load `portfolio_config.json`**: The `optimise` command now detects and loads `portfolio_config.json` from the working directory automatically. Override with `--config`.
- **Head-to-Head Fund Comparison** (`src/pysharpe/analysis/comparison.py`): Added `compare_two_funds()` for side-by-side risk/return evaluation of any two assets using the full data pipeline (YFinance → DuckDB cache → collation). Outputs CAGR, annualized volatility, max drawdown depth and duration, Sharpe ratio, Sortino ratio, Calmar ratio, 1-year rolling tracking error, and 1-year rolling return correlation in a single DataFrame. No multi-asset optimizer or VA allocation invoked.

### Bug Fixes

- **MER unit consistency**: Removed a double-division bug where MER values supplied as decimal fractions (e.g. `0.0017` for 0.17%) were being divided by 100 a second time inside `SharpeOptimizer` and `optimise_portfolio_for_sharpe`, causing reported costs to be 100× smaller than intended. The default VEQT MER in `config.py` has also been corrected from `0.17` (percentage point) to `0.0017` (decimal fraction) to match the documented convention.
- **FX lookahead bias**: `apply_fx_conversion` was calling `.bfill()` after `.ffill()` on aligned exchange-rate series, which applied future FX rates to historical prices and introduced lookahead bias. Back-fill has been removed. Leading rows where FX data is not yet available are now detected, logged as a warning, and excluded from the price history to preserve temporal integrity. If all rows are excluded (FX history does not overlap the price window at all) a `ValueError` is raised with a clear message. A tz-normalisation guard was also added so callers can pass tz-aware price DataFrames without triggering a `TypeError`.
- **Stale LRU cache**: `_cached_collated_prices` was keyed on `(portfolio_name, collated_dir, time_constraint)` only, so re-downloading a collated CSV within the same Python process silently returned the pre-download snapshot. File modification time is now included in the cache key: `_load_collated_prices` reads `csv_path.stat().st_mtime` (via `try/except FileNotFoundError`) and passes it as an extra `mtime: float` argument, causing a cache miss whenever the file is overwritten.
- **DuckDB wrapping of custom fetchers**: `CollationService` was wrapping every non-`DuckDBCachedPriceFetcher` fetcher in the write-through DuckDB cache, including test stubs and custom implementations. This meant test stubs were bypassed when the global cache already held data for the requested ticker. DuckDB wrapping is now applied only to `YFinancePriceFetcher` instances, where network-call caching is appropriate.
- **Infeasible geo constraints**: The optimiser now gracefully drops geographic `lower_bound` constraints for portfolios that contain no assets mapped to that region, preventing "infeasible solver" crashes.
- **pandas 2.2+ `groupby(axis=1)` deprecation**: Fixed a warning/crash affecting the collation layer on pandas ≥ 2.2.
- **`datetime.utcnow()` deprecation**: Fixed a timezone deprecation warning across internal logging utilities.
- **`read_tickers` CSV parsing**: Fixed a bug where structured CSV files (e.g. `current_state.csv`) were treated as single-column ticker files, causing entire rows to be read as ticker symbols.
- **Infeasible `max_weight` on small portfolios**: The default `max_weight=0.20` constraint is mathematically impossible for portfolios with ≤ 4 assets (e.g. 4 × 0.20 = 0.80 < 1.0). Previously this raised a hard `ValueError` blocking optimization entirely. Now `optimise_from_prices` and `optimise_portfolio_for_sharpe` auto-adjust `max_weight` to `1.0 / n_assets` (the minimum feasible cap) with a warning instead of crashing.

### Security

- Upgraded `starlette` from 1.0.0 to 1.2.1 (transitive dependency via `streamlit`) to address a missing Host header validation vulnerability (Dependabot #29).
- Upgraded `soupsieve` from 2.8.3 to 2.8.4 (transitive dependency via `beautifulsoup4`) to address two vulnerabilities: ReDoS via unterminated CSS selector quotes (CVE-2026-49477) and memory exhaustion via unbounded comma-separated selector lists (CVE-2026-49476).

## v0.2.0

- Added argparse-based CLI with `optimise`, `simulate-dca`, and `plot` subcommands.
- Expanded packaging metadata and development extras for ruff/black/pytest.
- Introduced automated CI (GitHub Actions) running lint checks and coverage tests.
- Documented contribution workflow and seeded offline quickstart notebook.
- Implemented MER and geographic exposure constraints for portfolio optimization.
- Added `--config` argument to CLI for loading constraints from JSON.
- Enhanced reporting with Portfolio MER, optimization time window, and limiting ticker identification.
