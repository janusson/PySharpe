---
name: pysharpe-backtesting
description: >-
  PySharpe backtesting and time-series analysis. Invoke when implementing
  or modifying: backtest engine (calendar/drift-based rebalancing),
  walk-forward optimization, GARCH volatility forecasting (arch package),
  VAR modeling, ADF stationarity tests, benchmark construction and
  comparison, transaction cost modeling, strategy scoring, or backtest
  visualization. Uses synthetic data with fixed seeds in tests — no live
  network calls. Do NOT invoke for live allocation/rebalancing logic or
  data fetching — those are separate skill domains.
---

# PySharpe Backtesting & Time-Series Analysis

This skill owns all **stateful temporal orchestration**: walk-forward loops,
calendar rebalancing state machines, drift tracking over time, and rolling
window evaluations. It does NOT perform stateless metric calculations — those
belong to `pysharpe-metrics`.

## Module Locations

- `src/pysharpe/analysis/backtest_engine.py` — Calendar and drift-based
  rebalancing backtests.
- `src/pysharpe/analysis/time_series.py` — ADF stationarity, GARCH volatility
  forecasting, VAR modeling.
- `src/pysharpe/analysis/benchmarks.py` — Benchmark construction.
- `src/pysharpe/analysis/scoring.py` — Strategy scoring utilities.
- `src/pysharpe/analysis/categorization.py` — Correlated ticker grouping.
- `src/pysharpe/analysis/visualization.py` — Backtest result visualization.

## Backtest Engine

The backtest engine simulates portfolio evolution over historical data:

### Calendar Rebalancing
- Fixed-interval rebalancing (e.g., monthly, quarterly).
- On each rebalance date: compute target weights (via optimizer or VA
  allocator), execute trades, apply transaction costs, update portfolio state.
- Tracks: portfolio value over time, cash flows, turnover, drawdowns.

### Drift-Based Rebalancing
- Rebalance triggers when portfolio weights drift beyond tolerance bands
  (e.g., ±5% from target).
- Requires state tracking: last rebalance date, current drift per asset.

### Walk-Forward Optimization
- Rolling training window → optimize → test on out-of-sample period → advance
  window.
- Stateful: must track window boundaries, in-sample/out-of-sample split,
  and cumulative performance across folds.

### Transaction Costs
- Model bid-ask spread, commission, and market impact.
- Applied per-trade during rebalancing events.
- Must not leak future cost information into current decisions.

## Time-Series Analysis

### ADF Stationarity Test
- Augmented Dickey-Fuller test on return/price series.
- Returns test statistic, p-value, and critical values.
- Used to determine if differencing is needed.

### GARCH Volatility Forecasting
- Uses the `arch` package (`arch_model`).
- Models conditional volatility clustering.
- Input: return series. Output: forecasted volatility for next N periods.

### VAR Modeling
- Vector Autoregression for multi-asset return dynamics.
- Lag order selection via AIC/BIC.
- Impulse response functions and forecast error variance decomposition.

## Benchmarks

`analysis/benchmarks.py` provides benchmark portfolio construction:
- Equal-weight portfolio.
- Market-cap-weight proxy.
- 60/40 stock/bond benchmark.

## Scoring

`analysis/scoring.py` provides shared scoring utilities:
- Sharpe-based ranking.
- Drawdown-based penalty scoring.
- Combined multi-factor scores.

Used by both backtests and live allocation evaluation.

## Stateful vs. Stateless Boundary

| Responsibility | Skill |
|---------------|-------|
| Walk-forward loop state machine | `pysharpe-backtesting` (here) |
| Calendar rebalancing state tracking | `pysharpe-backtesting` (here) |
| Drift tracking over time | `pysharpe-backtesting` (here) |
| Sharpe ratio from return series | `pysharpe-metrics` |
| Max drawdown from equity curve | `pysharpe-metrics` |
| Covariance matrix from returns | `pysharpe-metrics` |

The backtest engine calls metric functions as a stateless library. Metrics
never contain temporal state or rebalancing logic.

## Testing

Relevant test files:
- `tests/test_analysis_backtest_engine.py`
- `tests/test_analysis_transaction_costs.py`
- `tests/test_analysis_walk_forward.py`
- `tests/test_analysis_time_series.py`
- `tests/test_analysis_benchmarks.py`
- `tests/test_analysis_visualization.py`
- `tests/test_analysis.py`
- `tests/test_categorization.py`
- `tests/test_backtest_page.py`

Run: `uv run pytest tests/test_analysis_backtest_engine.py tests/test_analysis_time_series.py tests/test_analysis_walk_forward.py`

All tests must use synthetic data with fixed seeds. No network calls.
