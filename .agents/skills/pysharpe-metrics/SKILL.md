---
name: pysharpe-metrics
description: >-
  Financial performance metrics for PySharpe portfolio analytics. Invoke when
  implementing, auditing, or testing: Sharpe ratio, Sortino ratio, tracking
  error, maximum drawdown, Calmar ratio, covariance/correlation matrices,
  annualized return/volatility from pandas DataFrames of daily price series,
  risk-free rate adjustments, rolling window metrics, and excess return
  calculations. All operations must use vectorized NumPy/pandas, never
  heuristic approximations. Do NOT invoke for optimization solver logic or
  backtest orchestration — those are separate skill domains.
---

# PySharpe Financial Metrics

This skill covers **stateless, vectorized array and DataFrame evaluations**.
Given one or more return series as input (typically daily log or simple returns
in a `pandas.DataFrame` or `numpy.ndarray`), output scalar metrics or rolling
window series. This skill does NOT perform temporal orchestration, walk-forward
loops, or rebalancing state machines — those belong to `pysharpe-backtesting`.

## Module Location

All metric functions are in `src/pysharpe/metrics.py`.

## Core Metrics

### Sharpe Ratio

$$\text{Sharpe} = \frac{\bar{R} - R_f}{\sigma_R}$$

- Input: Daily excess returns (`returns - risk_free_rate`).
- Annualization: Multiply mean by 252, std by √252 for daily data.
- Vectorized: `(mean * 252) / (std * sqrt(252))` on a 1-D `numpy.ndarray`.
- `risk_free_rate` must be passed as a **daily** decimal (e.g., 0.05/252 for 5%
  annual), never annual.

### Sortino Ratio

- **`sortino_ratio(returns, risk_free_rate, periods_per_year, target_return)`**

$$\text{Sortino} = \frac{\bar{R} - R_f}{\sigma_{\text{downside}}}$$

- Downside deviation uses only returns below a target (default 0 or risk-free).
- Vectorized: `sqrt(mean(min(0, r - target)^2))` with annualization.

### Calmar Ratio

- **`calmar_ratio(value_series)`**

$$\text{Calmar} = \frac{\text{Annualized Return}}{\lvert \text{Max Drawdown} \rvert}$$

- Takes a DatetimeIndex-ed price series. Returns `inf` when there is no drawdown.

### Tracking Error

- **`tracking_error(returns_a, returns_b, periods_per_year)`**

$$\text{TE} = \sigma(R_{\text{a}} - R_{\text{b}})$$

- Annualized: multiply daily tracking error by √252.
- Validates that both return series have equal length.

### Covariance & Correlation Matrices

- `numpy.cov(returns.T)` for covariance, `numpy.corrcoef(returns.T)` for
  correlation on transposed DataFrames where rows are time and columns are
  assets.

## Constraints

- **All operations must be vectorized.** No Python `for` loops over time steps
  in metric calculations.
- **Annualization factor is 252** for daily data (trading days).
- **Risk-free rate is a daily decimal**, never annualized inside the function.
- **Returns are simple or log returns** — the calling code decides. All
  metrics accept the pre-computed return series.
- **NaN handling**: Metrics should use `np.nanmean`, `np.nanstd` or explicitly
  drop NaN periods. Document the behavior per function.

## Scope Boundary

This skill covers ONLY stateless metric computation. It does NOT cover:

- Backtest loops or walk-forward windows → use `pysharpe-backtesting`.
- Portfolio optimization or efficient frontier → use `pysharpe-optimization`.
- Allocation or rebalancing logic → use `pysharpe-allocation`.

## Testing

Run: `uv run pytest tests/test_metrics.py tests/test_analysis_comparison.py`

Tests must use synthetic return data with fixed `numpy.random` seeds. No
network calls. Test edge cases: zero-variance assets, all-negative returns,
single-period series, identical series, and extreme drawdowns.
