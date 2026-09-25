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

- Signature: `sharpe_ratio(returns, *, risk_free_rate=0.0, periods_per_year=252)`
  — everything after `returns` is keyword-only.
- Input: periodic returns as decimal fractions (simple or log — the caller
  decides). Pass the **raw** return series, not pre-subtracted excess returns.
- Annualization: mean × `periods_per_year`, std × √`periods_per_year`.
- `risk_free_rate` is an **annual** decimal (e.g. `0.02` for 2%). The function
  annualises the return series first and subtracts the rate afterwards:
  `(annualize_return(returns) - risk_free_rate) / annualize_volatility(returns)`.
  **Never pass a daily rate** (`0.02 / 252`): the numerator loses its risk-free
  deduction and the ratio is silently inflated.
  - Pinned by `tests/test_metrics.py::test_sharpe_ratio_handles_risk_free_rate`.

### Sortino Ratio

- **`sortino_ratio(returns, *, risk_free_rate=0.0, periods_per_year=252, target_return=0.0)`**
  — keyword-only, same risk-free convention as `sharpe_ratio`.

$$\text{Sortino} = \frac{\text{Annualised Return} - R_f^{\text{annual}}}{\sigma_{\text{downside}}^{\text{annual}}}$$

- `risk_free_rate` is an **annual** decimal; the function divides it by
  `periods_per_year` internally when building the downside series.
- `target_return` (the MAR) is a **daily** decimal, and the two combine:
  `downside = (returns - (target_return + risk_free_rate / periods_per_year)).clip(upper=0)`.
- Vectorized: `sqrt(mean(downside²)) * sqrt(periods_per_year)`.
- Note the deliberate asymmetry: annual `risk_free_rate`, daily `target_return`.

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
- **`risk_free_rate` is an annual decimal** (`0.02` for 2%) across this module;
  `sortino_ratio` converts it to a daily rate internally for the downside
  calculation. A daily rate must never be passed in.
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
