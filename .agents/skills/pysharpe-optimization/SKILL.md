---
name: pysharpe-optimization
description: >-
  PySharpe portfolio optimization: efficient frontier, Bayesian estimation,
  and constraint handling. Invoke when building or modifying optimizers,
  working with PyPortfolioOpt EfficientFrontier wrappers, PyMC-based
  posterior return/covariance estimation, geographic lower-bound constraints
  (drop constraints for absent regions), MER drag deduction (MER values are
  decimal fractions <0.10, never percentage points — no redundant /100),
  tax-location optimization, or frozen dataclass models (PortfolioWeights,
  OptimisationPerformance, OptimisationResult). Do NOT invoke for value
  averaging allocation logic — use pysharpe-allocation for that.
---

# PySharpe Portfolio Optimization

Covers the full optimization stack: efficient frontier construction, Bayesian
return estimation, constraint handling, and model definitions.

## Module Locations

- `src/pysharpe/optimization/sharpe_optimizer.py` — PyPortfolioOpt wrapper.
- `src/pysharpe/optimization/bayesian.py` — PyMC posterior estimation.
- `src/pysharpe/optimization/models.py` — Frozen dataclass models.
- `src/pysharpe/optimization/base.py` — Base optimizer class.
- `src/pysharpe/optimization/weights.py` — Weight constraint utilities.
- `src/pysharpe/optimization/tax_location.py` — Tax-location optimization.
- `src/pysharpe/portfolio_optimization.py` — Top-level orchestration.

## Sharpe Optimizer

`SharpeOptimizer` wraps PyPortfolioOpt's `EfficientFrontier` with PySharpe
extensions:

### MER Drag

- MER values are **decimal fractions** (e.g., `0.0017` for 0.17%).
- Applied directly as an annualized drag on expected returns: `mu_adjusted = mu - mer`.
- **Never divide by 100** — the values are already decimals.

### Geographic Constraints

- Supplied as `geo_lower_bounds` / `geo_upper_bounds` mappings (the CLI reads them from the `constraints` object in `portfolio_config.json`).
- Each configured region has a lower or upper bound (0.0–1.0).
- **Critical**: Drop lower-bound constraints for regions that contain no mapped
  assets. Blindly applying them causes an infeasible-solver crash.

### Standard Constraint Pipeline

For a complete, runnable template of the standard PyPortfolioOpt constraint
pipeline (expected returns, covariance, bounds, geo constraints, MER drag,
solver configuration), load:

- `scripts/efficient-frontier-template.py`

## Bayesian Estimation

Uses PyMC to estimate posterior distributions of asset returns, producing
expected returns and covariance matrices that can replace historical estimates.

For a runnable default PyMC model template (hierarchical multivariate normal
with LKJ prior on correlations), load:

- `scripts/bayesian-template.py`

## Models

Frozen dataclasses in `optimization/models.py`:

- `PortfolioWeights` — Named weights with optional category.
- `OptimisationPerformance` — Expected return, volatility, Sharpe ratio.
- `OptimisationResult` — Combines weights + performance.

## Critical Guardrails

- **MER is decimal**: `grep -rn '/ 100' src/pysharpe/ --include='*.py' | grep -i mer`
- **Infeasible geo**: `grep -rn 'lower_bound' src/pysharpe/ --include='*.py'`
- **No MPT replacement**: The optimizer is for efficient frontier analysis.
  Allocation decisions use the VA strategy (see `pysharpe-allocation`).

## Testing

Relevant test files:
- `tests/test_optimization_base.py`
- `tests/test_optimization_weights.py`
- `tests/test_portfolio_optimization.py`
- `tests/test_2d_allocation.py`
- `tests/test_tax_location.py`

Run: `uv run pytest tests/test_optimization_base.py tests/test_portfolio_optimization.py tests/test_tax_location.py`
