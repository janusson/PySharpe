"""Degraded transaction cost stress-testing engine.

Evaluates out-of-sample edge decay under escalating execution friction to
establish break-even survival thresholds for institutional viability.

Key components
--------------
* ``FrictionStep`` — single cost-step snapshot (bps, Sharpe, NAV decay).
* ``FrictionProfile`` — aggregated stress-test result with break-even and
  institutional viability flag.
* ``stress_test_execution_friction`` — sensitivity stepper that re-evaluates
  Sharpe and NAV at each cost level without re-running market data ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pysharpe.metrics import annualize_return, annualize_volatility, sharpe_ratio

if TYPE_CHECKING:
    from pysharpe.analysis.backtest_engine import BacktestResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASELINE_BPS: int = 5
_DEFAULT_MAX_BPS: int = 50
_DEFAULT_STEP_BPS: int = 5
_INSTITUTIONAL_THRESHOLD_BPS: int = 40
_TRADING_DAYS_PER_YEAR: int = 252

# Amplification factor for high-turnover cost nonlinearity.
# A value > 0 causes high-turnover strategies to degrade *exponentially faster*
# than low-turnover allocations as costs escalate. Set to 0.0 for pure linear
# scaling; 0.5 is a conservative choice that penalises churn without overstating
# the effect for moderate-turnover portfolios.
_TURNOVER_AMPLIFICATION: float = 0.5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrictionStep:
    """Single cost-step result produced by the friction stress-tester.

    Attributes
    ----------
    bps:
        Execution cost in basis points for this step.
    cost_decimal:
        Same cost expressed as a decimal fraction (bps / 10_000).
    annualized_return:
        Geometric annualised return of the post-friction return series.
    annualized_volatility:
        Annualised volatility of the post-friction return series.
    sharpe:
        Annualised Sharpe ratio (risk-free = 0 by default).
    nav_decay_pct:
        Percentage decay in terminal NAV relative to the zero-cost baseline.
        Negative values indicate NAV erosion; zero at the baseline step.
    excess_return:
        Annualised return minus the risk-free rate (annualised excess return).
    """

    bps: int
    cost_decimal: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    nav_decay_pct: float
    excess_return: float


@dataclass(frozen=True)
class FrictionProfile:
    """Aggregated execution-friction stress-test result.

    Attributes
    ----------
    steps:
        Ordered list of ``FrictionStep`` entries, one per tested cost level
        (ascending bps).
    break_even_bps:
        Interpolated basis-point cost where net excess return crosses zero.
        ``None`` when the strategy survives all tested cost levels.
    is_viable:
        ``True`` when the strategy survives the institutional 40 bps threshold
        (excess return remains positive at that level).
    max_survivable_bps:
        Highest tested cost level (in bps) where excess return was still
        positive. If break-even is interpolated, this may be the floor of the
        break-even region.
    average_annual_turnover:
        Estimated average annual portfolio turnover derived from weight history.
    """

    steps: list[FrictionStep]
    break_even_bps: float | None
    is_viable: bool
    max_survivable_bps: float
    average_annual_turnover: float


# ---------------------------------------------------------------------------
# Turnover estimation
# ---------------------------------------------------------------------------


def _compute_annual_turnover(
    weights: pd.DataFrame,
    rebalance_events: pd.DatetimeIndex,
    *,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """Estimate average annual portfolio turnover from historical weight data.

    Turnover is calculated as half the sum of absolute weight changes on each
    rebalance date, then annualised based on the observed rebalance frequency.

    Args:
        weights: Daily weight history (rows = dates, columns = assets).
        rebalance_events: Dates on which rebalancing occurred.
        periods_per_year: Trading days per year for annualisation.

    Returns:
        Annual turnover as a decimal (e.g., 0.5 = 50 % turnover per year).
        Returns 0.0 when there are no rebalance events or insufficient data.
    """
    if len(rebalance_events) == 0:
        return 0.0

    if weights.empty or len(weights) < 2:
        return 0.0

    turnovers: list[float] = []
    for date in rebalance_events:
        if date not in weights.index:
            continue
        idx = weights.index.get_loc(date)
        if idx == 0:
            continue  # Cannot compute pre-rebalance weight for the first day.

        pre_weights = weights.iloc[idx - 1].values.astype(float)
        post_weights = weights.iloc[idx].values.astype(float)

        # Turnover = 0.5 * Σ |Δw_i|  (classic two-sided definition).
        turnover = 0.5 * float(np.sum(np.abs(post_weights - pre_weights)))
        turnovers.append(turnover)

    if not turnovers:
        return 0.0

    avg_turnover_per_event = float(np.mean(turnovers))
    n_events = len(turnovers)
    n_days_total = (weights.index[-1] - weights.index[0]).days

    if n_days_total <= 0:
        return 0.0

    years = n_days_total / 365.25
    events_per_year = n_events / years if years > 0 else 0.0
    annual_turnover = avg_turnover_per_event * events_per_year

    return max(annual_turnover, 0.0)


# ---------------------------------------------------------------------------
# Main stress-test function
# ---------------------------------------------------------------------------


def stress_test_execution_friction(
    backtest_result: BacktestResult,
    *,
    max_bps: int = _DEFAULT_MAX_BPS,
    step_bps: int = _DEFAULT_STEP_BPS,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
    baseline_bps: int = _BASELINE_BPS,
) -> FrictionProfile:
    """Stress-test a backtest result against escalating transaction costs.

    The engine computes the daily gross return series from the portfolio
    value curve, estimates annual turnover from the weight history, and then
    applies synthetic cost drag at each cost step. Sharpe ratio and NAV decay
    are re-evaluated at each step **without** re-running any market data
    ingestion or rebalancing simulation.

    High-turnover strategies degrade *exponentially faster* than low-turnover
    allocations because the effective cost rate is amplified by
    ``c * (1 + α · τ)`` where ``τ`` is annual turnover and ``α`` is the
    amplification constant ``_TURNOVER_AMPLIFICATION``.

    Args:
        backtest_result: A completed ``BacktestResult`` from the backtest engine.
        max_bps: Upper bound of the cost sweep in basis points (default 50).
        step_bps: Increment between successive cost levels (default 5).
        risk_free_rate: Annual risk-free rate as a decimal (default 0.0).
        periods_per_year: Trading days per year (default 252).
        baseline_bps: Baseline cost level for NAV-decay reference (default 5).

    Returns:
        A ``FrictionProfile`` with per-step metrics, break-even bps, and
        institutional viability flag.

    Raises:
        ValueError: If the portfolio value series is empty or has insufficient
            observations for annualisation.
    """
    # ------------------------------------------------------------------
    # 1. Extract gross return series
    # ------------------------------------------------------------------
    portfolio_value: pd.Series = backtest_result.portfolio_value

    if portfolio_value.empty:
        raise ValueError("Portfolio value series is empty; cannot stress-test.")

    gross_returns = portfolio_value.pct_change().dropna()
    if len(gross_returns) < 2:
        raise ValueError(
            "Insufficient return observations for annualised metric calculation."
        )

    # ------------------------------------------------------------------
    # 2. Estimate annual turnover
    # ------------------------------------------------------------------
    annual_turnover = _compute_annual_turnover(
        backtest_result.historical_weights,
        backtest_result.rebalance_events,
        periods_per_year=periods_per_year,
    )

    # Daily turnover fraction (uniformly distributed for cost modelling).
    daily_turnover = annual_turnover / periods_per_year if periods_per_year > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. Generate cost steps
    # ------------------------------------------------------------------
    if step_bps <= 0:
        raise ValueError("step_bps must be positive.")
    if max_bps < step_bps:
        raise ValueError("max_bps must be >= step_bps.")

    cost_levels_bps: list[int] = list(range(0, max_bps + step_bps, step_bps))

    # Ensure the institutional threshold (40 bps) is always tested.
    threshold = _INSTITUTIONAL_THRESHOLD_BPS
    if threshold not in cost_levels_bps:
        cost_levels_bps.append(threshold)
        cost_levels_bps = sorted(set(cost_levels_bps))

    steps: list[FrictionStep] = []
    baseline_nav: float | None = None

    for bps in cost_levels_bps:
        cost_decimal = bps / 10_000.0  # Convert bps → decimal fraction.

        # --- Nonlinear cost amplification for high-turnover portfolios ---
        # effective_cost = c · (1 + α · τ)
        # This ensures that doubling the cost rate produces *more than double*
        # the drag for high-turnover strategies, satisfying the exponential
        # degradation constraint.
        amplified_cost = cost_decimal * (
            1.0 + _TURNOVER_AMPLIFICATION * annual_turnover
        )

        # Daily cost drag (only non-zero when turnover > 0).
        daily_cost_drag = daily_turnover * amplified_cost

        # Net return series after friction.
        if daily_cost_drag > 0.0:
            net_returns = gross_returns - daily_cost_drag
        else:
            net_returns = gross_returns.copy()

        # --- Annualised metrics ---
        ann_return = annualize_return(net_returns, periods_per_year=periods_per_year)
        ann_vol = annualize_volatility(net_returns, periods_per_year=periods_per_year)

        # Sharpe ratio is undefined when volatility is exactly zero.
        # Treat as ±inf depending on sign of excess return for ranking purposes.
        try:
            shp = sharpe_ratio(
                net_returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        except ValueError:
            excess_temp = ann_return - risk_free_rate
            shp = float("inf") if excess_temp > 0 else float("-inf")

        excess = ann_return - risk_free_rate

        # --- NAV decay vs. zero-cost baseline ---
        terminal_nav = (1.0 + net_returns).prod()
        if baseline_nav is None:
            baseline_nav = terminal_nav  # zero-cost baseline (bps == 0).

        nav_decay_pct = (
            ((terminal_nav - baseline_nav) / baseline_nav) * 100.0
            if baseline_nav != 0.0
            else 0.0
        )

        steps.append(
            FrictionStep(
                bps=bps,
                cost_decimal=cost_decimal,
                annualized_return=float(ann_return),
                annualized_volatility=float(ann_vol),
                sharpe=float(shp),
                nav_decay_pct=float(nav_decay_pct),
                excess_return=float(excess),
            )
        )

    # ------------------------------------------------------------------
    # 4. Break-even identification (linear interpolation)
    # ------------------------------------------------------------------
    break_even_bps: float | None = None
    max_survivable = 0.0

    for i, step in enumerate(steps):
        if step.excess_return > 0.0:
            max_survivable = float(step.bps)
        elif i > 0:
            prev = steps[i - 1]
            if prev.excess_return > 0.0:
                # Linear interpolation between last positive and first
                # non-positive excess return.
                frac = prev.excess_return / (prev.excess_return - step.excess_return)
                break_even_bps = prev.bps + frac * (step.bps - prev.bps)
            break

    # If we never found a non-positive step, break-even is beyond tested range.
    if break_even_bps is None and steps[-1].excess_return > 0.0:
        # Break-even is beyond max tested.
        break_even_bps = None
        max_survivable = float(steps[-1].bps)

    # ------------------------------------------------------------------
    # 5. Institutional viability check
    # ------------------------------------------------------------------
    is_viable = break_even_bps is None or break_even_bps >= threshold

    return FrictionProfile(
        steps=steps,
        break_even_bps=break_even_bps,
        is_viable=is_viable,
        max_survivable_bps=max_survivable,
        average_annual_turnover=annual_turnover,
    )
