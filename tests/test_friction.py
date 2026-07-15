"""Integration tests for the execution-friction stress-testing engine.

All tests use synthetic data with fixed seeds — no network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.backtest_engine import BacktestResult
from pysharpe.validation.friction import (
    FrictionStep,
    _compute_annual_turnover,
    stress_test_execution_friction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backtest_result(
    returns: np.ndarray,
    weights: np.ndarray | None = None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    *,
    seed: int = 42,
    initial_value: float = 10_000.0,
    freq: str = "B",
) -> BacktestResult:
    """Build a synthetic ``BacktestResult`` from a daily return vector.

    Args:
        returns: 1-D array of daily return fractions.
        weights: Optional (n_days+1, n_assets) weight matrix.  When ``None`` a
            single-asset (100 %) weight history is generated.
        rebalance_dates: Optional DatetimeIndex of rebalance events.
        seed: RNG seed for reproducibility.
        initial_value: Starting portfolio value.
        freq: Pandas frequency string for the date index.

    Returns:
        A ``BacktestResult`` with the supplied return/weight characteristics.
    """
    _ = seed  # reserved for future reproducibility extensions
    n_days = len(returns) + 1  # portfolio value has one extra observation
    dates = pd.date_range("2024-01-01", periods=n_days, freq=freq)

    pv = initial_value * np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    pv_series = pd.Series(pv, index=dates, name="Portfolio Value")

    if weights is None:
        # Single-asset, always 100 %.
        w = np.ones((n_days, 1))
        columns = ["ASSET_A"]
    else:
        w = weights
        n_assets = w.shape[1]
        columns = [f"ASSET_{chr(65 + i)}" for i in range(n_assets)]

    w_df = pd.DataFrame(w, index=dates, columns=columns)

    if rebalance_dates is None:
        rebalance_dates = pd.DatetimeIndex([])

    return BacktestResult(
        portfolio_value=pv_series,
        historical_weights=w_df,
        rebalance_events=rebalance_dates,
    )


# ---------------------------------------------------------------------------
# Turnover estimation tests
# ---------------------------------------------------------------------------


def test_turnover_no_rebalance_events():
    """Turnover is zero when there are no rebalance events."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    weights = pd.DataFrame(
        {"A": [0.5, 0.6, 0.7, 0.8, 0.9]},
        index=dates,
    )
    result = _compute_annual_turnover(weights, pd.DatetimeIndex([]))
    assert result == 0.0


def test_turnover_empty_weights():
    """Turnover is zero with empty weight history."""
    weights = pd.DataFrame(columns=["A"])
    events = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])
    result = _compute_annual_turnover(weights, events)
    assert result == 0.0


def test_turnover_single_rebalance():
    """Turnover correctly computed from a single rebalance event."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    # Pre-rebalance (idx=1): [0.6, 0.4]. Post-rebalance (idx=2): [0.5, 0.5].
    # Turnover = 0.5 * (|0.5-0.6| + |0.5-0.4|) = 0.5 * 0.2 = 0.1
    weights = pd.DataFrame(
        {"A": [0.5, 0.6, 0.5], "B": [0.5, 0.4, 0.5]},
        index=dates,
    )
    events = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])

    turnover = _compute_annual_turnover(weights, events, periods_per_year=252)

    # Per-event turnover = 0.1.
    # One event over 2 days → events_per_year = 1 / (2/365.25) = 182.625
    # annual = 0.1 * 182.625 = 18.2625
    expected = 0.1 * 365.25 / 2
    assert turnover == pytest.approx(expected, rel=1e-9)


def test_turnover_first_day_rebalance_skipped():
    """Rebalance on the first day is skipped (no pre-weight to compare)."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]},
        index=dates,
    )
    events = pd.DatetimeIndex([pd.Timestamp("2024-01-01")])  # first day
    result = _compute_annual_turnover(weights, events)
    assert result == 0.0


def test_turnover_multiple_rebalances():
    """Turnover averaged across multiple rebalance events."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    # idx=2 rebalance: pre [0.6,0.4], post [0.5,0.5] -> turnover = 0.1
    # idx=4 rebalance: pre [0.6,0.4], post [0.5,0.5] -> turnover = 0.1
    weights = pd.DataFrame(
        {
            "A": [0.5, 0.6, 0.5, 0.6, 0.5],
            "B": [0.5, 0.4, 0.5, 0.4, 0.5],
        },
        index=dates,
    )
    events = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")]
    )

    turnover = _compute_annual_turnover(weights, events, periods_per_year=252)

    # Per-event turnovers: both 0.1 -> avg = 0.1
    # 2 events over 4 days -> events_per_year = 2 / (4/365.25) = 182.625
    expected = 0.1 * 365.25 * 2 / 4
    assert turnover == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Friction stress-test: zero turnover
# ---------------------------------------------------------------------------


def test_zero_turnover_no_nav_decay():
    """With zero turnover, friction costs have no effect on NAV or Sharpe."""
    n_days = 252
    rng = np.random.default_rng(seed=123)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)

    result = _make_backtest_result(daily_rets)

    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=10, periods_per_year=252
    )

    # All steps should have identical Sharpe and zero NAV decay.
    sharpe_values = [step.sharpe for step in profile.steps]
    nav_decays = [step.nav_decay_pct for step in profile.steps]

    assert len(set(round(s, 10) for s in sharpe_values)) == 1
    assert all(abs(d) < 1e-10 for d in nav_decays)
    assert profile.average_annual_turnover == 0.0
    # With zero turnover, break-even should be None (always survives).
    assert profile.break_even_bps is None
    assert profile.is_viable is True


# ---------------------------------------------------------------------------
# Friction stress-test: break-even analytical verification
# ---------------------------------------------------------------------------


def test_break_even_analytical_match():
    """Break-even point matches pen-and-paper hand calculation.

    Setup:
    - Daily gross return: 1 bp (0.0001).
    - Annual turnover: 3.0 (300 %, churn-heavy strategy).
    - Amplification α = 0.5.

    Analytical break-even:
        c = r * 252 / (τ · (1 + α · τ))
          = 0.0001 * 252 / (3.0 * (1 + 0.5 * 3.0))
          = 0.0252 / 7.5
          = 0.00336
          = 33.6 bps

    The function sweeps at 5 bps intervals, so the interpolated break-even
    should fall between 30 and 35 bps, close to 33.6.
    """
    n_days = 252
    daily_ret = 0.0001
    # Add tiny noise to avoid zero-volatility edge case while preserving
    # the analytical break-even calculation (noise contribution is negligible).
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=1e-8, size=n_days)
    returns = np.full(n_days, daily_ret) + noise

    # Construct weight history to produce annual turnover τ = 3.0.
    # With approximately monthly rebalancing (every 21 trading days),
    # we need per-event turnover = 3.0 / 12 ≈ 0.25.
    #
    # Strategy: two assets, alternating between 50/50 post-rebalance
    # and drifting.  Set pre-weight = [0.75, 0.25] and post = [0.5, 0.5].
    # Turnover = 0.5 * (|0.5-0.75| + |0.5-0.25|) = 0.5 * 0.5 = 0.25 ✓
    n_points = n_days + 1
    dates = pd.date_range("2024-01-01", periods=n_points, freq="B")

    weights = np.zeros((n_points, 2))
    weights[:, 0] = 0.5
    weights[:, 1] = 0.5

    rebalance_indices: list[int] = []
    rebalance_step = 21  # ~monthly

    for i in range(rebalance_step, n_points, rebalance_step):
        # Set pre-rebalance weight at i-1 to drifted values.
        weights[i - 1, 0] = 0.75
        weights[i - 1, 1] = 0.25
        rebalance_indices.append(i)

    w_df = pd.DataFrame(weights, index=dates, columns=["ASSET_A", "ASSET_B"])
    rebalance_dates = pd.DatetimeIndex([dates[i] for i in rebalance_indices])

    # Portfolio value: deterministic growth.
    pv = 10_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    pv_series = pd.Series(pv, index=dates, name="Portfolio Value")

    bt_result = BacktestResult(
        portfolio_value=pv_series,
        historical_weights=w_df,
        rebalance_events=rebalance_dates,
    )

    profile = stress_test_execution_friction(
        bt_result, max_bps=50, step_bps=5, periods_per_year=252
    )

    # Verify turnover is close to 3.0.
    assert profile.average_annual_turnover == pytest.approx(3.0, rel=0.15)

    # Break-even should be interpolated between 30 and 35 bps.
    assert profile.break_even_bps is not None
    assert 30.0 <= profile.break_even_bps <= 38.0, (
        f"Expected break-even near 33.6 bps, got {profile.break_even_bps}"
    )

    # At 30 bps: excess should be positive.
    step_30 = next(s for s in profile.steps if s.bps == 30)
    assert step_30.excess_return > 0.0, (
        f"Expected positive excess at 30 bps, got {step_30.excess_return}"
    )

    # At 35 bps: excess should be negative.
    step_35 = next(s for s in profile.steps if s.bps == 35)
    assert step_35.excess_return < 0.0, (
        f"Expected negative excess at 35 bps, got {step_35.excess_return}"
    )

    # NAV should decay monotonically (increasingly negative).
    nav_decays = [s.nav_decay_pct for s in profile.steps]
    for i in range(1, len(nav_decays)):
        assert nav_decays[i] <= nav_decays[i - 1] + 1e-10, (
            f"NAV decay not monotonic at step {i}: "
            f"{nav_decays[i-1]} -> {nav_decays[i]}"
        )


# ---------------------------------------------------------------------------
# Friction stress-test: institutional viability
# ---------------------------------------------------------------------------


def test_viable_strategy_passes_threshold():
    """A low-turnover strategy with strong returns passes the 40 bps threshold."""
    n_days = 504  # 2 years
    rng = np.random.default_rng(seed=99)
    # Strong returns: ~15 % annual.
    daily_rets = rng.normal(loc=0.0006, scale=0.008, size=n_days)

    result = _make_backtest_result(daily_rets)

    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=5, periods_per_year=252
    )

    # Zero turnover → always viable.
    assert profile.is_viable is True
    assert profile.break_even_bps is None


def test_unviable_strategy_fails_threshold():
    """A high-turnover, low-return strategy fails the 40 bps threshold."""
    n_days = 252
    # Very low returns: ~2.5 % annual, with tiny noise for non-zero vol.
    daily_ret = 0.0001
    rng = np.random.default_rng(seed=99)
    noise = rng.normal(loc=0.0, scale=1e-8, size=n_days)
    returns = np.full(n_days, daily_ret) + noise

    n_points = n_days + 1
    dates = pd.date_range("2024-01-01", periods=n_points, freq="B")

    # Construct high turnover: ~350 % annual.
    # Per-event turnover: 0.30, with ~12 events/year → 3.6 annual.
    weights = np.zeros((n_points, 2))
    weights[:, 0] = 0.5
    weights[:, 1] = 0.5

    rebalance_indices: list[int] = []
    for i in range(21, n_points, 21):
        # drifted: [0.80, 0.20] -> turnover = 0.5*(0.30+0.30) = 0.30
        weights[i - 1, 0] = 0.80
        weights[i - 1, 1] = 0.20
        rebalance_indices.append(i)

    w_df = pd.DataFrame(weights, index=dates, columns=["A", "B"])
    rebalance_dates = pd.DatetimeIndex([dates[i] for i in rebalance_indices])

    pv = 10_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    pv_series = pd.Series(pv, index=dates, name="Portfolio Value")

    bt_result = BacktestResult(
        portfolio_value=pv_series,
        historical_weights=w_df,
        rebalance_events=rebalance_dates,
    )

    profile = stress_test_execution_friction(
        bt_result, max_bps=50, step_bps=5, periods_per_year=252
    )

    # With high turnover and low returns, should fail the 40 bps threshold.
    assert profile.break_even_bps is not None
    assert profile.break_even_bps < 40.0
    assert profile.is_viable is False


# ---------------------------------------------------------------------------
# Friction stress-test: exponential degradation for high turnover
# ---------------------------------------------------------------------------


def test_high_turnover_degrades_faster():
    """High-turnover strategies show exponentially larger NAV decay than low."""
    n_days = 252
    rng = np.random.default_rng(seed=77)
    daily_rets = rng.normal(loc=0.0004, scale=0.01, size=n_days)

    # --- Low-turnover setup (~20 % annual) ---
    n_points = n_days + 1
    dates = pd.date_range("2024-01-01", periods=n_points, freq="B")
    weights_low = np.zeros((n_points, 2))
    weights_low[:, 0] = 0.5
    weights_low[:, 1] = 0.5

    rebalance_low: list[int] = []
    for i in range(63, n_points, 63):  # quarterly
        weights_low[i - 1, 0] = 0.55
        weights_low[i - 1, 1] = 0.45
        rebalance_low.append(i)

    w_low = pd.DataFrame(weights_low, index=dates, columns=["A", "B"])
    reb_low = pd.DatetimeIndex([dates[i] for i in rebalance_low])

    pv = 10_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + daily_rets]))
    pv_series = pd.Series(pv, index=dates, name="Portfolio Value")

    bt_low = BacktestResult(
        portfolio_value=pv_series,
        historical_weights=w_low,
        rebalance_events=reb_low,
    )

    profile_low = stress_test_execution_friction(
        bt_low, max_bps=50, step_bps=10, periods_per_year=252
    )

    # --- High-turnover setup (~200 % annual) ---
    weights_high = np.zeros((n_points, 2))
    weights_high[:, 0] = 0.5
    weights_high[:, 1] = 0.5

    rebalance_high: list[int] = []
    for i in range(21, n_points, 21):  # monthly, larger drift
        weights_high[i - 1, 0] = 0.75
        weights_high[i - 1, 1] = 0.25
        rebalance_high.append(i)

    w_high = pd.DataFrame(weights_high, index=dates, columns=["A", "B"])
    reb_high = pd.DatetimeIndex([dates[i] for i in rebalance_high])

    bt_high = BacktestResult(
        portfolio_value=pv_series,
        historical_weights=w_high,
        rebalance_events=reb_high,
    )

    profile_high = stress_test_execution_friction(
        bt_high, max_bps=50, step_bps=10, periods_per_year=252
    )

    # High-turnover should have higher annual turnover.
    assert profile_high.average_annual_turnover > profile_low.average_annual_turnover

    # At max cost (50 bps), high-turnover NAV decay should be substantially
    # larger than low-turnover decay.
    decay_low_50 = next(s.nav_decay_pct for s in profile_low.steps if s.bps == 50)
    decay_high_50 = next(s.nav_decay_pct for s in profile_high.steps if s.bps == 50)

    # The high-turnover decay ratio should be at least 2x that of low-turnover.
    ratio = abs(decay_high_50) / max(abs(decay_low_50), 1e-10)
    assert ratio > 2.0, (
        f"High-turnover NAV decay ({decay_high_50:.4f}%) should be "
        f"substantially larger than low-turnover ({decay_low_50:.4f}%). "
        f"Ratio: {ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Friction stress-test: edge cases
# ---------------------------------------------------------------------------


def test_empty_portfolio_value_raises():
    """Empty portfolio value series raises ValueError."""
    empty = BacktestResult(
        portfolio_value=pd.Series(dtype=float),
        historical_weights=pd.DataFrame(),
        rebalance_events=pd.DatetimeIndex([]),
    )
    with pytest.raises(ValueError, match="empty"):
        stress_test_execution_friction(empty)


def test_single_observation_raises():
    """Single price observation (zero returns) raises ValueError."""
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    result = BacktestResult(
        portfolio_value=pd.Series([100.0], index=dates),
        historical_weights=pd.DataFrame({"A": [0.5]}, index=dates),
        rebalance_events=pd.DatetimeIndex([]),
    )
    with pytest.raises(ValueError, match="Insufficient"):
        stress_test_execution_friction(result)


def test_invalid_step_params_raise():
    """Invalid step parameters raise ValueError."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    pv = pd.Series(100.0 * 1.001 ** np.arange(10), index=dates)
    result = BacktestResult(
        portfolio_value=pv,
        historical_weights=pd.DataFrame({"A": [0.5] * 10}, index=dates),
        rebalance_events=pd.DatetimeIndex([]),
    )

    with pytest.raises(ValueError, match="step_bps must be positive"):
        stress_test_execution_friction(result, step_bps=0)

    with pytest.raises(ValueError, match="max_bps must be >= step_bps"):
        stress_test_execution_friction(result, max_bps=5, step_bps=10)


def test_institutional_threshold_always_tested():
    """The 40 bps institutional threshold is always included in the sweep."""
    n_days = 252
    rng = np.random.default_rng(seed=1)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    result = _make_backtest_result(daily_rets)

    # Use a step that would normally skip 40 bps.
    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=15, periods_per_year=252
    )

    bps_values = [s.bps for s in profile.steps]
    assert 40 in bps_values, f"40 bps must be tested; got steps at {bps_values}"


def test_nav_decay_baseline_is_zero():
    """At 0 bps cost, NAV decay must be exactly 0 %."""
    n_days = 252
    rng = np.random.default_rng(seed=42)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    result = _make_backtest_result(daily_rets)

    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=10, periods_per_year=252
    )

    step_0 = profile.steps[0]
    assert step_0.bps == 0
    assert step_0.nav_decay_pct == pytest.approx(0.0, abs=1e-10)


def test_friction_step_fields_are_consistent():
    """Each FrictionStep's cost_decimal matches bps / 10000."""
    n_days = 252
    rng = np.random.default_rng(seed=5)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    result = _make_backtest_result(daily_rets)

    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=10, periods_per_year=252
    )

    for step in profile.steps:
        assert step.cost_decimal == pytest.approx(step.bps / 10_000.0)
        assert step.excess_return == pytest.approx(step.annualized_return)


def test_sharpe_decreases_with_cost():
    """Sharpe ratio should monotonically decrease as cost increases."""
    n_days = 504
    rng = np.random.default_rng(seed=13)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)

    # Add some turnover to see the effect.
    n_points = n_days + 1
    dates = pd.date_range("2024-01-01", periods=n_points, freq="B")
    weights = np.zeros((n_points, 2))
    weights[:, 0] = 0.5
    weights[:, 1] = 0.5

    rebalance_indices: list[int] = []
    for i in range(21, n_points, 21):
        weights[i - 1, 0] = 0.6
        weights[i - 1, 1] = 0.4
        rebalance_indices.append(i)

    w_df = pd.DataFrame(weights, index=dates, columns=["A", "B"])
    reb_dates = pd.DatetimeIndex([dates[i] for i in rebalance_indices])

    pv = 10_000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + daily_rets]))
    pv_series = pd.Series(pv, index=dates, name="Portfolio Value")

    bt = BacktestResult(
        portfolio_value=pv_series,
        historical_weights=w_df,
        rebalance_events=reb_dates,
    )

    profile = stress_test_execution_friction(
        bt, max_bps=50, step_bps=10, periods_per_year=252
    )

    sharpes = [s.sharpe for s in profile.steps]
    # With positive turnover and returns, Sharpe should decline.
    for i in range(1, len(sharpes)):
        assert sharpes[i] <= sharpes[i - 1] + 1e-10, (
            f"Sharpe increased from {sharpes[i-1]} to {sharpes[i]} at step {i}"
        )


def test_profile_dataclass_fields_match():
    """FrictionProfile aggregate fields are internally consistent."""
    n_days = 252
    rng = np.random.default_rng(seed=7)
    daily_rets = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    result = _make_backtest_result(daily_rets)

    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=10, periods_per_year=252
    )

    assert isinstance(profile.steps, list)
    assert all(isinstance(s, FrictionStep) for s in profile.steps)
    assert len(profile.steps) > 0
    assert profile.average_annual_turnover >= 0.0
    # max_survivable should be the highest bps with positive excess.
    surviving = [s.bps for s in profile.steps if s.excess_return > 0.0]
    if surviving:
        assert profile.max_survivable_bps == pytest.approx(max(surviving))


# ---------------------------------------------------------------------------
# Regression: division-by-zero in zero-turnover cost attribution
# ---------------------------------------------------------------------------


def test_zero_turnover_no_division_error():
    """Zero-turnover periods must not raise division-by-zero errors."""
    n_days = 252
    rng = np.random.default_rng(seed=42)
    daily_rets = rng.normal(loc=0.0003, scale=0.01, size=n_days)
    result = _make_backtest_result(daily_rets)

    # Should complete without error.
    profile = stress_test_execution_friction(
        result, max_bps=50, step_bps=5, periods_per_year=252
    )
    assert profile.average_annual_turnover == 0.0
    assert len(profile.steps) > 0
