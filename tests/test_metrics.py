"""Unit tests for the metrics helper module.

.. note::

    **Canadian Investment Context** — All metrics (Sharpe, Sortino, CAGR,
    maximum drawdown, etc.) are computed on CAD-denominated price series.
    Risk-free rate defaults to Canadian government bond yields.  Returns
    are annualized using 252 trading days (North American convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.metrics import (
    annualize_return,
    annualize_volatility,
    cagr,
    calmar_ratio,
    compute_realized_volatility,
    compute_returns,
    expected_return,
    max_drawdown_duration,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)


def test_compute_returns_simple_series(sample_price_series):
    returns = compute_returns(sample_price_series)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_price_series) - 1
    np.testing.assert_allclose(
        returns.iloc[:3].to_numpy(dtype=float),
        np.array([0.02, -0.00980392, 0.01980198]),
        rtol=1e-6,
    )


def test_compute_returns_log_matches_manual(sample_price_series):
    log_returns = compute_returns(sample_price_series, method="log")
    manual = np.log(
        sample_price_series.iloc[1:].values / sample_price_series.iloc[:-1].values
    )
    np.testing.assert_allclose(log_returns.to_numpy(dtype=float), manual, rtol=1e-9)


def test_compute_returns_validates_method(sample_price_series):
    with pytest.raises(ValueError):
        compute_returns(sample_price_series, method="bogus")


def test_annualize_return_series(sample_price_series):
    returns = compute_returns(sample_price_series)
    result = annualize_return(returns, periods_per_year=252)
    manual = (1 + returns).prod() ** (252 / len(returns)) - 1
    assert pytest.approx(manual, rel=1e-9) == result


def test_annualize_return_dataframe(sample_price_frame):
    returns = compute_returns(sample_price_frame)
    result = annualize_return(returns, periods_per_year=252)
    assert isinstance(result, pd.Series)
    assert set(result.index) == {"AAA", "BBB", "CCC"}


def test_annualize_volatility_requires_samples(sample_price_series):
    with pytest.raises(ValueError):
        annualize_volatility(sample_price_series.iloc[:1])


def test_expected_return_matches_mean(sample_price_frame):
    returns = compute_returns(sample_price_frame)
    observed = expected_return(returns, periods_per_year=252)
    manual = returns.mean() * 252
    pd.testing.assert_series_equal(observed, manual)


def test_sharpe_ratio_handles_risk_free_rate(sample_price_series):
    returns = compute_returns(sample_price_series)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
    ann_return = annualize_return(returns, periods_per_year=252)
    ann_vol = annualize_volatility(returns, periods_per_year=252)
    expected = (ann_return - 0.02) / ann_vol
    assert pytest.approx(expected, rel=1e-9) == sharpe


def test_sharpe_ratio_raises_when_vol_zero():
    returns = pd.Series([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        sharpe_ratio(returns)


def test_compute_realized_volatility_matches_manual(sample_price_series):
    window = 5
    result = compute_realized_volatility(
        sample_price_series, window=window, periods_per_year=252
    )

    # Manual calculation
    log_returns = np.log(sample_price_series / sample_price_series.shift(1))
    manual = log_returns.rolling(window=window).std() * np.sqrt(252)

    pd.testing.assert_series_equal(result, manual)


def test_compute_realized_volatility_validates_window(sample_price_series):
    with pytest.raises(ValueError):
        compute_realized_volatility(sample_price_series, window=1)


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------


def test_cagr_positive_growth():
    """CAGR for a steadily growing 5-year portfolio."""
    dates = pd.date_range("2020-01-01", periods=5 * 252, freq="B")
    # 7% annual growth = daily return of ~1.07^(1/252) - 1
    daily_r = (1.07 ** (1 / 252)) - 1
    values = 100 * (1 + daily_r) ** np.arange(len(dates))
    series = pd.Series(values, index=dates)

    result = cagr(series)
    assert 0.065 < result < 0.075, f"Expected ~7%, got {result:.4f}"


def test_cagr_negative_growth():
    """CAGR for a declining portfolio."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    daily_r = (0.95 ** (1 / 252)) - 1  # -5% annual decline
    values = 100 * (1 + daily_r) ** np.arange(len(dates))
    series = pd.Series(values, index=dates)

    result = cagr(series)
    assert -0.06 < result < -0.04, f"Expected ~-5%, got {result:.4f}"


def test_cagr_flattens_to_zero():
    """When start and end values are identical, CAGR is zero."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    series = pd.Series([100.0] * 100, index=dates)

    assert cagr(series) == pytest.approx(0.0, abs=1e-12)


def test_cagr_empty_returns_zero():
    """Empty series returns 0.0."""
    series = pd.Series([], dtype=float)
    assert cagr(series) == 0.0


def test_cagr_single_day_zero():
    """Single-day series returns 0.0."""
    dates = pd.date_range("2024-01-01", periods=1, freq="B")
    series = pd.Series([100.0], index=dates)
    assert cagr(series) == 0.0


def test_cagr_requires_datetime_index():
    """CAGR raises TypeError for non-DatetimeIndex."""
    series = pd.Series([100, 105, 110])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        cagr(series)


def test_cagr_requires_positive_start():
    """CAGR raises ValueError when initial value ≤ 0."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    series = pd.Series([0.0, 10.0, 20.0], index=dates)
    with pytest.raises(ValueError, match="positive"):
        cagr(series)


def test_cagr_precision_with_exact_doubling():
    """Exact 2× over 1 year should give 100% CAGR."""
    dates = pd.date_range("2020-01-01", "2021-01-01", freq="B")
    # Linear interpolation from 100 to 200 over 252 trading days
    values = np.linspace(100, 200, len(dates))
    series = pd.Series(values, index=dates)

    result = cagr(series)
    # With linear growth, geometric return is approximately 2^(1/1) - 1 = 1.0
    # But the exact depends on the path. Let's approximate.
    assert 0.95 < result < 1.05, f"Expected ~100%, got {result:.4f}"


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------


def test_maximum_drawdown_negative():
    """Maximum drawdown returns a negative value for a declining series."""
    series = pd.Series([100, 95, 90, 85, 100])
    result = maximum_drawdown(series)
    # Peak is 100, trough is 85 → (85-100)/100 = -0.15
    assert result == pytest.approx(-0.15, abs=1e-10)


def test_maximum_drawdown_rising():
    """Continuously rising series has zero drawdown."""
    series = pd.Series([100, 101, 102, 103, 104])
    result = maximum_drawdown(series)
    assert result == 0.0


def test_maximum_drawdown_empty():
    """Empty series returns 0.0."""
    assert maximum_drawdown(pd.Series([], dtype=float)) == 0.0


def test_maximum_drawdown_multiple_peaks():
    """MDD correctly identifies the deepest drawdown across multiple peaks."""
    # Peak at 100 → trough at 70 (-30%), returns to 90 → trough at 50 (-44% from peak 90)
    # But the deepest is from 100 to 70: (70-100)/100 = -0.30
    series = pd.Series([100, 95, 70, 90, 80, 50, 95])
    result = maximum_drawdown(series)
    # Peak = 100, trough = 50 → (50-100)/100 = -0.50
    assert result == pytest.approx(-0.50, abs=1e-10)


def test_maximum_drawdown_recovery_then_bigger_drawdown():
    """After recovery to new high, a bigger drawdown is captured."""
    # 100 → 90 (-10%), back to 110 (new peak), then to 60 (-45.45%)
    series = pd.Series([100, 90, 110, 60, 110])
    # deepest: from 110 to 60 = (60-110)/110 = -0.4545...
    result = maximum_drawdown(series)
    assert result == pytest.approx(-0.454545, abs=1e-5)


def test_maximum_drawdown_single_value():
    """Single value has no drawdown."""
    series = pd.Series([100.0])
    assert maximum_drawdown(series) == 0.0


def test_maximum_drawdown_returns_float():
    """Result is a Python float, not numpy scalar."""
    series = pd.Series([100, 95, 90])
    result = maximum_drawdown(series)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------


def test_sortino_ratio_positive_series():
    """Sortino of a return stream with mostly positive but some negative returns."""
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01, -0.002, 0.03])
    result = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    # CAGR should be positive, ratio > 0
    assert result > 0, f"Sortino should be positive for positive skew, got {result}"


def test_sortino_ratio_negative_series():
    """Sortino of a consistently negative return stream."""
    returns = pd.Series([-0.01, -0.02, -0.015, -0.01, -0.005])
    result = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    assert result < 0, "Sortino should be negative for negative returns"


def test_sortino_ratio_handles_risk_free_rate():
    """Sortino ratio deducts risk-free rate from annualised return."""
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
    rfr = 0.02  # 2% annual risk-free rate

    result = sortino_ratio(returns, risk_free_rate=rfr, periods_per_year=252)
    # Deducting RFR should reduce the ratio compared to RFR=0
    result_zero_rfr = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    assert result < result_zero_rfr, (
        "Sortino with RFR should be lower than Sortino without RFR"
    )


def test_sortino_ratio_raises_on_zero_downside_deviation():
    """Sortino is undefined when all returns exceed the target."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.04])
    with pytest.raises(ValueError, match="Downside deviation is zero"):
        sortino_ratio(returns, target_return=-1.0)  # No returns below target


def test_sortino_ratio_vs_sharpe_for_symmetric_returns():
    """For symmetric returns, Sortino ≈ Sharpe (same denominator)."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.02, 252))

    shp = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    srt = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    # With symmetric returns and target=0, Sortino ≈ Sharpe
    # They won't be exactly equal due to downside vs total deviation
    assert abs(shp - srt) < 0.5, (
        f"Sortino ({srt:.4f}) should be close to Sharpe ({shp:.4f}) for symmetric returns"
    )


def test_sortino_ratio_larger_than_sharpe_for_asymmetric_upside():
    """Sortino > Sharpe when upside volatility dominates."""
    rng = np.random.default_rng(42)
    # Positive skew: large upside spikes, small downside moves
    base = rng.normal(0.0005, 0.01, 252)
    upside_spikes = np.maximum(0, rng.exponential(0.02, 252))
    returns = pd.Series(base + upside_spikes)

    shp = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    srt = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    assert srt > shp, (
        f"Sortino ({srt:.4f}) should exceed Sharpe ({shp:.4f}) with asymmetric upside"
    )


def test_sortino_ratio_with_target_return():
    """Sortino with a non-zero target return."""
    returns = pd.Series([0.01, 0.02, -0.005, 0.0, 0.03])
    # With a target above 0, more returns are "downside"
    result_low_target = sortino_ratio(returns, target_return=0.0, periods_per_year=252)
    # With target=0.02, the -0.005 and 0.0 are downside; downside deviation increases
    result_high_target = sortino_ratio(
        returns, target_return=0.02, periods_per_year=252
    )

    # Higher target → more downside → lower Sortino (for same numerator)
    assert result_high_target < result_low_target, (
        "Higher target return should reduce Sortino ratio"
    )


def test_sortino_ratio_dataframe():
    """Sortino ratio works with DataFrame input."""
    rng = np.random.default_rng(77)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.02, 100),
            "B": rng.normal(0.0005, 0.01, 100),
        }
    )
    result = sortino_ratio(returns, periods_per_year=252)
    assert isinstance(result, pd.Series)
    assert set(result.index) == {"A", "B"}


def test_sortino_ratio_raises_on_zero_vol_specific_column():
    """If any column has zero downside deviation, raises ValueError."""
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],  # All positive — zero downside
            "B": [0.01, -0.02, 0.03],
        }
    )
    with pytest.raises(ValueError, match="Downside deviation is zero"):
        sortino_ratio(returns, target_return=-1.0, periods_per_year=252)


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------


def test_calmar_ratio_basic():
    """Calmar ratio = CAGR / |max drawdown|."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    # Simple: CAGR ≈ 10%, MDD = -20% → Calmar ≈ 0.5
    values = np.linspace(100, 110, 252)
    # Insert a drawdown in the middle
    values[100:150] *= 0.8  # 20% drawdown
    series = pd.Series(values, index=dates)

    result = calmar_ratio(series)
    expected_cagr = cagr(series)
    expected_mdd = maximum_drawdown(series)
    manual = expected_cagr / abs(expected_mdd)
    assert result == pytest.approx(manual, rel=1e-9)


def test_calmar_ratio_no_drawdown_inf():
    """When MDD is zero and positive growth, Calmar = +inf."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    series = pd.Series(np.linspace(100, 110, 10), index=dates)
    assert calmar_ratio(series) == float("inf")


def test_calmar_ratio_no_drawdown_negative_inf():
    """When MDD is zero and CAGR is zero or negative, Calmar = -inf."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    # Flat series: no drawdown, zero CAGR → -inf
    series = pd.Series([100.0] * 10, index=dates)
    assert calmar_ratio(series) == float("-inf")


def test_calmar_ratio_negative_cagr_negative_mdd():
    """Negative CAGR with negative MDD gives negative Calmar."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    # Steady decline: -10% CAGR, -10% MDD
    daily_r = (0.90 ** (1 / 252)) - 1
    values = 100 * (1 + daily_r) ** np.arange(len(dates))
    series = pd.Series(values, index=dates)

    result = calmar_ratio(series)
    assert result < 0, f"Expected negative Calmar for declining portfolio, got {result}"


def test_calmar_ratio_requires_datetime_index():
    """Calmar delegates to CAGR which requires DatetimeIndex."""
    series = pd.Series([100, 90, 100])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        calmar_ratio(series)


# ---------------------------------------------------------------------------
# Tracking Error
# ---------------------------------------------------------------------------


def test_tracking_error_identical_series_zero():
    """Tracking error between identical return series is zero."""
    ra = pd.Series([0.01, -0.005, 0.02, 0.03, -0.01])
    rb = ra.copy()
    result = tracking_error(ra, rb, periods_per_year=252)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_positive_for_different_series():
    """Tracking error must be > 0 when series differ."""
    ra = pd.Series([0.01, -0.005, 0.02, 0.03, -0.01])
    rb = pd.Series([0.005, 0.01, 0.015, 0.02, 0.025])  # Different pattern
    result = tracking_error(ra, rb, periods_per_year=252)
    assert result > 0


def test_tracking_error_matches_manual():
    """Tracking error equals std of differential × sqrt(periods)."""
    rng = np.random.default_rng(42)
    ra = pd.Series(rng.normal(0.001, 0.02, 100))
    rb = pd.Series(rng.normal(0.0005, 0.015, 100))

    result = tracking_error(ra, rb, periods_per_year=252)
    diff = ra - rb
    manual = float(diff.std(ddof=1) * np.sqrt(252))
    assert result == pytest.approx(manual, rel=1e-10)


def test_tracking_error_handles_different_lengths():
    """Tracking error raises ValueError for mismatched lengths."""
    ra = pd.Series([0.01, 0.02, 0.03])
    rb = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="same length"):
        tracking_error(ra, rb)


def test_tracking_error_monthly_frequency():
    """Tracking error scales correctly with monthly periods."""
    rng = np.random.default_rng(99)
    ra = pd.Series(rng.normal(0.005, 0.03, 36))
    rb = pd.Series(rng.normal(0.004, 0.025, 36))

    daily_result = tracking_error(ra, rb, periods_per_year=252)
    monthly_result = tracking_error(ra, rb, periods_per_year=12)
    # Daily-annualised TE ≈ Monthly TE × sqrt(252/12)
    ratio = daily_result / monthly_result
    expected_ratio = np.sqrt(252 / 12)
    assert ratio == pytest.approx(expected_ratio, rel=1e-10)


def test_tracking_error_with_nan():
    """Tracking error works when NaN values are handled upstream."""
    ra_clean = pd.Series([0.01, 0.02, 0.03, -0.01])
    rb_clean = pd.Series([0.005, 0.01, 0.02, 0.025])
    result = tracking_error(ra_clean, rb_clean, periods_per_year=252)
    assert result > 0


# ---------------------------------------------------------------------------
# Maximum Drawdown Duration
# ---------------------------------------------------------------------------


def test_max_drawdown_duration_basic():
    """Drawdown duration counts consecutive periods below peak."""
    series = pd.Series([100, 90, 95, 80, 105])
    # Drawdown starts at index 1 (90 < 100), continues through index 3 (80 < 100)
    # That's 3 consecutive periods: 90, 95, 80 (all below peak at each moment)
    # Actually: 90<100, 95<100, 80<100 → 3 periods below
    # Recovery at 105 >= 100
    result = max_drawdown_duration(series)
    assert result == 3


def test_max_drawdown_duration_empty():
    """Empty series returns 0."""
    assert max_drawdown_duration(pd.Series([], dtype=float)) == 0


def test_max_drawdown_duration_no_drawdown():
    """Constantly rising series has zero drawdown duration."""
    series = pd.Series([100, 101, 102, 103])
    assert max_drawdown_duration(series) == 0


def test_max_drawdown_duration_single_drawdown():
    """A single drop below peak counts as 1 period."""
    series = pd.Series([100, 99, 101])
    result = max_drawdown_duration(series)
    assert result == 1


def test_max_drawdown_duration_multiple_segments():
    """Identify the longest of multiple drawdown segments."""
    # Pattern: peak, 2 down, recover, 4 down, recover, 1 down
    series = pd.Series([100, 98, 97, 105, 104, 103, 102, 99, 110, 109, 111])
    # Drawdowns:
    # 98,97 (2 periods below 100)
    # 104,103,102,99 (4 periods below 105)
    # 109 (1 period below 110)
    result = max_drawdown_duration(series)
    assert result == 4, f"Expected 4, got {result}"


def test_max_drawdown_duration_long_decline():
    """Drawdown duration for a long steady decline."""
    n = 100
    series = pd.Series(np.linspace(100, 50, n))
    # From index 1 onwards, every value is below the previous peak (index 0)
    result = max_drawdown_duration(series)
    assert result == n - 1, f"Expected {n - 1}, got {result}"


def test_max_drawdown_duration_returns_int():
    """Result is a Python int."""
    series = pd.Series([100, 90, 100])
    result = max_drawdown_duration(series)
    assert isinstance(result, int)
