"""Unit tests for sample-size validation and MinBTL calculation."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from pysharpe.validation.sample_size import (
    _adjusted_critical_z,
    _annualised_sharpe_se,
    calculate_min_btl,
    evaluate_trade_sample,
)

# ---------------------------------------------------------------------------
# evaluate_trade_sample
# ---------------------------------------------------------------------------


class TestEvaluateTradeSample:
    """Trade-count classification against institutional thresholds."""

    # -- Reject tier (< 30 trades) -------------------------------------------

    def test_reject_below_hard_floor(self):
        result = evaluate_trade_sample(trade_count=5, years_span=2.0)

        assert result.classification == "reject"
        assert result.meets_statistical_floor is False
        assert result.meets_basic_reliability is False
        assert result.meets_institutional_confidence is False
        assert "below the hard statistical floor" in result.recommendation.lower()

    def test_reject_at_boundary_below_floor(self):
        """29 trades is still below the hard floor of 30."""
        result = evaluate_trade_sample(trade_count=29, years_span=5.0)

        assert result.classification == "reject"
        assert result.meets_statistical_floor is False

    def test_meets_floor_at_exactly_30(self):
        result = evaluate_trade_sample(trade_count=30, years_span=1.0)

        assert result.meets_statistical_floor is True
        assert result.classification != "reject"

    # -- High-variance tier (30–99 trades) -----------------------------------

    @pytest.mark.parametrize("trade_count", [30, 50, 99])
    def test_high_variance_tier(self, trade_count):
        result = evaluate_trade_sample(trade_count=trade_count, years_span=1.0)

        assert result.classification == "high_variance"
        assert "high_variance" == result.classification
        assert result.meets_statistical_floor is True
        assert result.meets_basic_reliability is False
        assert result.meets_institutional_confidence is False
        assert "high variance" in result.recommendation.lower()

    # -- Basic tier (100–199 trades) -----------------------------------------

    @pytest.mark.parametrize("trade_count", [100, 150, 199])
    def test_basic_tier(self, trade_count):
        result = evaluate_trade_sample(trade_count=trade_count, years_span=2.0)

        assert result.classification == "basic"
        assert result.meets_statistical_floor is True
        assert result.meets_basic_reliability is True
        assert result.meets_institutional_confidence is False

    # -- Institutional tier (≥200 trades AND ≥3.0 years) ---------------------

    def test_institutional_with_both_conditions_met(self):
        result = evaluate_trade_sample(trade_count=250, years_span=4.0)

        assert result.classification == "institutional"
        assert result.meets_statistical_floor is True
        assert result.meets_basic_reliability is True
        assert result.meets_institutional_confidence is True
        assert "bull, bear, sideways" in result.recommendation.lower()

    def test_200_trades_but_short_duration_stays_basic(self):
        """250 trades over 1.5 years lacks multi-regime duration — a key edge
        case specified in the task requirements."""
        result = evaluate_trade_sample(trade_count=250, years_span=1.5)

        assert result.classification == "basic"
        assert result.meets_basic_reliability is True
        assert result.meets_institutional_confidence is False
        assert "not yet met institutional" in result.recommendation.lower()

    def test_long_duration_but_few_trades_stays_high_variance(self):
        """10 years with only 50 trades — long span but low count."""
        result = evaluate_trade_sample(trade_count=50, years_span=10.0)

        assert result.classification == "high_variance"
        assert result.meets_institutional_confidence is False

    def test_boundary_institutional_years_exact(self):
        """Exactly 3.0 years and 200 trades qualifies."""
        result = evaluate_trade_sample(trade_count=200, years_span=3.0)

        assert result.classification == "institutional"
        assert result.meets_institutional_confidence is True

    def test_500_trades_meets_institutional(self):
        result = evaluate_trade_sample(trade_count=500, years_span=5.0)

        assert result.classification == "institutional"
        assert result.meets_institutional_confidence is True

    # -- Input validation ----------------------------------------------------

    def test_rejects_negative_trade_count(self):
        with pytest.raises(ValueError, match="trade_count"):
            evaluate_trade_sample(trade_count=-1, years_span=2.0)

    def test_rejects_zero_years_span(self):
        with pytest.raises(ValueError, match="years_span"):
            evaluate_trade_sample(trade_count=100, years_span=0.0)

    def test_rejects_negative_years_span(self):
        with pytest.raises(ValueError, match="years_span"):
            evaluate_trade_sample(trade_count=100, years_span=-0.5)

    # -- Dataclass is frozen --------------------------------------------------

    def test_sample_reliability_is_immutable(self):
        result = evaluate_trade_sample(trade_count=30, years_span=1.0)
        with pytest.raises(Exception):  # noqa: B017 — testing dataclass immutability
            result.classification = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _adjusted_critical_z
# ---------------------------------------------------------------------------


class TestAdjustedCriticalZ:
    """Bonferroni-corrected z-score for multiple testing."""

    def test_single_trial_matches_standard_normal(self):
        """With one trial the adjustment should equal the unadjusted quantile."""
        z = _adjusted_critical_z(confidence_level=0.95, num_trials=1)
        expected = norm.ppf(0.95)
        assert z == pytest.approx(expected)

    def test_increases_with_more_trials(self):
        z1 = _adjusted_critical_z(confidence_level=0.95, num_trials=1)
        z10 = _adjusted_critical_z(confidence_level=0.95, num_trials=10)
        z100 = _adjusted_critical_z(confidence_level=0.95, num_trials=100)
        assert z1 < z10 < z100

    def test_higher_confidence_increases_z(self):
        z90 = _adjusted_critical_z(confidence_level=0.90, num_trials=5)
        z99 = _adjusted_critical_z(confidence_level=0.99, num_trials=5)
        assert z90 < z99

    def test_rejects_zero_trials(self):
        with pytest.raises(ValueError):
            _adjusted_critical_z(confidence_level=0.95, num_trials=0)

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValueError):
            _adjusted_critical_z(confidence_level=1.0, num_trials=1)
        with pytest.raises(ValueError):
            _adjusted_critical_z(confidence_level=0.0, num_trials=1)
        with pytest.raises(ValueError):
            _adjusted_critical_z(confidence_level=1.5, num_trials=1)


# ---------------------------------------------------------------------------
# _annualised_sharpe_se
# ---------------------------------------------------------------------------


class TestAnnualisedSharpeSE:
    """Per-√year standard error of the Sharpe ratio estimator."""

    def test_normal_distribution_baseline(self):
        """Under normality (skew=0, excess kurtosis=0), SE matches the
        standard i.i.d. formula: sqrt((1 + 0.5*SR²) / n)."""
        sr = 0.5
        se = _annualised_sharpe_se(sr, skewness=0.0, excess_kurtosis=0.0, periods_per_year=252)
        expected = math.sqrt((1 + 0.5 * sr**2) / 252)
        assert se == pytest.approx(expected)

    def test_negative_skewness_increases_se(self):
        """Negative skewness widens the standard error."""
        se_normal = _annualised_sharpe_se(0.5, 0.0, 0.0, 252)
        se_skewed = _annualised_sharpe_se(0.5, -1.0, 0.0, 252)
        assert se_skewed > se_normal

    def test_positive_excess_kurtosis_increases_se(self):
        """Fat tails (excess kurtosis > 0) widen the standard error."""
        se_normal = _annualised_sharpe_se(0.5, 0.0, 0.0, 252)
        se_fat = _annualised_sharpe_se(0.5, 0.0, 3.0, 252)
        assert se_fat > se_normal

    def test_positive_skewness_decreases_se(self):
        """Positive skewness reduces variance (more upside surprises)."""
        se_normal = _annualised_sharpe_se(0.5, 0.0, 0.0, 252)
        se_pos = _annualised_sharpe_se(0.5, 0.8, 0.0, 252)
        assert se_pos < se_normal

    def test_negative_excess_kurtosis_decreases_se(self):
        """Platykurtic distributions (excess kurtosis < 0) have narrower SE."""
        se_normal = _annualised_sharpe_se(0.5, 0.0, 0.0, 252)
        se_thin = _annualised_sharpe_se(0.5, 0.0, -1.0, 252)
        assert se_thin < se_normal

    def test_non_normal_dominates_with_high_sr(self):
        """At high Sharpe ratios the non-normality terms are magnified by SR²."""
        se_low = _annualised_sharpe_se(0.2, skewness=-1.0, excess_kurtosis=5.0, periods_per_year=252)
        se_high = _annualised_sharpe_se(1.5, skewness=-1.0, excess_kurtosis=5.0, periods_per_year=252)
        # The gap should be larger at high SR.
        gap_low = se_low - _annualised_sharpe_se(0.2, 0.0, 0.0, 252)
        gap_high = se_high - _annualised_sharpe_se(1.5, 0.0, 0.0, 252)
        assert gap_high > gap_low

    def test_zero_sharpe(self):
        se = _annualised_sharpe_se(0.0, skewness=0.0, excess_kurtosis=0.0, periods_per_year=252)
        expected = math.sqrt(1.0 / 252)
        assert se == pytest.approx(expected)

    def test_zero_variance_edge_case(self):
        """When the variance expression becomes negative due to extreme
        parameters, return zero to avoid NaN propagation."""
        # Extreme positive skewness can drive variance_per_obs negative.
        se = _annualised_sharpe_se(0.5, skewness=10.0, excess_kurtosis=0.0, periods_per_year=252)
        assert se == 0.0


# ---------------------------------------------------------------------------
# calculate_min_btl
# ---------------------------------------------------------------------------


class TestCalculateMinBTL:
    """Minimum Backtest Length calculation."""

    def test_normal_baseline_single_trial(self):
        """Under normality with a single trial, verify against the closed-form
        solution for known parameters."""
        target_sr = 0.5
        benchmark_sr = 0.0
        skewness = 0.0
        kurtosis = 0.0  # excess
        num_trials = 1
        confidence = 0.95

        result = calculate_min_btl(
            target_sr, benchmark_sr, skewness, kurtosis, num_trials, confidence
        )

        # Manual calculation.
        z = norm.ppf(confidence)
        variance_per_obs = 1 + 0.5 * target_sr**2
        # SE per √year = sqrt(variance_per_obs / 252)
        se_per_sqrt_year = math.sqrt(variance_per_obs / 252)
        expected = (z * se_per_sqrt_year / (target_sr - benchmark_sr)) ** 2

        assert result == pytest.approx(expected)

    def test_multiple_trials_increase_min_btl(self):
        """More parameter variations → higher z-score → longer MinBTL."""
        btl_1 = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1)
        btl_10 = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=10)
        btl_100 = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=100)
        assert btl_1 < btl_10 < btl_100

    def test_non_normality_increases_min_btl(self):
        """Negative skew + fat tails significantly increase required track
        record length."""
        btl_normal = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1)
        btl_non_normal = calculate_min_btl(0.5, 0.0, -1.5, 5.0, num_trials=1)
        assert btl_non_normal > btl_normal * 1.5  # Substantially larger.

    def test_target_below_benchmark_returns_infinity(self):
        """If the strategy doesn't beat the benchmark, no data can prove it
        does — MinBTL is infinite."""
        result = calculate_min_btl(0.3, 0.5, 0.0, 0.0, num_trials=1)
        assert math.isinf(result)

    def test_target_equals_benchmark_returns_infinity(self):
        result = calculate_min_btl(0.5, 0.5, 0.0, 0.0, num_trials=1)
        assert math.isinf(result)

    def test_higher_target_sharpe_reduces_min_btl(self):
        """A larger SR vs benchmark needs less data to prove superiority."""
        btl_modest = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1)
        btl_strong = calculate_min_btl(1.0, 0.0, 0.0, 0.0, num_trials=1)
        assert btl_strong < btl_modest

    def test_higher_confidence_increases_min_btL(self):
        btl_90 = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, confidence_level=0.90)
        btl_99 = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, confidence_level=0.99)
        assert btl_99 > btl_90

    def test_monthly_frequency(self):
        """Verify periods_per_year=12 yields a larger MinBTL than daily (252).
        Fewer observations per year → higher uncertainty → longer MinBTL."""
        btl_daily = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, periods_per_year=252)
        btl_monthly = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, periods_per_year=12)
        # Monthly has fewer observations/year → larger SE → longer MinBTL.
        assert btl_monthly > btl_daily

    # -- Input validation ----------------------------------------------------

    def test_rejects_zero_trials(self):
        with pytest.raises(ValueError):
            calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=0)

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValueError):
            calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, confidence_level=0.0)
        with pytest.raises(ValueError):
            calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, confidence_level=1.0)

    def test_rejects_negative_periods_per_year(self):
        with pytest.raises(ValueError):
            calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1, periods_per_year=-1)

    # -- Edge cases ----------------------------------------------------------

    def test_extreme_negative_skew_fat_tails(self):
        """A strategy with severe negative skew and massive fat tails should
        require a meaningfully long backtest when the SR is modest."""
        # High multiple testing, hostile distribution, and modest SR.
        btl = calculate_min_btl(0.15, 0.0, -4.0, 40.0, num_trials=1000)
        # Should be a meaningful duration (multiple years).
        assert btl > 4.0

    def test_mathematical_vs_interpretation_distinction(self):
        """MinBTL is the calculated requirement — it's separate from whether
        the user's dataset meets it.  This test just confirms the function
        returns a finite number when the SR exceeds the benchmark."""
        btl = calculate_min_btl(0.5, 0.0, 0.0, 0.0, num_trials=1)
        assert isinstance(btl, float)
        assert math.isfinite(btl)
        assert btl > 0

    def test_result_is_non_negative(self):
        """MinBTL should never be negative."""
        btl = calculate_min_btl(2.0, 0.0, 0.5, -1.0, num_trials=1)
        assert btl >= 0


# ---------------------------------------------------------------------------
# Integration-style scenarios
# ---------------------------------------------------------------------------


class TestCompleteWorkflow:
    """Combine trade-count classification with MinBTL for a holistic view."""

    def test_250_trades_1_5_years_lacks_multi_regime_duration(self):
        """From the task spec: 250 trades over 1.5 years should be flagged as
        lacking multi-regime duration despite meeting the raw trade count
        threshold."""
        reliability = evaluate_trade_sample(trade_count=250, years_span=1.5)

        assert reliability.meets_basic_reliability is True
        assert reliability.meets_institutional_confidence is False
        assert reliability.classification == "basic"

    def test_high_frequency_strategy_meets_institutional(self):
        """500 trades over 5 years — should clear all thresholds."""
        reliability = evaluate_trade_sample(trade_count=500, years_span=5.0)
        assert reliability.classification == "institutional"

        # MinBTL for a reasonable strategy should be far less than 5 years.
        btl = calculate_min_btl(0.8, 0.0, 0.0, 0.0, num_trials=5)
        assert btl < 5.0  # Well within the available history.

    def test_low_count_long_duration_vs_high_count_short_duration(self):
        """Compare two regimes:
        - 40 trades / 10 years → high variance
        - 250 trades / 1.5 years → basic (meets count, lacks duration)
        Neither should reach institutional."""
        scenario_a = evaluate_trade_sample(trade_count=40, years_span=10.0)
        scenario_b = evaluate_trade_sample(trade_count=250, years_span=1.5)

        assert scenario_a.classification == "high_variance"
        assert scenario_b.classification == "basic"
        assert not scenario_a.meets_institutional_confidence
        assert not scenario_b.meets_institutional_confidence
