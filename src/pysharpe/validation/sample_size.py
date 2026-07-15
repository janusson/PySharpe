"""Sample-size validation and Minimum Backtest Length (MinBTL) calculation.

This module provides two complementary tools for evaluating the statistical
reliability of backtest results:

1. **Trade-count classification** (`evaluate_trade_sample`) — a rules-based
   system that flags whether a strategy has generated enough independent
   observations to trust its performance metrics.

2. **Minimum Backtest Length** (`calculate_min_btl`) — a parametric method
   based on Bailey & López de Prado's work that computes the required track
   record length (in years) needed to prove that an estimated Sharpe ratio
   exceeds a benchmark, accounting for non-normal return distributions and
   multiple-testing corrections.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
Journal of Portfolio Management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

# ---------------------------------------------------------------------------
# Trade-count classification thresholds
# ---------------------------------------------------------------------------

_TRADE_COUNT_HARD_FLOOR: int = 30
_TRADE_COUNT_HIGH_VARIANCE: int = 100
_TRADE_COUNT_BASIC: int = 200
_INSTITUTIONAL_MIN_YEARS: float = 3.0

# Periods-per-year constant used in MinBTL (daily trading data).
_PERIODS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# SampleReliability data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleReliability:
    """Classification of a strategy's trade sample for backtest reliability.

    Attributes
    ----------
    classification
        The reliability tier as a human-readable label.
    trade_count
        Number of independent trades executed over the backtest window.
    years_span
        Calendar duration covered by the backtest (in years).
    meets_statistical_floor
        ``True`` when *trade_count* ≥ 30 (the minimum for any inference).
    meets_basic_reliability
        ``True`` when *trade_count* ≥ 100.
    meets_institutional_confidence
        ``True`` when *trade_count* ≥ 200 **and** *years_span* ≥ 3.0, ensuring
        multi-regime exposure (bull, bear, sideways).
    recommendation
        A plain-English interpretation of the classification.
    """

    classification: str
    trade_count: int
    years_span: float
    meets_statistical_floor: bool
    meets_basic_reliability: bool
    meets_institutional_confidence: bool
    recommendation: str


# ---------------------------------------------------------------------------
# evaluate_trade_sample
# ---------------------------------------------------------------------------


def evaluate_trade_sample(trade_count: int, years_span: float) -> SampleReliability:
    """Classify trade-sample reliability based on institutional thresholds.

    Parameters
    ----------
    trade_count
        Total number of independent trades in the backtest window.
    years_span
        Calendar duration spanned by the backtest (e.g. 2.5 for 2½ years).

    Returns
    -------
    SampleReliability
        A frozen record with boolean flags and a plain-English recommendation.

    Raises
    ------
    ValueError
        If *trade_count* is negative or *years_span* is not positive.

    Notes
    -----
    The institutional thresholds are calibrated for daily-frequency equity
    strategies but are commonly applied to any strategy that generates
    independent entry/exit signals.
    """
    if trade_count < 0:
        raise ValueError("trade_count must be non-negative.")
    if years_span <= 0:
        raise ValueError("years_span must be positive.")

    meets_floor: bool = trade_count >= _TRADE_COUNT_HARD_FLOOR
    meets_basic: bool = trade_count >= _TRADE_COUNT_HIGH_VARIANCE
    meets_inst: bool = (
        trade_count >= _TRADE_COUNT_BASIC and years_span >= _INSTITUTIONAL_MIN_YEARS
    )

    # Determine the classification label and recommendation.
    if not meets_floor:
        classification: str = "reject"
        recommendation: str = (
            f"Only {trade_count} trade(s) observed — below the hard statistical "
            f"floor of {_TRADE_COUNT_HARD_FLOOR}. The sample is too small to "
            f"reliably estimate any performance metric. Increase the backtest "
            f"window or trade frequency."
        )
    elif not meets_inst:
        if not meets_basic:
            classification = "high_variance"
            recommendation = (
                f"{trade_count} trades observed — the statistical floor "
                f"({_TRADE_COUNT_HARD_FLOOR}) is met, but fewer than "
                f"{_TRADE_COUNT_HIGH_VARIANCE} trades means metric estimates "
                f"carry high variance. Proceed with caution; confidence "
                f"intervals will be wide."
            )
        else:
            classification = "basic"
            recommendation = (
                f"{trade_count} trades observed over {years_span:.1f} years — "
                f"basic metric reliability is established, but the sample has "
                f"not yet met institutional multi-regime standards "
                f"(≥{_TRADE_COUNT_BASIC} trades AND "
                f"≥{_INSTITUTIONAL_MIN_YEARS:.1f} years). "
                f"The results may not generalise across full market cycles."
            )
    else:
        classification = "institutional"
        recommendation = (
            f"{trade_count} trades observed over {years_span:.1f} years — "
            f"institutional confidence established. The sample spans multiple "
            f"market regimes (bull, bear, sideways), providing a robust basis "
            f"for inference."
        )

    return SampleReliability(
        classification=classification,
        trade_count=trade_count,
        years_span=years_span,
        meets_statistical_floor=meets_floor,
        meets_basic_reliability=meets_basic,
        meets_institutional_confidence=meets_inst,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# calculate_min_btl
# ---------------------------------------------------------------------------


def _adjusted_critical_z(
    confidence_level: float,
    num_trials: int,
) -> float:
    """Return the Bonferroni-adjusted z-score for the one-sided test.

    The adjustment accounts for the fact that the researcher may have tried
    *num_trials* independent parameter variations and is reporting the best
    result.  Without this correction the probability of a false discovery
    grows with the number of trials.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")

    # Per-trial significance level under Bonferroni correction.
    alpha_adj: float = (1.0 - confidence_level) / num_trials
    return float(norm.ppf(1.0 - alpha_adj))


def _annualised_sharpe_se(
    sharpe: float,
    skewness: float,
    excess_kurtosis: float,
    periods_per_year: int,
) -> float:
    """Standard error of the annualised Sharpe ratio under non-normality.

    Under the standard i.i.d. normal assumption the asymptotic variance of the
    Sharpe ratio estimator is ``(1 + 0.5·SR²) / T``.  When returns exhibit
    skewness (γ₃) and excess kurtosis (γ₄ = κ − 3), the variance expands to::

        Var[SR̂] ≈ (1 + 0.5·SR² − γ₃·SR + (γ₄/4)·SR²) / (n · Y)

    where *n* is the number of observations per year and *Y* is the track-record
    length in years.  This function returns the square root of that variance
    **without** dividing by years, i.e. the per-√year standard error.
    """
    # Asymptotic variance of SR per observation (see Lo 2002, Bailey & López de
    # Prado 2014 for the non-normality extension).
    variance_per_obs: float = (
        1.0 + 0.5 * sharpe**2 - skewness * sharpe + (excess_kurtosis / 4.0) * sharpe**2
    )
    # Scale to annual frequency: more observations per year → lower variance.
    variance_per_year: float = variance_per_obs / periods_per_year

    if variance_per_year <= 0:
        return 0.0

    return math.sqrt(variance_per_year)


def calculate_min_btl(
    target_sharpe: float,
    benchmark_sharpe: float,
    skewness: float,
    kurtosis: float,
    num_trials: int,
    confidence_level: float = 0.95,
    periods_per_year: int = _PERIODS_PER_YEAR,
) -> float:
    """Compute the Minimum Backtest Length (MinBTL) in years.

    The MinBTL is the number of years of track record required to reject the
    null hypothesis that the true Sharpe ratio equals *benchmark_sharpe* in
    favour of the alternative that it exceeds it, at the given *confidence_level*
    and after adjusting for *num_trials* independent parameter variations.

    The formula accounts for non-normality in the return distribution through
    the third (skewness) and fourth (kurtosis) central moments.

    Parameters
    ----------
    target_sharpe
        The estimated annualised Sharpe ratio of the strategy.
    benchmark_sharpe
        The annualised Sharpe ratio of the benchmark (or the threshold to beat).
    skewness
        Skewness of the strategy's return distribution.
    kurtosis
        **Excess** kurtosis of the strategy's return distribution
        (κ − 3 where κ is Pearson kurtosis).  A value > 0 indicates
        heavier tails than the normal distribution.
    num_trials
        Number of independent parameter variations tested.  Used to apply a
        Bonferroni-type multiple-testing correction.
    confidence_level
        Desired confidence level for the test (default 0.95).
    periods_per_year
        Observation frequency of the underlying return data
        (252 for daily, 12 for monthly).

    Returns
    -------
    float
        Minimum backtest length in years.  Returns ``math.inf`` when the
        estimated Sharpe ratio does not exceed the benchmark (i.e. the null
        hypothesis can never be rejected regardless of sample size).

    Raises
    ------
    ValueError
        If *num_trials* ≤ 0, *confidence_level* is outside (0, 1), or
        *periods_per_year* ≤ 0.

    Notes
    -----
    - The function returns the **mathematical** MinBTL.  Whether the user's
      data meets this requirement is a separate interpretation step that
      callers must perform.
    - Negative skewness and positive excess kurtosis both **increase** the
      required MinBTL because they widen the standard error of the Sharpe
      ratio estimate.
    - The Bonferroni correction is conservative.  When *num_trials* is large
      consider whether trials are truly independent — correlated tests inflate
      the effective number of trials less severely.

    References
    ----------
    Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio."
    Journal of Portfolio Management, 40(5), 94-107.

    Lo, A. W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts
    Journal, 58(4), 36-52.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    sharpe_diff: float = target_sharpe - benchmark_sharpe
    if sharpe_diff <= 0:
        # The strategy does not outperform the benchmark; no amount of data
        # can make the estimated difference statistically significant.
        return math.inf

    # Excess kurtosis = kurtosis - 3 (the parameter is already excess).
    excess_kurtosis: float = kurtosis

    z_crit: float = _adjusted_critical_z(confidence_level, num_trials)
    se_per_sqrt_year: float = _annualised_sharpe_se(
        target_sharpe, skewness, excess_kurtosis, periods_per_year
    )

    if se_per_sqrt_year == 0:
        return 0.0

    # MinBTL = (z_crit * SE_per_√year / ΔSR)²
    min_btl: float = (z_crit * se_per_sqrt_year / sharpe_diff) ** 2
    return max(min_btl, 0.0)


__all__ = [
    "SampleReliability",
    "evaluate_trade_sample",
    "calculate_min_btl",
]
