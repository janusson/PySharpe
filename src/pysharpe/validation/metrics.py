"""Deflated Sharpe Ratio and validation-metric aggregation.

Implements the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR)
framework alongside auxiliary diagnostics: Lo‑adjusted Sharpe (autocorrelation-
corrected annualisation), effective‑trials estimation, and a unified frozen
``ValidationMetrics`` container for the cross‑validation ledger.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
Journal of Portfolio Management, 40(5), 94‑107.

Lo, A. W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts
Journal, 58(4), 36‑52.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pysharpe.metrics import sharpe_ratio

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------

_EULER_MASCHERONI: float = 0.5772156649015328606065120900824024310421
"""Euler–Mascheroni constant γ used in the extreme-value approximation."""

_DEFAULT_LAGS: int = 5
"""Default number of autocorrelation lags for Lo‑adjusted Sharpe."""

# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def compute_dsr(
    observed_sr: float,
    n_trials: int,
    t_obs: int,
    skew: float,
    excess_kurtosis: float,
    sr_std: float = 1.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (DSR).

    The DSR is the probability that the observed Sharpe ratio is
    statistically significant after accounting for the number of
    independent trials and the non‑normality of the return distribution.
    A DSR close to 1.0 indicates the strategy's performance is unlikely
    to be the result of data‑mining bias; a DSR close to 0.0 suggests
    the observed Sharpe could easily arise from noise.

    Parameters
    ----------
    observed_sr
        The annualised Sharpe ratio of the strategy under evaluation.
    n_trials
        Number of **independent** trials (or effective trials from
        ``estimate_effective_trials``).  Must be ≥ 1.
    t_obs
        Number of return observations used to estimate *observed_sr*.
    skew
        Sample skewness of the return distribution (γ₃).
    excess_kurtosis
        Sample **excess** kurtosis (κ − 3) of the return distribution.
        A normal distribution has *excess_kurtosis* = 0.
    sr_std
        Standard deviation of the *annualised* Sharpe ratios across the
        N trials under the null hypothesis.  Defaults to 1.0 per Bailey
        & López de Prado (2014).

    Returns
    -------
    float
        Deflated Sharpe Ratio ∈ [0, 1].  Values below 0.05 are typically
        interpreted as insufficient evidence against the null of no skill.

    Raises
    ------
    ValueError
        If *n_trials* < 1, *t_obs* < 2, or *sr_std* ≤ 0.

    Notes
    -----
    **Expected maximum Sharpe ratio under the null** (SR₀):

    .. math::

        SR_0 = \\sqrt{2 \\ln N} \\Bigl(1 - \\frac{\\gamma}{2 \\ln N}\\Bigr)
               + \\frac{\\gamma}{\\sqrt{2 \\ln N}} \\, \\sigma_{SR}

    where γ ≈ 0.5772 is the Euler–Mascheroni constant and σ_{SR} is
    *sr_std*.  This is an extreme‑value‑theory approximation for the
    expected maximum of N independent standard normal draws.

    **Standard error of the Sharpe ratio** (SE(SR)):

    .. math::

        SE(SR) = \\sqrt{
            \\frac{
                1 - \\gamma_3 \\cdot SR + \\frac{\\gamma_4}{4} \\cdot SR^2
            }{T - 1}
        }

    where T = *t_obs*, γ₃ = *skew*, and γ₄ = *excess_kurtosis*.

    **Deflated Sharpe Ratio**:

    .. math::

        DSR = \\Phi\\!\\left(\\frac{SR - SR_0}{SE(SR)}\\right)

    where Φ is the standard normal CDF.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if n_trials < 1:
        raise ValueError(f"n_trials must be ≥ 1, got {n_trials}")
    if t_obs < 2:
        raise ValueError(f"t_obs must be ≥ 2, got {t_obs}")
    if sr_std <= 0.0:
        raise ValueError(f"sr_std must be positive, got {sr_std}")

    # ------------------------------------------------------------------
    # Null-hypothesis expected maximum Sharpe ratio (SR₀)
    # ------------------------------------------------------------------
    # For N = 1 the expected maximum of a single standard normal draw is
    # trivially 0.  The extreme-value approximation is asymptotic and
    # diverges at ln(1) = 0.
    if n_trials == 1:
        sr_null: float = 0.0
    else:
        log_n: float = math.log(n_trials)
        sqrt_2_log_n: float = math.sqrt(2.0 * log_n)

        # SR₀ = √(2 ln N) · [1 − γ / (2 ln N)] + γ / √(2 ln N) · sr_std
        sr_null = (
            sqrt_2_log_n * (1.0 - _EULER_MASCHERONI / (2.0 * log_n))
            + (_EULER_MASCHERONI / sqrt_2_log_n) * sr_std
        )

    # ------------------------------------------------------------------
    # Standard error of the Sharpe ratio under non‑normality
    # ------------------------------------------------------------------
    # SE² = [1 − γ₃·SR + (γ₄/4)·SR²] / (T − 1)
    variance: float = (
        1.0 - skew * observed_sr + (excess_kurtosis / 4.0) * observed_sr**2
    ) / (t_obs - 1)

    if variance <= 0.0:
        # Degenerate case: if the variance is non-positive the standard
        # error is ill-defined.  Return 0.0 (no evidence against the
        # null — the observed SR cannot be separated from SR₀).
        return 0.0

    se_sr: float = math.sqrt(variance)

    # ------------------------------------------------------------------
    # DSR via standard normal CDF
    # ------------------------------------------------------------------
    z_score: float = (observed_sr - sr_null) / se_sr
    dsr: float = float(norm.cdf(z_score))

    return dsr


# ---------------------------------------------------------------------------
# Effective trials estimation
# ---------------------------------------------------------------------------


def estimate_effective_trials(
    returns_matrix: np.ndarray,
) -> float:
    """Estimate the effective number of independent trials.

    When multiple strategy configurations are tested on overlapping data,
    the trials are not independent.  This function estimates the effective
    number of independent trials N_eff using the average pair‑wise
    correlation across all trial return series.

    Parameters
    ----------
    returns_matrix
        2‑D array of shape ``(T, M)`` where *T* is the number of
        observations and *M* is the number of tested configurations
        (columns).  Each column is the return series of one trial.

    Returns
    -------
    float
        Effective number of independent trials N_eff.  Bounded between
        1.0 (all trials perfectly correlated) and *M* (all trials
        independent).

    Notes
    -----
    The estimation formula from Bailey & López de Prado (2014) is:

    .. math::

        \\bar{\\rho} = \\frac{1}{M(M-1)} \\sum_{i \\neq j} \\rho_{ij}

        N_{eff} = \\bar{\\rho} + (1 - \\bar{\\rho}) \\cdot M

    where ρ̄ is the average absolute pair‑wise Pearson correlation and
    M is the raw number of trials.
    """
    if returns_matrix.ndim != 2:
        raise ValueError(
            f"returns_matrix must be 2-D, got {returns_matrix.ndim} dimensions"
        )

    t_obs, m_trials = returns_matrix.shape

    if m_trials < 2:
        # With 0 or 1 trial, the effective count is the raw count.
        return float(m_trials)

    # Compute the correlation matrix.
    corr_matrix: np.ndarray = np.corrcoef(returns_matrix, rowvar=False)

    # Extract the upper triangle (excluding the diagonal) for pair‑wise
    # absolute correlations.
    triu_indices: tuple[np.ndarray, np.ndarray] = np.triu_indices(m_trials, k=1)
    pair_corrs: np.ndarray = np.abs(corr_matrix[triu_indices])

    # Average pair‑wise correlation.
    rho_bar: float = float(np.nanmean(pair_corrs))

    # Clamp invalid values (NaN from constant series, negative after abs clip).
    if not np.isfinite(rho_bar):
        rho_bar = 0.0
    rho_bar = max(min(rho_bar, 1.0), 0.0)

    # N_eff = ρ̄ + (1 − ρ̄) · M
    n_eff: float = rho_bar + (1.0 - rho_bar) * float(m_trials)
    return n_eff


# ---------------------------------------------------------------------------
# Lo‑adjusted Sharpe ratio (autocorrelation correction)
# ---------------------------------------------------------------------------


def _compute_lo_adjustment_factor(
    returns: np.ndarray,
    max_lags: int = _DEFAULT_LAGS,
) -> float:
    """Compute the Lo (2002) variance‑inflation factor for autocorrelation.

    Parameters
    ----------
    returns
        1‑D array of periodic returns (decimal fractions).
    max_lags
        Maximum lag order for the Newey‑West style autocorrelation
        correction.

    Returns
    -------
    float
        Adjustment factor θ ≥ 1.0.  Values above 1.0 indicate positive
        serial correlation that inflates the IID‑assumed Sharpe ratio.
    """
    n: int = len(returns)
    if n < 2:
        return 1.0

    lags: int = min(max_lags, n - 1)
    if lags < 1:
        return 1.0

    theta: float = 1.0
    for k in range(1, lags + 1):
        # Correlation at lag k
        rho_k: float = float(np.corrcoef(returns[:-k], returns[k:])[0, 1])
        if not np.isfinite(rho_k):
            rho_k = 0.0
        # Newey‑West kernel weight: w_k = 1 − k / (q + 1)
        weight: float = 1.0 - k / (lags + 1)
        theta += 2.0 * weight * rho_k

    # θ cannot be less than 1.0 (zero or negative autocorrelation does
    # not reduce the variance inflation).
    return max(theta, 1.0)


def _lo_adjusted_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    max_lags: int = _DEFAULT_LAGS,
) -> float:
    """Compute the Lo‑adjusted annualised Sharpe ratio.

    The standard Sharpe ratio assumes IID returns.  When returns exhibit
    serial correlation the annualisation factor √T overstates the true
    information ratio.  Lo (2002) shows that the IID‑assumed Sharpe
    should be deflated by √θ where θ accounts for the autocorrelation
    structure.

    Parameters
    ----------
    returns
        1‑D array of periodic returns (decimal fractions).
    risk_free_rate
        Annual risk‑free rate as a decimal.
    periods_per_year
        Observation frequency (252 for daily, 12 for monthly).
    max_lags
        Maximum autocorrelation lag for the variance‑inflation factor.

    Returns
    -------
    float
        Lo‑adjusted annualised Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    # Annualised Sharpe under the IID assumption.
    import pandas as pd

    sr_raw: float = sharpe_ratio(
        pd.Series(returns, dtype=float),
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    theta: float = _compute_lo_adjustment_factor(returns, max_lags=max_lags)
    return sr_raw / math.sqrt(theta)


# ---------------------------------------------------------------------------
# ValidationMetrics container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationMetrics:
    """Aggregate cross‑validation diagnostics for a strategy.

    This immutable container collects the four key overfitting and
    statistical‑reliability metrics that the validation ledger tracks
    for each strategy configuration.

    Attributes
    ----------
    raw_sharpe
        Annualised Sharpe ratio under the standard IID assumption.
        Computed via ``pysharpe.metrics.sharpe_ratio``.
    lo_adjusted_sharpe
        Lo (2002) autocorrelation‑adjusted annualised Sharpe ratio.
        When returns are positively serially correlated this value will
        be lower than *raw_sharpe*.
    pbo
        Probability of Backtest Overfitting ∈ [0, 1] from the logit‑
        transformed Spearman rank correlation between in‑sample and
        out‑of‑sample performance.  Values near 0.5 indicate no
        predictive relationship (i.e. overfitting).  See
        ``pysharpe.validation.ledger.compute_pbo``.
    dsr
        Deflated Sharpe Ratio ∈ [0, 1].  The probability that the
        observed Sharpe ratio is statistically significant after
        accounting for multiple testing and non‑normality.  See
        ``compute_dsr``.

    Notes
    -----
    All fields are plain ``float`` values.  The consumer is responsible
    for interpreting cut‑offs (e.g. DSR < 0.05 → reject, PBO > 0.5 →
    likely overfit).
    """

    raw_sharpe: float
    lo_adjusted_sharpe: float
    pbo: float
    dsr: float

    def __post_init__(self) -> None:
        """Validate metric ranges."""
        if not (0.0 <= self.pbo <= 1.0):
            raise ValueError(f"pbo must be in [0, 1], got {self.pbo}")
        if not (0.0 <= self.dsr <= 1.0):
            raise ValueError(f"dsr must be in [0, 1], got {self.dsr}")


def compute_validation_metrics(
    returns: np.ndarray,
    n_trials: float,
    pbo: float,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    max_lags: int = _DEFAULT_LAGS,
    sr_std: float = 1.0,
) -> ValidationMetrics:
    """Convenience factory that computes all ``ValidationMetrics`` fields.

    Parameters
    ----------
    returns
        1‑D array of periodic returns (decimal fractions) for the
        strategy under evaluation.
    n_trials
        Number of independent (or effective) trials tested.  Pass the
        output of ``estimate_effective_trials`` when trials are correlated.
    pbo
        Pre‑computed Probability of Backtest Overfitting from
        ``pysharpe.validation.ledger.compute_pbo``.
    risk_free_rate
        Annual risk‑free rate as a decimal.
    periods_per_year
        Observation frequency (252 for daily, 12 for monthly).
    max_lags
        Maximum lag order for the Lo‑adjusted Sharpe autocorrelation
        correction.
    sr_std
        Standard deviation of the *annualised* Sharpe ratios across
        trials (default 1.0 per Bailey & López de Prado).

    Returns
    -------
    ValidationMetrics
        Frozen record with all four metrics populated.
    """
    import pandas as pd

    ret_series: pd.Series = pd.Series(returns, dtype=float)

    # Raw Sharpe ratio.
    raw_sr: float = sharpe_ratio(
        ret_series,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    # Lo‑adjusted Sharpe.
    lo_sr: float = _lo_adjusted_sharpe(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        max_lags=max_lags,
    )

    # Distribution moments for DSR standard‑error calculation.
    skew: float = float(ret_series.skew())
    # pandas .kurtosis() returns *excess* kurtosis (Fisher definition).
    excess_kurt: float = float(ret_series.kurtosis())

    t_obs: int = len(returns)
    n_trials_int: int = max(int(round(n_trials)), 1)

    dsr: float = compute_dsr(
        observed_sr=raw_sr,
        n_trials=n_trials_int,
        t_obs=t_obs,
        skew=skew,
        excess_kurtosis=excess_kurt,
        sr_std=sr_std,
    )

    return ValidationMetrics(
        raw_sharpe=raw_sr,
        lo_adjusted_sharpe=lo_sr,
        pbo=pbo,
        dsr=dsr,
    )


__all__ = [
    "ValidationMetrics",
    "compute_dsr",
    "compute_validation_metrics",
    "estimate_effective_trials",
]
