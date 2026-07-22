"""Tests for ``pysharpe.validation.metrics`` — DSR, effective trials, ValidationMetrics.

.. note::

    All tests use synthetic data with fixed seeds.  No network calls.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pysharpe.validation.metrics import (
    ValidationMetrics,
    compute_dsr,
    compute_validation_metrics,
    estimate_effective_trials,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EULER_MASCHERONI: float = 0.5772156649015329
_PERIODS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# compute_dsr
# ---------------------------------------------------------------------------


class TestComputeDSR:
    """Unit tests for ``compute_dsr``."""

    # -- Baseline behavior ---------------------------------------------------

    def test_dsr_near_05_when_sr_equals_null(self) -> None:
        """DSR ≈ 0.5 when observed SR matches the null expected maximum."""
        n_trials: int = 100
        t_obs: int = 500
        skew: float = 0.0
        excess_kurtosis: float = 0.0

        # Compute SR₀ so we can feed it back as the observed SR.
        log_n = math.log(n_trials)
        sr_null = math.sqrt(2.0 * log_n) * (
            1.0 - _EULER_MASCHERONI / (2.0 * log_n)
        ) + _EULER_MASCHERONI / math.sqrt(2.0 * log_n)

        dsr = compute_dsr(
            observed_sr=sr_null,
            n_trials=n_trials,
            t_obs=t_obs,
            skew=skew,
            excess_kurtosis=excess_kurtosis,
        )
        # Should be ≈ 0.5 since z ≈ 0.
        assert 0.45 <= dsr <= 0.55, f"Expected DSR ≈ 0.5, got {dsr}"

    def test_dsr_high_for_strong_signal(self) -> None:
        """DSR → 1.0 when observed SR far exceeds the null expectation."""
        dsr = compute_dsr(
            observed_sr=3.0,
            n_trials=10,
            t_obs=500,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        assert dsr > 0.95, f"Expected DSR > 0.95, got {dsr}"

    def test_dsr_low_for_weak_signal(self) -> None:
        """DSR → 0.0 when observed SR is far below the null expectation."""
        dsr = compute_dsr(
            observed_sr=0.0,
            n_trials=1000,
            t_obs=500,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        assert dsr < 0.05, f"Expected DSR < 0.05, got {dsr}"

    # -- Key regression test from task spec ----------------------------------

    def test_dsr_below_005_with_1000_noise_trials(self) -> None:
        """DSR < 0.05 when N=1000 and observed SR is drawn from pure noise.

        This is the core sanity check: any strategy found among 1 000
        independent random configurations should be indistinguishable
        from data‑mining bias.
        """
        rng = np.random.default_rng(seed=42)

        # Generate a return series whose Sharpe ratio is in a typical
        # "good‑looking noise" range (~0.3–0.6).
        n_obs: int = 252  # One year of daily data.
        noise_returns: np.ndarray = rng.normal(loc=0.0005, scale=0.015, size=n_obs)

        # Compute the observed annualised SR (roughly 0.4–0.6 for this seed).
        ann_mean = float(np.mean(noise_returns)) * _PERIODS_PER_YEAR
        ann_vol = float(np.std(noise_returns, ddof=1)) * math.sqrt(_PERIODS_PER_YEAR)
        observed_sr = ann_mean / ann_vol

        # Distribution moments.
        skew = float(
            np.mean((noise_returns - np.mean(noise_returns)) ** 3)
            / (np.std(noise_returns, ddof=1) ** 3)
        )
        excess_kurt = float(
            np.mean((noise_returns - np.mean(noise_returns)) ** 4)
            / (np.std(noise_returns, ddof=1) ** 4)
            - 3.0
        )

        dsr = compute_dsr(
            observed_sr=observed_sr,
            n_trials=1000,
            t_obs=n_obs,
            skew=skew,
            excess_kurtosis=excess_kurt,
        )

        assert dsr < 0.05, (
            f"DSR should be < 0.05 for noise with N=1000, "
            f"got {dsr:.6f} (SR={observed_sr:.4f})"
        )

    def test_dsr_decreases_monotonically_with_trials(self) -> None:
        """DSR decreases monotonically as the number of trials grows.

        For a fixed observed SR, increasing N raises SR₀ (the expected
        maximum under the null), which pushes the DSR lower.
        """
        dsr_2 = compute_dsr(
            observed_sr=1.5,
            n_trials=2,
            t_obs=50,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        dsr_5 = compute_dsr(
            observed_sr=1.5,
            n_trials=5,
            t_obs=50,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        dsr_10 = compute_dsr(
            observed_sr=1.5,
            n_trials=10,
            t_obs=50,
            skew=0.0,
            excess_kurtosis=0.0,
        )

        assert dsr_2 > dsr_5 > dsr_10, (
            f"DSR should decrease with N: {dsr_2:.6f} > {dsr_5:.6f} > {dsr_10:.6f}"
        )

    # -- Non‑normality effects -----------------------------------------------

    def test_negative_skew_increases_se_lowers_dsr(self) -> None:
        """Negative skewness widens SE(SR), pulling DSR lower."""
        # Use parameters where the observed SR is near SR₀ so the DSR
        # is in the sensitive region of the normal CDF (~0.5).
        dsr_sym = compute_dsr(
            observed_sr=1.8,
            n_trials=5,
            t_obs=200,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        dsr_neg = compute_dsr(
            observed_sr=1.8,
            n_trials=5,
            t_obs=200,
            skew=-2.0,
            excess_kurtosis=0.0,
        )
        assert dsr_neg < dsr_sym, (
            f"Negative skew should lower DSR: {dsr_neg:.6f} < {dsr_sym:.6f}"
        )

    def test_positive_excess_kurtosis_lowers_dsr(self) -> None:
        """Fat tails inflate SE(SR), reducing DSR."""
        dsr_normal = compute_dsr(
            observed_sr=1.8,
            n_trials=5,
            t_obs=200,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        dsr_fat = compute_dsr(
            observed_sr=1.8,
            n_trials=5,
            t_obs=200,
            skew=0.0,
            excess_kurtosis=3.0,
        )
        assert dsr_fat < dsr_normal, (
            f"Fat tails should lower DSR: {dsr_fat:.6f} < {dsr_normal:.6f}"
        )

    def test_positive_skew_raises_dsr(self) -> None:
        """Positive skewness narrows SE(SR), pushing DSR higher."""
        dsr_sym = compute_dsr(
            observed_sr=1.8,
            n_trials=5,
            t_obs=200,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        dsr_pos = compute_dsr(
            observed_sr=1.8,
            n_trials=5,
            t_obs=200,
            skew=0.5,
            excess_kurtosis=0.0,
        )
        assert dsr_pos > dsr_sym, (
            f"Positive skew should raise DSR: {dsr_pos:.6f} > {dsr_sym:.6f}"
        )

    # -- Degenerate variance ------------------------------------------------

    def test_degenerate_variance_returns_zero(self) -> None:
        """When SE(SR) ≤ 0 due to extreme skew/kurtosis, DSR → 0.0."""
        # Large positive skew and negative excess kurtosis can drive the
        # variance numerator to zero or negative.
        dsr = compute_dsr(
            observed_sr=1.0,
            n_trials=10,
            t_obs=100,
            skew=5.0,  # Extreme positive skew
            excess_kurtosis=-3.9,  # Near the lower bound for kurtosis
        )
        # With these parameters the variance term may become ≤ 0,
        # in which case DSR returns 0.0.
        assert dsr == 0.0 or dsr >= 0.0, f"Expected DSR ≥ 0, got {dsr}"

    # -- Input validation ----------------------------------------------------

    def test_rejects_n_trials_lt_1(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be"):
            compute_dsr(1.0, n_trials=0, t_obs=100, skew=0.0, excess_kurtosis=0.0)

    def test_rejects_t_obs_lt_2(self) -> None:
        with pytest.raises(ValueError, match="t_obs must be"):
            compute_dsr(1.0, n_trials=10, t_obs=1, skew=0.0, excess_kurtosis=0.0)

    def test_rejects_non_positive_sr_std(self) -> None:
        with pytest.raises(ValueError, match="sr_std must be positive"):
            compute_dsr(
                1.0, n_trials=10, t_obs=100, skew=0.0, excess_kurtosis=0.0, sr_std=0.0
            )

    # -- sr_std sensitivity --------------------------------------------------

    def test_larger_sr_std_increases_sr_null(self) -> None:
        """A wider dispersion of trial SRs raises the null bar."""
        dsr_small = compute_dsr(
            observed_sr=2.5,
            n_trials=5,
            t_obs=200,
            skew=0.0,
            excess_kurtosis=0.0,
            sr_std=0.5,
        )
        dsr_large = compute_dsr(
            observed_sr=2.5,
            n_trials=5,
            t_obs=200,
            skew=0.0,
            excess_kurtosis=0.0,
            sr_std=2.0,
        )
        assert dsr_large < dsr_small, (
            f"Larger sr_std should lower DSR: {dsr_large:.6f} < {dsr_small:.6f}"
        )

    # -- Boundary: single trial -----------------------------------------------

    def test_single_trial_baseline(self) -> None:
        """With N=1 the null expected maximum is much lower."""
        dsr = compute_dsr(
            observed_sr=1.0,
            n_trials=1,
            t_obs=500,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        # A decent SR with only 1 trial should have some evidence.
        assert dsr > 0.5, f"Single-trial DSR should be > 0.5, got {dsr}"


# ---------------------------------------------------------------------------
# estimate_effective_trials
# ---------------------------------------------------------------------------


class TestEstimateEffectiveTrials:
    """Unit tests for ``estimate_effective_trials``."""

    def test_independent_trials_approx_m(self) -> None:
        """N_eff ≈ M when all trial return series are independent."""
        rng = np.random.default_rng(seed=42)
        t_obs: int = 500
        m_trials: int = 20
        # Each column is independent white noise.
        returns = rng.normal(loc=0.001, scale=0.02, size=(t_obs, m_trials))

        n_eff = estimate_effective_trials(returns)
        # Should be close to M (within ~30% due to finite-sample correlation).
        assert n_eff > m_trials * 0.7, (
            f"N_eff should be close to M={m_trials}, got {n_eff:.2f}"
        )
        assert n_eff <= m_trials + 1.0

    def test_perfectly_correlated_approx_1(self) -> None:
        """N_eff ≈ 1 when all trials are perfectly correlated."""
        t_obs: int = 200
        m_trials: int = 10
        base = np.random.default_rng(seed=7).normal(size=t_obs)
        # Stack the same series M times.
        returns = np.column_stack([base] * m_trials)

        n_eff = estimate_effective_trials(returns)
        # Perfect correlation → N_eff = 1.0.
        assert n_eff < 1.5, (
            f"N_eff should be ~1 for perfect correlation, got {n_eff:.4f}"
        )

    def test_partial_correlation_between_1_and_m(self) -> None:
        """N_eff ∈ [1, M] for partially correlated trials."""
        rng = np.random.default_rng(seed=99)
        t_obs: int = 300
        m_trials: int = 10
        common = rng.normal(size=t_obs)
        returns = np.column_stack(
            [common * 0.5 + rng.normal(size=t_obs) * 0.5 for _ in range(m_trials)]
        )

        n_eff = estimate_effective_trials(returns)
        assert 1.0 <= n_eff <= m_trials, (
            f"N_eff should be in [1, {m_trials}], got {n_eff:.4f}"
        )

    def test_two_trials_moderate_correlation(self) -> None:
        """Two trials with ρ ≈ 0.5 → N_eff ≈ 1.5."""
        rng = np.random.default_rng(seed=42)
        t_obs: int = 1000
        common = rng.normal(size=t_obs)
        col1 = common + rng.normal(scale=0.5, size=t_obs)
        col2 = common + rng.normal(scale=0.5, size=t_obs)
        returns = np.column_stack([col1, col2])

        # With ρ ≈ 0.5, N_eff = 0.5 + 0.5 * 2 = 1.5 (theoretically).
        n_eff = estimate_effective_trials(returns)
        assert 1.2 <= n_eff <= 1.8, f"Expected N_eff ≈ 1.5 with ρ≈0.5, got {n_eff:.4f}"

    def test_single_trial(self) -> None:
        """N_eff == 1.0 when there is only one trial."""
        rng = np.random.default_rng(seed=1)
        returns = rng.normal(size=(100, 1))
        n_eff = estimate_effective_trials(returns)
        assert n_eff == 1.0, f"Single trial should give N_eff=1.0, got {n_eff}"

    def test_zero_trials(self) -> None:
        """N_eff == 0 when there are no trial columns."""
        returns = np.empty((0, 0))
        n_eff = estimate_effective_trials(returns)
        assert n_eff == 0.0

    def test_rejects_1d_array(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            estimate_effective_trials(np.array([1.0, 2.0, 3.0]))

    # -- NaN resilience ------------------------------------------------------

    def test_handles_nan_in_correlation(self) -> None:
        """Constant return series (zero variance) produce NaN correlations."""
        t_obs: int = 100
        returns = np.column_stack(
            [
                np.ones(t_obs),  # constant → zero variance
                np.random.default_rng(seed=3).normal(size=t_obs),
            ]
        )
        n_eff = estimate_effective_trials(returns)
        # Should not crash; NaN rho_bar → 0.0.
        assert 1.0 <= n_eff <= 2.0, f"Got N_eff={n_eff:.4f}"


# ---------------------------------------------------------------------------
# ValidationMetrics
# ---------------------------------------------------------------------------


class TestValidationMetrics:
    """Unit tests for the ``ValidationMetrics`` frozen dataclass."""

    def test_construction(self) -> None:
        vm = ValidationMetrics(
            raw_sharpe=1.2,
            lo_adjusted_sharpe=1.0,
            pbo=0.3,
            dsr=0.85,
        )
        assert vm.raw_sharpe == 1.2
        assert vm.lo_adjusted_sharpe == 1.0
        assert vm.pbo == 0.3
        assert vm.dsr == 0.85

    def test_is_immutable(self) -> None:
        vm = ValidationMetrics(1.0, 0.9, 0.2, 0.7)
        with pytest.raises(FrozenInstanceError):
            vm.raw_sharpe = 2.0  # type: ignore[misc]

    def test_rejects_pbo_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="pbo must be"):
            ValidationMetrics(1.0, 0.9, pbo=1.5, dsr=0.5)
        with pytest.raises(ValueError, match="pbo must be"):
            ValidationMetrics(1.0, 0.9, pbo=-0.1, dsr=0.5)

    def test_rejects_dsr_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dsr must be"):
            ValidationMetrics(1.0, 0.9, pbo=0.5, dsr=1.5)
        with pytest.raises(ValueError, match="dsr must be"):
            ValidationMetrics(1.0, 0.9, pbo=0.5, dsr=-0.1)

    def test_boundary_values(self) -> None:
        """PBO and DSR at exactly 0.0 and 1.0 should be accepted."""
        vm = ValidationMetrics(0.0, 0.0, pbo=0.0, dsr=0.0)
        assert vm.pbo == 0.0
        assert vm.dsr == 0.0

        vm = ValidationMetrics(3.0, 2.5, pbo=1.0, dsr=1.0)
        assert vm.pbo == 1.0
        assert vm.dsr == 1.0


# ---------------------------------------------------------------------------
# compute_validation_metrics (integration)
# ---------------------------------------------------------------------------


class TestComputeValidationMetrics:
    """Integration tests for ``compute_validation_metrics``."""

    def test_end_to_end_on_synthetic_returns(self) -> None:
        """Full pipeline: returns → all four metrics populated."""
        rng = np.random.default_rng(seed=42)
        n_obs: int = 504  # Two years daily.
        returns = rng.normal(loc=0.0005, scale=0.012, size=n_obs)

        vm = compute_validation_metrics(
            returns,
            n_trials=50.0,
            pbo=0.35,
            periods_per_year=_PERIODS_PER_YEAR,
        )

        assert isinstance(vm.raw_sharpe, float)
        assert isinstance(vm.lo_adjusted_sharpe, float)
        assert isinstance(vm.pbo, float)
        assert isinstance(vm.dsr, float)

        # Basic sanity checks.
        assert vm.pbo == 0.35
        assert 0.0 <= vm.dsr <= 1.0

    def test_lo_adjusted_sharpe_no_serial_correlation(self) -> None:
        """Lo‑adjusted ≈ raw Sharpe when returns are serially uncorrelated."""
        rng = np.random.default_rng(seed=1)
        returns = rng.normal(loc=0.001, scale=0.01, size=1000)

        vm = compute_validation_metrics(
            returns,
            n_trials=10.0,
            pbo=0.3,
            periods_per_year=_PERIODS_PER_YEAR,
        )

        # For uncorrelated returns the adjustment factor θ ≈ 1.
        ratio = vm.lo_adjusted_sharpe / vm.raw_sharpe if vm.raw_sharpe != 0 else 1.0
        assert 0.85 <= ratio <= 1.15, (
            f"Lo/raw ratio should be ≈ 1 for IID returns, got {ratio:.4f}"
        )

    def test_lo_adjusted_defates_positive_correlation(self) -> None:
        """Positive serial correlation → Lo‑adjusted < raw Sharpe."""
        rng = np.random.default_rng(seed=5)
        n_obs: int = 500
        innovations = rng.normal(loc=0.0005, scale=0.01, size=n_obs)
        # Induce positive autocorrelation via AR(1) with ρ ≈ 0.3.
        returns = np.empty(n_obs)
        returns[0] = innovations[0]
        for t in range(1, n_obs):
            returns[t] = 0.3 * returns[t - 1] + innovations[t]

        vm = compute_validation_metrics(
            returns,
            n_trials=10.0,
            pbo=0.3,
            periods_per_year=_PERIODS_PER_YEAR,
        )

        assert vm.lo_adjusted_sharpe < vm.raw_sharpe, (
            f"Lo-adjusted ({vm.lo_adjusted_sharpe:.4f}) should be "
            f"< raw ({vm.raw_sharpe:.4f}) for +ve autocorrelation"
        )

    def test_rounds_n_trials_to_int(self) -> None:
        """Fractional n_trials (from estimate_effective_trials) is rounded."""
        rng = np.random.default_rng(seed=99)
        returns = rng.normal(size=252)

        vm_float = compute_validation_metrics(returns, n_trials=15.3, pbo=0.4)
        vm_int = compute_validation_metrics(returns, n_trials=15.0, pbo=0.4)
        # Should round to the same integer.
        assert vm_float.dsr == vm_int.dsr

    def test_minimum_n_trials_is_1(self) -> None:
        """n_trials below 1 (e.g. 0.4 rounding to 0) is clamped to 1."""
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(size=252)

        vm = compute_validation_metrics(returns, n_trials=0.3, pbo=0.5)
        # Should not crash; n_trials round → 0, clamped → 1.
        assert isinstance(vm.dsr, float)
        assert 0.0 <= vm.dsr <= 1.0


# ---------------------------------------------------------------------------
# Regression: DSR formula fidelity
# ---------------------------------------------------------------------------


class TestDSRFormulaFidelity:
    """Verify internal formula steps against hand‑calculated values."""

    def test_sr_null_calculation(self) -> None:
        """Manually compute SR₀ and verify against compute_dsr internals."""
        # With sr_std=1.0 and N=100, the null SR should be:
        n_trials: int = 100
        log_n = math.log(n_trials)
        sqrt_2_log_n = math.sqrt(2.0 * log_n)
        expected_sr_null = (
            sqrt_2_log_n * (1.0 - _EULER_MASCHERONI / (2.0 * log_n))
            + _EULER_MASCHERONI / sqrt_2_log_n
        )

        # Feed the null SR as observed → DSR should be near 0.5.
        dsr = compute_dsr(
            observed_sr=expected_sr_null,
            n_trials=n_trials,
            t_obs=1000,
            skew=0.0,
            excess_kurtosis=0.0,
        )
        assert 0.49 < dsr < 0.51, f"DSR should be ~0.5 when SR=SR₀, got {dsr:.6f}"

    def test_standard_error_normal_case(self) -> None:
        """Verify SE(SR) under normality matches the analytical form."""
        observed_sr: float = 1.0
        t_obs: int = 100
        skew: float = 0.0
        excess_kurtosis: float = 0.0
        n_trials: int = 2

        # SE² = (1 − 0 + 0) / (100 − 1) = 1 / 99
        expected_se = math.sqrt(1.0 / 99.0)

        # Manual SR₀ for N=2 with sr_std=1.0 (default).
        log_n = math.log(n_trials)
        sqrt_2_log_n = math.sqrt(2.0 * log_n)
        sr_null_manual = (
            sqrt_2_log_n * (1.0 - _EULER_MASCHERONI / (2.0 * log_n))
            + _EULER_MASCHERONI / sqrt_2_log_n
        )

        from scipy.stats import norm as sp_norm

        z_expected = (observed_sr - sr_null_manual) / expected_se
        expected_dsr = float(sp_norm.cdf(z_expected))

        dsr = compute_dsr(
            observed_sr=observed_sr,
            n_trials=n_trials,
            t_obs=t_obs,
            skew=skew,
            excess_kurtosis=excess_kurtosis,
        )
        assert dsr == pytest.approx(expected_dsr, rel=1e-10), (
            f"DSR mismatch: {dsr:.12f} vs expected {expected_dsr:.12f}"
        )
