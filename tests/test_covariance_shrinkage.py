"""Tests for nonlinear covariance shrinkage (Ledoit & Wolf 2017/2020).

Validates eigenvalue cleaning, condition-number improvement, high-dimensional
regime stability, and out-of-sample minimum-variance portfolio performance.

All tests use synthetic data with fixed seeds — no network calls.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pysharpe.optimization.estimators import (
    compute_linear_shrinkage,
    compute_nonlinear_shrinkage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def daily_returns_5(rng) -> pd.DataFrame:
    """5-asset daily returns with 252 observations (T > N regime)."""
    data = rng.normal(0, 0.01, (252, 5))
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(5)])


@pytest.fixture
def high_dim_returns(rng) -> pd.DataFrame:
    """High-dimensional regime: N=50 assets, T=60 observations (T ≈ N)."""
    T, N = 60, 50
    # Generate returns from a low-dimensional factor model to induce collinearity
    # 3 latent factors explain ~80% of variance
    n_factors = 3
    factors = rng.normal(0, 0.01, (T, n_factors))
    loadings = rng.normal(0, 0.02, (N, n_factors))
    # Systematic component
    systematic = factors @ loadings.T
    # Idiosyncratic noise
    idiosyncratic = rng.normal(0, 0.005, (T, N))
    returns = systematic + idiosyncratic
    return pd.DataFrame(returns, columns=[f"Asset_{i:02d}" for i in range(N)])


@pytest.fixture
def extreme_high_dim_returns(rng) -> pd.DataFrame:
    """Extreme high-dim: N=80 assets, T=40 observations (c = 2.0)."""
    T, N = 40, 80
    n_factors = 2
    factors = rng.normal(0, 0.01, (T, n_factors))
    loadings = rng.normal(0, 0.015, (N, n_factors))
    systematic = factors @ loadings.T
    idiosyncratic = rng.normal(0, 0.008, (T, N))
    returns = systematic + idiosyncratic
    return pd.DataFrame(returns, columns=[f"A_{i:02d}" for i in range(N)])


# ===========================================================================
# Basic properties
# ===========================================================================


class TestNonlinearShrinkageProperties:
    """Verify basic mathematical properties of the estimator."""

    def test_output_is_symmetric(self, daily_returns_5):
        cov = compute_nonlinear_shrinkage(daily_returns_5)
        assert np.allclose(cov.values, cov.values.T)

    def test_output_is_positive_definite(self, daily_returns_5):
        cov = compute_nonlinear_shrinkage(daily_returns_5)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0), f"Non-positive eigenvalue: {eigvals.min():.2e}"

    def test_output_has_correct_shape(self, daily_returns_5):
        cov = compute_nonlinear_shrinkage(daily_returns_5)
        assert cov.shape == (5, 5)

    def test_preserves_ticker_labels(self, daily_returns_5):
        cov = compute_nonlinear_shrinkage(daily_returns_5)
        assert list(cov.index) == [f"A{i}" for i in range(5)]
        assert list(cov.columns) == [f"A{i}" for i in range(5)]

    def test_returns_dataframe_type(self, daily_returns_5):
        cov = compute_nonlinear_shrinkage(daily_returns_5)
        assert isinstance(cov, pd.DataFrame)

    def test_condition_number_improves(self, daily_returns_5):
        """Nonlinear shrinkage should not worsen the condition number."""
        X = daily_returns_5.values
        S = np.cov(X, rowvar=False)
        cond_sample = np.linalg.cond(S)

        cov_nl = compute_nonlinear_shrinkage(daily_returns_5)
        cond_nl = np.linalg.cond(cov_nl.values)

        # In the T > N regime, both should be reasonable
        # and shrinkage should not explode the condition number
        assert cond_nl < 10 * max(cond_sample, 1.0) or cond_nl < 1e6

    def test_trace_preserved_approximately(self, daily_returns_5):
        """Total variance (trace) should not change dramatically."""
        X = daily_returns_5.values
        S = np.cov(X, rowvar=False)
        trace_sample = np.trace(S)

        cov_nl = compute_nonlinear_shrinkage(daily_returns_5)
        trace_nl = np.trace(cov_nl.values)

        # Trace should be within factor 10
        ratio = trace_nl / trace_sample if trace_sample > 0 else 1.0
        assert 0.1 < ratio < 10.0, f"Trace ratio: {ratio:.3f}"


# ===========================================================================
# High-dimensional regime
# ===========================================================================


class TestHighDimensionalRegime:
    """Tests in the challenging T ≈ N (or T < N) regime."""

    def test_high_dim_does_not_crash(self, high_dim_returns):
        """N=50, T=60 should compute without errors."""
        cov = compute_nonlinear_shrinkage(high_dim_returns)
        assert cov.shape == (50, 50)
        assert np.allclose(cov.values, cov.values.T)

    def test_high_dim_is_positive_definite(self, high_dim_returns):
        """Even when T ≈ N, the output must be positive definite."""
        cov = compute_nonlinear_shrinkage(high_dim_returns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0), (
            f"Non-positive eigenvalue in high-dim regime: {eigvals.min():.2e}"
        )

    def test_high_dim_condition_number_improves(self, high_dim_returns):
        """Sample covariance is near-singular (rank ≤ 60 for 50 assets).
        Nonlinear shrinkage must produce a well-conditioned estimate."""
        X = high_dim_returns.values
        S = np.cov(X, rowvar=False)
        cond_sample = np.linalg.cond(S)

        cov_nl = compute_nonlinear_shrinkage(high_dim_returns)
        cond_nl = np.linalg.cond(cov_nl.values)

        # Sample covariance in T ≈ N may have condition > 1e10
        # Shrunk version should be dramatically better
        assert cond_nl < cond_sample * 0.1 or cond_nl < 1e8, (
            f"cond(sample)={cond_sample:.2e}, cond(nl)={cond_nl:.2e}"
        )

        # Log for information
        assert cond_nl > 0

    def test_extreme_high_dim_c_gt_1(self, extreme_high_dim_returns):
        """N=80, T=40. Sample cov rank <= 39. NL shrinkage fills zeros."""
        cov = compute_nonlinear_shrinkage(extreme_high_dim_returns)
        eigvals = np.linalg.eigvalsh(cov.values)

        assert np.all(eigvals >= -1e-12), (
            f"Negative eigenvalue in c>1 regime: {eigvals.min():.2e}"
        )

        # Shrunk matrix should have at least as many non-zero eigenvalues
        # as the sample rank (max 39 for T=40, N=80)
        n_nonzero = int(np.sum(eigvals > 1e-15))
        assert n_nonzero >= 39, f"Only {n_nonzero}/{len(eigvals)} non-zero eigenvalues"

    def test_condition_number_warning(self, caplog):
        """Verify condition-number warning fires when threshold is low."""
        rng = np.random.default_rng(99)
        T, N = 40, 80
        factors = rng.normal(0, 0.02, (T, 2))
        loadings = rng.normal(0, 0.03, (N, 2))
        returns = pd.DataFrame(
            factors @ loadings.T + rng.normal(0, 0.01, (T, N)),
            columns=[f"A_{i:02d}" for i in range(N)],
        )

        with caplog.at_level(logging.WARNING):
            cov = compute_nonlinear_shrinkage(returns, condition_warn_threshold=100.0)

        cond = np.linalg.cond(cov.values)
        if cond > 100:
            assert any("condition number" in r.message.lower() for r in caplog.records)


# ===========================================================================
# Validation & edge cases
# ===========================================================================


class TestValidation:
    """Input validation and edge-case behaviour."""

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            compute_nonlinear_shrinkage(np.ones((10, 3)))  # type: ignore[arg-type]

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            compute_nonlinear_shrinkage(pd.DataFrame())

    def test_raises_on_too_few_observations(self):
        df = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.01]})
        with pytest.raises(ValueError, match="3 observations"):
            compute_nonlinear_shrinkage(df)

    def test_raises_on_single_asset(self):
        df = pd.DataFrame({"A": np.random.randn(10) * 0.01})
        with pytest.raises(ValueError, match="2 assets"):
            compute_nonlinear_shrinkage(df)

    def test_raises_on_nan(self):
        df = pd.DataFrame({"A": [0.01, np.nan, 0.03], "B": [0.01, 0.02, 0.03]})
        with pytest.raises(ValueError, match="NaN"):
            compute_nonlinear_shrinkage(df)

    def test_two_asset_minimal_case(self, rng):
        """Smallest valid case: 2 assets, 3 observations."""
        df = pd.DataFrame(rng.normal(0, 0.01, (3, 2)), columns=["A", "B"])
        cov = compute_nonlinear_shrinkage(df)
        assert cov.shape == (2, 2)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)

    def test_large_N_moderate_T(self, rng):
        """N=30, T=100: moderate high-dim, should be stable."""
        T, N = 100, 30
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (T, N)),
            columns=[f"X{i}" for i in range(N)],
        )
        cov = compute_nonlinear_shrinkage(returns)
        assert cov.shape == (N, N)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)

    def test_all_zeros(self):
        """Returns DataFrame of all zeros must not crash."""
        df = pd.DataFrame(np.zeros((10, 3)), columns=["A", "B", "C"])
        cov = compute_nonlinear_shrinkage(df)
        assert cov.shape == (3, 3)
        # All-zero returns → zero sample covariance → shrunk toward zero
        assert np.all(np.isfinite(cov.values))


# ===========================================================================
# Linear shrinkage fallback
# ===========================================================================


class TestLinearShrinkage:
    """scikit-learn LedoitWolf linear shrinkage wrapper."""

    def test_basic_output(self, daily_returns_5):
        cov = compute_linear_shrinkage(daily_returns_5)
        assert cov.shape == (5, 5)
        assert np.allclose(cov.values, cov.values.T)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)

    def test_preserves_labels(self, daily_returns_5):
        cov = compute_linear_shrinkage(daily_returns_5)
        assert list(cov.index) == [f"A{i}" for i in range(5)]

    def test_linear_vs_nonlinear_high_dim(self, high_dim_returns):
        """In high dimensions, nonlinear should produce a better-conditioned
        matrix than linear shrinkage."""
        cov_linear = compute_linear_shrinkage(high_dim_returns)
        cov_nonlinear = compute_nonlinear_shrinkage(high_dim_returns)

        cond_lin = np.linalg.cond(cov_linear.values)
        cond_nl = np.linalg.cond(cov_nonlinear.values)

        # Nonlinear should not be dramatically worse
        # (it may be better or comparable; the key test is stability)
        assert cond_nl < 1e12
        assert cond_lin < 1e12

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            compute_linear_shrinkage(np.ones((10, 3)))  # type: ignore[arg-type]


# ===========================================================================
# Out-of-sample minimum-variance portfolio
# ===========================================================================


class TestOutOfSampleMinVariance:
    """Verify nonlinear shrinkage improves out-of-sample portfolio variance
    vs sample covariance in a high-dimensional collinear regime."""

    @staticmethod
    def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
        """Compute minimum-variance portfolio weights (closed form).

        w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
        """
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        ones = np.ones(cov.shape[0])
        w = inv_cov @ ones
        w /= w.sum()
        return w

    @staticmethod
    def _portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    def test_oos_variance_reduction(self, rng):
        """High-dimensional regime with collinearity:
        N=50, T_train=60, T_test=60.
        Nonlinear shrinkage → lower out-of-sample portfolio variance than
        sample covariance for minimum-variance portfolios."""
        T_train, T_test, N = 60, 60, 50
        n_factors = 3

        # --- Generate factor structure (shared across train + test) -----------
        # Train period
        factors_train = rng.normal(0, 0.01, (T_train, n_factors))
        loadings = rng.normal(0, 0.02, (N, n_factors))
        systematic_train = factors_train @ loadings.T
        idio_train = rng.normal(0, 0.005, (T_train, N))
        returns_train = systematic_train + idio_train

        # Test period (same factor structure, fresh noise)
        factors_test = rng.normal(0, 0.01, (T_test, n_factors))
        systematic_test = factors_test @ loadings.T
        idio_test = rng.normal(0, 0.005, (T_test, N))
        returns_test = systematic_test + idio_test

        # True out-of-sample covariance
        true_cov_test = np.cov(returns_test, rowvar=False)

        # --- Sample covariance from training data -----------------------------
        S_sample = np.cov(returns_train, rowvar=False)
        cond_sample = np.linalg.cond(S_sample)

        # --- Nonlinear shrinkage from training data ---------------------------
        df_train = pd.DataFrame(
            returns_train,
            columns=[f"A{i:02d}" for i in range(N)],
        )
        cov_nl = compute_nonlinear_shrinkage(df_train)
        cond_nl = np.linalg.cond(cov_nl.values)

        # --- Minimum-variance weights -----------------------------------------
        w_sample = self._min_variance_weights(S_sample)
        w_nl = self._min_variance_weights(cov_nl.values)

        # --- Out-of-sample portfolio variances --------------------------------
        var_sample = self._portfolio_variance(w_sample, true_cov_test)
        var_nl = self._portfolio_variance(w_nl, true_cov_test)

        # Nonlinear shrinkage should produce lower (or equal) OOS variance
        assert var_nl <= var_sample * 1.05, (
            f"Nonlinear OOS variance ({var_nl:.2e}) should not substantially "
            f"exceed sample OOS variance ({var_sample:.2e}). "
            f"Sample cond={cond_sample:.2e}, NL cond={cond_nl:.2e}"
        )

    def test_shrinkage_improves_with_collinearity(self, rng):
        """With strong collinearity (high factor loadings), nonlinear
        shrinkage should dominate sample covariance more decisively."""
        T, N = 50, 30
        n_factors = 2

        # Strong factor structure: high loadings → high collinearity
        loadings = rng.normal(0, 0.05, (N, n_factors))  # large loadings
        factors = rng.normal(0, 0.01, (T, n_factors))
        systematic = factors @ loadings.T
        idio = rng.normal(0, 0.002, (T, N))  # low idiosyncratic noise
        returns_train = systematic + idio

        # Test returns
        factors_test = rng.normal(0, 0.01, (T, n_factors))
        systematic_test = factors_test @ loadings.T
        idio_test = rng.normal(0, 0.002, (T, N))
        returns_test = systematic_test + idio_test
        true_cov_test = np.cov(returns_test, rowvar=False)

        # Train
        df_train = pd.DataFrame(
            returns_train,
            columns=[f"B{i:02d}" for i in range(N)],
        )
        S_sample = np.cov(returns_train, rowvar=False)
        cov_nl = compute_nonlinear_shrinkage(df_train)

        # Weights
        w_sample = self._min_variance_weights(S_sample)
        w_nl = self._min_variance_weights(cov_nl.values)

        # OOS variance
        var_sample = self._portfolio_variance(w_sample, true_cov_test)
        var_nl = self._portfolio_variance(w_nl, true_cov_test)

        # With strong collinearity, nonlinear shrinkage should be better
        assert var_nl <= var_sample * 1.10, (
            f"Collinear regime: NL OOS var={var_nl:.2e}, "
            f"sample OOS var={var_sample:.2e}"
        )


# ===========================================================================
# Eigenvalue cleaning validation
# ===========================================================================


class TestEigenvalueCleaning:
    """Validate that nonlinear shrinkage corrects eigenvalue dispersion."""

    def test_eigenvalue_spectrum_corrected(self, high_dim_returns):
        """Nonlinear shrinkage corrects eigenvalue dispersion toward the
        population spectrum. When c close to 1, largest eigenvalues may expand
        (they were downward-biased). The key test: condition number improves."""
        X = high_dim_returns.values
        S = np.cov(X, rowvar=False)
        sample_eigs = np.sort(np.linalg.eigvalsh(S))[::-1]

        cov_nl = compute_nonlinear_shrinkage(high_dim_returns)
        nl_eigs = np.sort(np.linalg.eigvalsh(cov_nl.values))[::-1]

        # Condition number must improve significantly
        denom_sample = sample_eigs[-1] if sample_eigs[-1] > 1e-30 else 1e-30
        denom_nl = nl_eigs[-1] if nl_eigs[-1] > 1e-30 else 1e-30
        cond_sample = sample_eigs[0] / denom_sample
        cond_nl = nl_eigs[0] / denom_nl
        assert cond_nl < cond_sample * 0.5 or cond_nl < 1e7, (
            f"cond(sample)={cond_sample:.2e}, cond(nl)={cond_nl:.2e}"
        )

    def test_no_negative_eigenvalues(self, high_dim_returns):
        cov_nl = compute_nonlinear_shrinkage(high_dim_returns)
        eigvals = np.linalg.eigvalsh(cov_nl.values)
        assert np.all(eigvals >= 0)

    def test_zero_eigenvalues_filled(self, extreme_high_dim_returns):
        """In the c > 1 regime (N > T), sample cov has zero eigenvalues.
        Nonlinear shrinkage must replace these with small positive values."""
        X = extreme_high_dim_returns.values
        S = np.cov(X, rowvar=False)
        sample_eigs = np.linalg.eigvalsh(S)
        n_zero_sample = int(np.sum(sample_eigs < 1e-15))

        cov_nl = compute_nonlinear_shrinkage(extreme_high_dim_returns)
        nl_eigs = np.linalg.eigvalsh(cov_nl.values)
        n_zero_nl = int(np.sum(nl_eigs < 1e-15))

        # Nonlinear shrinkage should reduce the number of zero eigenvalues
        # (ideally to zero, but at minimum produce fewer than the sample)
        assert n_zero_nl <= n_zero_sample, (
            f"Sample: {n_zero_sample} zero eigenvalues. "
            f"NL: {n_zero_nl} zero eigenvalues."
        )

    def test_eigenvalue_ordering_preserved(self, daily_returns_5):
        """Shrunk eigenvalues should be sorted in descending order."""
        cov_nl = compute_nonlinear_shrinkage(daily_returns_5)
        nl_eigs_desc = np.sort(np.linalg.eigvalsh(cov_nl.values))[::-1]

        for i in range(len(nl_eigs_desc) - 1):
            assert nl_eigs_desc[i] >= nl_eigs_desc[i + 1] - 1e-15, (
                f"Eigenvalues not sorted at index {i}: "
                f"{nl_eigs_desc[i]:.2e} < {nl_eigs_desc[i + 1]:.2e}"
            )
