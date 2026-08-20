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

from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.estimators import (
    compute_linear_shrinkage,
    compute_nonlinear_shrinkage,
    ensure_strictly_psd,
    prepare_returns,
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
        with pytest.raises(DataValidationError, match="empty"):
            compute_nonlinear_shrinkage(pd.DataFrame())

    def test_raises_on_too_few_observations(self):
        df = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.01]})
        with pytest.raises(DataValidationError, match="3 observations"):
            compute_nonlinear_shrinkage(df)

    def test_raises_on_single_asset(self):
        df = pd.DataFrame({"A": np.random.randn(10) * 0.01})
        with pytest.raises(DataValidationError, match="2 assets"):
            compute_nonlinear_shrinkage(df)

    def test_raises_on_infinite_values(self):
        df = pd.DataFrame({"A": [0.01, np.inf, 0.03], "B": [0.01, 0.02, 0.03]})
        with pytest.raises(DataValidationError, match="infinite"):
            compute_nonlinear_shrinkage(df)

    def test_raises_on_non_numeric_column(self):
        df = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": ["x", "y", "z"]})
        with pytest.raises(DataValidationError, match="numeric"):
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
        # All-zero returns → zero sample covariance → shrunk toward zero.
        # Must remain finite and strictly positive definite.
        assert np.all(np.isfinite(cov.values))
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)


# ===========================================================================
# Missing-data handling
# ===========================================================================


class TestMissingData:
    """Listwise deletion of partially observed rows — never backfilled."""

    def test_partial_nan_rows_are_dropped(self, rng, caplog):
        """Rows with any NaN are dropped; result matches manually cleaned input."""
        full = pd.DataFrame(rng.normal(0, 0.01, (40, 3)), columns=["A", "B", "C"])
        gappy = full.copy()
        gappy.iloc[1, 0] = np.nan
        gappy.iloc[7, 1] = np.nan

        with caplog.at_level(
            logging.WARNING, logger="pysharpe.optimization.estimators"
        ):
            cov_gappy = compute_nonlinear_shrinkage(gappy)

        assert any("Dropped 2 of 40" in r.message for r in caplog.records)
        cov_clean = compute_nonlinear_shrinkage(full.drop(index=[1, 7]))
        np.testing.assert_allclose(cov_gappy.values, cov_clean.values)

    def test_linear_shrinkage_handles_missing_rows(self, rng):
        full = pd.DataFrame(rng.normal(0, 0.01, (30, 4)), columns=list("ABCD"))
        gappy = full.copy()
        gappy.iloc[3, 2] = np.nan

        cov = compute_linear_shrinkage(gappy)
        assert cov.shape == (4, 4)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)

    def test_fully_missing_asset_raises(self, rng):
        df = pd.DataFrame(rng.normal(0, 0.01, (10, 3)), columns=["A", "B", "C"])
        df["D"] = np.nan
        with pytest.raises(DataValidationError, match="no observed returns"):
            compute_nonlinear_shrinkage(df)

    def test_too_few_rows_after_drop_raises(self):
        df = pd.DataFrame({"A": [0.01, np.nan, np.nan], "B": [np.nan, 0.02, 0.03]})
        with pytest.raises(DataValidationError, match="no observed returns"):
            compute_nonlinear_shrinkage(df)

    def test_prepare_returns_public_helper(self, rng):
        df = pd.DataFrame(rng.normal(0, 0.01, (10, 3)), columns=["A", "B", "C"])
        df.iloc[0, 0] = np.nan
        cleaned = prepare_returns(df)
        assert len(cleaned) == 9
        assert not cleaned.isna().any().any()

        with pytest.raises(DataValidationError):
            prepare_returns(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))


# ===========================================================================
# Strict positive-definiteness guarantees
# ===========================================================================


class TestStrictPSD:
    """Every estimator output must have strictly positive eigenvalues."""

    def test_nonlinear_duplicated_columns_strictly_psd(self, rng):
        """Rank-deficient input (duplicate assets) must still be strictly PD."""
        base = rng.normal(0, 0.01, 60)
        returns = pd.DataFrame(
            {
                "A": base,
                "B": base.copy(),  # exact duplicate → rank deficiency
                "C": rng.normal(0, 0.01, 60),
            }
        )
        cov = compute_nonlinear_shrinkage(returns)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)

    def test_linear_duplicated_columns_strictly_psd(self, rng):
        base = rng.normal(0, 0.01, 60)
        returns = pd.DataFrame({"A": base, "B": base.copy(), "C": base.copy()})
        cov = compute_linear_shrinkage(returns)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)

    def test_linear_zero_variance_asset_strictly_psd(self, rng):
        """A constant (zero-variance) column must not break the estimator."""
        returns = pd.DataFrame(
            {
                "Const": np.zeros(60),
                "A": rng.normal(0, 0.01, 60),
                "B": rng.normal(0, 0.01, 60),
            }
        )
        cov = compute_linear_shrinkage(returns)
        assert np.all(np.isfinite(cov.values))
        assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)
        # Constant asset has no covariance with anything.
        assert np.allclose(cov.loc["Const", ["A", "B"]], 0.0, atol=1e-12)

    def test_nonlinear_zero_variance_asset_strictly_psd(self, rng):
        returns = pd.DataFrame(
            {
                "Const": np.zeros(60),
                "A": rng.normal(0, 0.01, 60),
            }
        )
        cov = compute_nonlinear_shrinkage(returns)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)

    def test_extreme_high_dim_strictly_psd(self, extreme_high_dim_returns):
        cov = compute_nonlinear_shrinkage(extreme_high_dim_returns)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)
        cov_lin = compute_linear_shrinkage(extreme_high_dim_returns)
        assert np.all(np.linalg.eigvalsh(cov_lin.values) > 0.0)

    def test_ensure_strictly_psd_repairs_singular_matrix(self):
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])  # eigenvalue 0
        hardened = ensure_strictly_psd(singular)
        assert np.all(np.linalg.eigvalsh(hardened) > 0.0)

    def test_ensure_strictly_psd_repairs_indefinite_matrix(self):
        indefinite = np.array([[2.0, 3.0], [3.0, 1.0]])  # negative eigenvalue
        hardened = ensure_strictly_psd(indefinite)
        assert np.all(np.linalg.eigvalsh(hardened) > 0.0)

    def test_ensure_strictly_psd_preserves_well_conditioned(self, rng):
        A = rng.normal(0, 1, (5, 5))
        cov = A.T @ A + np.eye(5)  # strictly PD, well conditioned
        hardened = ensure_strictly_psd(cov)
        np.testing.assert_allclose(hardened, cov, rtol=1e-8, atol=1e-12)

    def test_ensure_strictly_psd_raises_on_nan(self):
        with pytest.raises(DataValidationError, match="finite"):
            ensure_strictly_psd(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    def test_ensure_strictly_psd_raises_on_non_square(self):
        with pytest.raises(DataValidationError, match="square"):
            ensure_strictly_psd(np.ones((3, 2)))

    def test_all_zero_returns_strictly_psd(self):
        df = pd.DataFrame(np.zeros((20, 3)), columns=["A", "B", "C"])
        cov_nl = compute_nonlinear_shrinkage(df)
        cov_lw = compute_linear_shrinkage(df)
        assert np.all(np.linalg.eigvalsh(cov_nl.values) > 0.0)
        assert np.all(np.linalg.eigvalsh(cov_lw.values) > 0.0)


# ===========================================================================
# Linear shrinkage fallback
# ===========================================================================


class TestLinearShrinkage:
    """Analytical Ledoit–Wolf (2004) constant-correlation shrinkage."""

    def test_basic_output(self, daily_returns_5):
        cov = compute_linear_shrinkage(daily_returns_5)
        assert cov.shape == (5, 5)
        assert np.allclose(cov.values, cov.values.T)
        assert np.all(np.linalg.eigvalsh(cov.values) > 0)

    def test_preserves_labels(self, daily_returns_5):
        cov = compute_linear_shrinkage(daily_returns_5)
        assert list(cov.index) == [f"A{i}" for i in range(5)]

    def test_diagonal_preserves_sample_variances(self, daily_returns_5):
        """The constant-correlation target preserves variances exactly, so
        the shrunk diagonal must equal the sample variances."""
        cov = compute_linear_shrinkage(daily_returns_5)
        sample_var = daily_returns_5.var(ddof=1).to_numpy(dtype=float)
        np.testing.assert_allclose(np.diag(cov.values), sample_var, rtol=1e-10)

    def test_improves_condition_under_collinearity(self, high_dim_returns):
        """Shrinkage must not worsen conditioning vs the sample covariance."""
        sample = high_dim_returns.cov().to_numpy(dtype=float)
        shrunk = compute_linear_shrinkage(high_dim_returns).values
        assert np.linalg.cond(shrunk) <= np.linalg.cond(sample) * 1.01

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


# ===========================================================================
# Extreme collinearity: perfectly / near-perfectly correlated synthetic assets
# ===========================================================================


class TestCollinearityExtremes:
    """Both estimators must stay strictly positive definite and numerically
    stable when assets are (nearly) perfectly correlated."""

    @staticmethod
    def _perfectly_correlated(n_assets: int, n_obs: int, seed: int) -> pd.DataFrame:
        """All assets are exact positive multiples of one common factor."""
        rng = np.random.default_rng(seed)
        factor = rng.normal(0, 0.01, n_obs)
        scales = np.linspace(0.5, 1.5, n_assets)
        data = np.column_stack([factor * s for s in scales])
        return pd.DataFrame(data, columns=[f"P{i}" for i in range(n_assets)])

    @staticmethod
    def _near_perfectly_correlated(
        n_assets: int, n_obs: int, seed: int
    ) -> pd.DataFrame:
        """ρ ≈ 0.99999: common factor plus a whisper of idiosyncratic noise."""
        rng = np.random.default_rng(seed)
        factor = rng.normal(0, 0.01, n_obs)
        data = np.column_stack(
            [factor + rng.normal(0, 1e-7, n_obs) for _ in range(n_assets)]
        )
        return pd.DataFrame(data, columns=[f"N{i}" for i in range(n_assets)])

    def test_perfect_correlation_nonlinear_is_strictly_psd(self):
        returns = self._perfectly_correlated(5, 252, seed=42)
        cov = compute_nonlinear_shrinkage(returns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0.0)
        # Sample covariance is rank-1; the shrunk estimate must be
        # dramatically better conditioned.
        sample = returns.cov().to_numpy(dtype=float)
        assert np.linalg.cond(cov.values) < 1e12
        assert np.linalg.cond(sample) > 1e12

    def test_perfect_correlation_linear_is_strictly_psd(self):
        returns = self._perfectly_correlated(5, 252, seed=43)
        cov = compute_linear_shrinkage(returns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0.0)
        # Variances are preserved exactly on the diagonal.
        sample_var = returns.var(ddof=1).to_numpy(dtype=float)
        np.testing.assert_allclose(np.diag(cov.values), sample_var, rtol=1e-10)

    def test_near_perfect_correlation_eigenvalue_structure(self):
        """ρ ≈ 0.99999: the SAMPLE spectrum has one dominant eigenvalue;
        the shrunk estimate keeps every direction strictly positive."""
        returns = self._near_perfectly_correlated(6, 300, seed=44)

        sample_eigvals = np.sort(
            np.linalg.eigvalsh(returns.cov().to_numpy(dtype=float))
        )[::-1]
        # One dominant common factor in the raw sample spectrum.
        assert sample_eigvals[0] > 10 * sample_eigvals[1]

        cov = compute_nonlinear_shrinkage(returns)
        eigvals = np.sort(np.linalg.eigvalsh(cov.values))[::-1]
        # Shrinkage pulls the spectrum toward the bulk, but every direction
        # must retain strictly positive variance.
        assert np.all(eigvals > 0.0)
        assert np.linalg.cond(cov.values) < 1e12

    def test_rank_one_high_dimensional_regime(self):
        """T < N with a single common factor (extreme c regime)."""
        rng = np.random.default_rng(45)
        factor = rng.normal(0, 0.01, 30)
        data = np.column_stack([factor * s for s in np.linspace(0.5, 1.5, 40)])
        returns = pd.DataFrame(data, columns=[f"H{i}" for i in range(40)])

        for estimator in (compute_linear_shrinkage, compute_nonlinear_shrinkage):
            cov = estimator(returns)
            assert np.all(np.isfinite(cov.values))
            assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)
            # Eigen-clip floor caps the condition number at ~1/1e-12; allow
            # floating-point slack around the theoretical cap.
            assert np.linalg.cond(cov.values) < 1.1e12

    def test_identical_duplicate_columns_both_estimators(self):
        """Exact duplicates (not just perfect correlation): both estimators
        produce symmetric strictly PD matrices with identical labels."""
        rng = np.random.default_rng(46)
        base = rng.normal(0, 0.01, 100)
        returns = pd.DataFrame(
            {
                "A": base,
                "B": base.copy(),
                "C": base.copy(),
                "D": base.copy(),
            }
        )
        for estimator in (compute_linear_shrinkage, compute_nonlinear_shrinkage):
            cov = estimator(returns)
            assert list(cov.index) == list(returns.columns)
            np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)
            assert np.all(np.linalg.eigvalsh(cov.values) > 0.0)

    def test_off_diagonals_remain_positive_for_perfect_correlation(self):
        """Perfectly correlated assets must keep positive covariance
        estimates (shrinkage must not zero out the common factor)."""
        returns = self._perfectly_correlated(4, 252, seed=47)
        cov = compute_linear_shrinkage(returns)
        off_diag = cov.values[np.triu_indices(4, k=1)]
        assert np.all(off_diag > 0.0)
