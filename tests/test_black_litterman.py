"""Tests for the Black-Litterman reverse optimisation module.

All tests use synthetic data with fixed seeds — no network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.black_litterman import (
    blend_views,
    build_views_uncertainty,
    compute_implied_returns,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def cov_4asset(rng) -> np.ndarray:
    """A realistic 4-asset annualised covariance matrix."""
    daily_cov = np.cov(rng.normal(0, 0.01, (500, 4)).T)
    return daily_cov * 252  # annualise


@pytest.fixture
def market_weights_4asset() -> np.ndarray:
    """Equal-weighted market-cap weights for 4 assets."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def cov_3asset_df(rng) -> pd.DataFrame:
    """3-asset covariance as a labelled DataFrame."""
    daily_cov = np.cov(rng.normal(0, 0.01, (300, 3)).T)
    ann_cov = daily_cov * 252
    return pd.DataFrame(
        ann_cov,
        index=["VFV", "VDY", "QQC"],
        columns=["VFV", "VDY", "QQC"],
    )


# ---------------------------------------------------------------------------
# compute_implied_returns
# ---------------------------------------------------------------------------


class TestComputeImpliedReturns:
    def test_basic_computation(self, cov_4asset, market_weights_4asset):
        """Π = δ · Σ · w_mkt."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        expected = 2.5 * cov_4asset @ market_weights_4asset
        np.testing.assert_array_almost_equal(pi, expected)

    def test_output_shape(self, cov_4asset, market_weights_4asset):
        """Output is a 1-D array of length n."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        assert isinstance(pi, np.ndarray)
        assert pi.shape == (4,)

    def test_higher_risk_aversion_increases_magnitude(self, cov_4asset):
        """Larger δ produces proportionally larger implied returns."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        pi_low = compute_implied_returns(cov_4asset, w, risk_aversion=1.0)
        pi_high = compute_implied_returns(cov_4asset, w, risk_aversion=5.0)
        np.testing.assert_array_almost_equal(pi_high, 5.0 * pi_low)

    def test_pandas_series_preserved(self, cov_3asset_df):
        """When market_weights is a Series, output is a Series with index."""
        w = pd.Series([0.4, 0.35, 0.25], index=["VFV", "VDY", "QQC"])
        pi = compute_implied_returns(cov_3asset_df, w)
        assert isinstance(pi, pd.Series)
        assert list(pi.index) == ["VFV", "VDY", "QQC"]
        assert pi.name == "Implied Returns"

    def test_numpy_inputs_return_ndarray(self, cov_3asset_df):
        """NumPy array inputs produce NumPy array output."""
        w = np.array([0.4, 0.35, 0.25])
        pi = compute_implied_returns(cov_3asset_df.values, w)
        assert isinstance(pi, np.ndarray)

    def test_raises_on_dimension_mismatch(self, cov_4asset):
        """A 4-asset Σ must be paired with 4 weights."""
        with pytest.raises(ValueError, match="matching lengths"):
            compute_implied_returns(cov_4asset, np.array([0.5, 0.5]))

    def test_raises_on_non_square_cov(self):
        """cov_matrix must be square."""
        with pytest.raises(ValueError, match="must be square"):
            compute_implied_returns(np.ones((3, 4)), np.ones(3))

    def test_raises_on_nan(self, cov_4asset):
        """NaN in inputs must raise."""
        bad_cov = cov_4asset.copy()
        bad_cov[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            compute_implied_returns(bad_cov, np.ones(4))


# ---------------------------------------------------------------------------
# build_views_uncertainty
# ---------------------------------------------------------------------------


class TestBuildViewsUncertainty:
    def test_basic_construction(self, cov_4asset):
        """Build Ω from P and confidences."""
        P = np.array([[1, -1, 0, 0]])
        Omega = build_views_uncertainty(cov_4asset, P, [50.0])
        assert Omega.shape == (1, 1)
        assert Omega[0, 0] > 0

    def test_full_confidence_yields_small_omega(self, cov_4asset):
        """100 % confidence → ω ≈ 0."""
        P = np.array([[1, 0, -1, 0]])
        Omega_full = build_views_uncertainty(cov_4asset, P, [100.0])
        Omega_high = build_views_uncertainty(cov_4asset, P, [99.999])
        assert Omega_full[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert Omega_high[0, 0] > 0  # Not quite 100 %

    def test_low_confidence_yields_large_omega(self, cov_4asset):
        """Low confidence → large ω."""
        P = np.array([[1, 0, 0, -1]])
        Omega_low = build_views_uncertainty(cov_4asset, P, [1.0])
        Omega_high = build_views_uncertainty(cov_4asset, P, [50.0])
        assert Omega_low[0, 0] > Omega_high[0, 0]

    def test_multiple_views(self, cov_4asset):
        """Ω is K×K diagonal for K views."""
        P = np.array(
            [
                [1, -1, 0, 0],
                [0, 0, 1, -1],
            ]
        )
        Omega = build_views_uncertainty(cov_4asset, P, [70.0, 30.0])
        assert Omega.shape == (2, 2)
        assert Omega[0, 0] != Omega[1, 1]
        # Off-diagonal entries must be zero for diagonal uncertainty
        assert Omega[0, 1] == 0.0
        assert Omega[1, 0] == 0.0

    def test_raises_on_invalid_confidence(self, cov_4asset):
        """Confidence must be in (0, 100]."""
        P = np.array([[1, 0, 0, 0]])
        with pytest.raises(ValueError, match="view_confidences"):
            build_views_uncertainty(cov_4asset, P, [0.0])
        with pytest.raises(ValueError, match="view_confidences"):
            build_views_uncertainty(cov_4asset, P, [100.1])
        with pytest.raises(ValueError, match="view_confidences"):
            build_views_uncertainty(cov_4asset, P, [-5.0])

    def test_raises_on_shape_mismatch(self, cov_4asset):
        """Number of confidences must equal number of P rows."""
        P = np.array([[1, -1, 0, 0], [0, 1, -1, 0]])
        with pytest.raises(ValueError, match="match"):
            build_views_uncertainty(cov_4asset, P, [70.0])  # 2 views, 1 confidence

    def test_raises_on_P_column_mismatch(self, cov_4asset):
        """P columns must match cov_matrix dimension."""
        P_wrong = np.array([[1, -1, 0]])  # 3 cols for 4-asset cov
        with pytest.raises(ValueError, match="has 3 columns"):
            build_views_uncertainty(cov_4asset, P_wrong, [70.0])

    def test_tau_scales_omega(self, cov_4asset):
        """Larger τ → proportionally larger ω entries."""
        P = np.array([[1, 0, -1, 0]])
        Omega_small = build_views_uncertainty(cov_4asset, P, [50.0], tau=0.01)
        Omega_large = build_views_uncertainty(cov_4asset, P, [50.0], tau=0.10)
        np.testing.assert_array_almost_equal(Omega_large, 10.0 * Omega_small)


# ---------------------------------------------------------------------------
# blend_views
# ---------------------------------------------------------------------------


class TestBlendViews:
    def test_no_views_recovers_implied(self, cov_4asset, market_weights_4asset):
        """When K = 0 (no views), posterior returns converge exactly to Π.

        With no views, M = (τΣ)⁻¹, so M⁻¹ = τΣ.  The expected returns
        recover Π exactly, and the posterior covariance is (1+τ)Σ — the
        extra τΣ accounts for prior-estimation uncertainty.
        """
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        # Empty views: K = 0
        P = np.empty((0, 4))
        Q = np.empty(0)
        Omega = np.empty((0, 0))

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)

        # Returns converge to Π
        np.testing.assert_array_almost_equal(er, pi)

        # Σ_p = Σ + τΣ = (1+τ)Σ (default τ = 0.05)
        expected_cov = (1 + 0.05) * cov_4asset
        np.testing.assert_array_almost_equal(cov_p, expected_cov)

    def test_infinite_uncertainty_converges_to_implied(
        self, cov_4asset, market_weights_4asset
    ):
        """As Ω_k → ∞, the k-th view has no effect and E(R) → Π."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.05])  # 5 % relative outperformance view

        # Extremely large uncertainty → view is ignored
        Omega_weak = np.diag([1e12])
        er_weak, _ = blend_views(pi, cov_4asset, P, Q, Omega_weak)
        np.testing.assert_array_almost_equal(er_weak, pi, decimal=8)

    def test_extreme_confidence_dominates(self, cov_4asset, market_weights_4asset):
        """When Ω ≈ 0 (total confidence), posterior reflects the view exactly."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        # Single-asset absolute view: asset 0 will return 10 %
        P = np.array([[1, 0, 0, 0]])
        Q = np.array([0.10])
        Omega = np.diag([1e-12])  # near-zero uncertainty

        er, _ = blend_views(pi, cov_4asset, P, Q, Omega)
        assert er[0] == pytest.approx(0.10, rel=0.01)

    def test_single_view_shifts_posterior(self, cov_4asset, market_weights_4asset):
        """A single view with moderate confidence shifts posterior sensibly."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        # View: asset 0 will outperform asset 1 by 3 %
        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.03])
        Omega = build_views_uncertainty(cov_4asset, P, [50.0])

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)

        # Posterior return spread should be between Π spread and Π spread + Q
        implied_spread = pi[0] - pi[1]
        posterior_spread = er[0] - er[1]
        assert posterior_spread > implied_spread
        assert posterior_spread < implied_spread + Q[0]

        # Σ_p − Σ = M⁻¹ must be positive semi-definite
        diff = cov_p - cov_4asset
        eigvals = np.linalg.eigvalsh(diff)
        assert np.all(eigvals >= -1e-12), (
            f"Σ_p − Σ is not PSD: min eigval = {eigvals.min():.2e}"
        )

        # Trace (total variance) must increase
        assert np.trace(cov_p) >= np.trace(cov_4asset) - 1e-12

    def test_multiple_views(self, cov_4asset, market_weights_4asset):
        """Two independent views are blended correctly."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        P = np.array(
            [
                [1, -1, 0, 0],  # asset 0 > asset 1
                [0, 0, 1, -1],  # asset 2 > asset 3
            ]
        )
        Q = np.array([0.03, 0.02])
        Omega = build_views_uncertainty(cov_4asset, P, [60.0, 80.0])

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)

        assert er.shape == (4,)
        assert cov_p.shape == (4, 4)
        # Both views shift spreads in the expected direction
        assert er[0] - er[1] > pi[0] - pi[1]
        assert er[2] - er[3] > pi[2] - pi[3]

    def test_pandas_labels_preserved(self, cov_3asset_df):
        """Pandas Series/DataFrame inputs → Pandas outputs with labels."""
        pi = pd.Series(
            [0.04, 0.06, 0.07],
            index=["VFV", "VDY", "QQC"],
        )

        P = np.array([[1, -1, 0]])
        Q = np.array([0.02])
        Omega = np.diag([0.0001])

        er, cov_p = blend_views(pi, cov_3asset_df, P, Q, Omega)

        assert isinstance(er, pd.Series)
        assert list(er.index) == ["VFV", "VDY", "QQC"]
        assert er.name == "BL Posterior Returns"
        assert isinstance(cov_p, pd.DataFrame)
        assert list(cov_p.index) == ["VFV", "VDY", "QQC"]
        assert list(cov_p.columns) == ["VFV", "VDY", "QQC"]

    def test_raises_on_dimension_mismatch(self, cov_4asset, market_weights_4asset):
        """Invalid shapes must raise ValueError."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.03])
        Omega = np.diag([0.001])

        # Mismatched implied_returns length
        with pytest.raises(ValueError, match="implied_returns"):
            blend_views(np.array([0.01, 0.02]), cov_4asset, P, Q, Omega)

        # Mismatched P columns
        P_bad = np.array([[1, -1, 0]])  # 3 cols
        with pytest.raises(ValueError, match="P must be"):
            blend_views(pi, cov_4asset, P_bad, Q, Omega)

        # Mismatched Q length
        with pytest.raises(ValueError, match="Q length"):
            blend_views(pi, cov_4asset, P, np.array([0.01, 0.02]), Omega)

        # Mismatched Omega shape
        Omega_bad = np.diag([0.001, 0.002])
        with pytest.raises(ValueError, match="Omega shape"):
            blend_views(pi, cov_4asset, P, Q, Omega_bad)

    def test_bayesian_shrinkage_property(self, cov_4asset, market_weights_4asset):
        """BL posterior returns are a weighted average of Π and views — shrinkage."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        P = np.array([[1, 0, 0, 0]])
        Q = np.array([0.20])  # Strong absolute view on asset 0

        # Moderate confidence
        Omega_moderate = build_views_uncertainty(cov_4asset, P, [50.0])
        er_moderate, _ = blend_views(pi, cov_4asset, P, Q, Omega_moderate)

        # er[0] should be between pi[0] and 0.20 (shrinkage toward the view)
        assert abs(er_moderate[0] - pi[0]) < abs(0.20 - pi[0])

        # For assets not in the view, the shift is through covariance
        # (should not change drastically)
        for i in range(1, 4):
            assert abs(er_moderate[i] - pi[i]) < 0.15  # sensible bound

    def test_posterior_covariance_increases(self, cov_4asset, market_weights_4asset):
        """Σ_p = Σ + M⁻¹.  The difference M⁻¹ must be PSD and trace must grow."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.03])
        Omega = build_views_uncertainty(cov_4asset, P, [50.0])

        _, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)

        # Σ_p − Σ = M⁻¹ is PSD (element-wise ≥ not guaranteed for off-diagonals)
        diff = cov_p - cov_4asset
        eigvals = np.linalg.eigvalsh(diff)
        assert np.all(eigvals >= -1e-12), (
            f"Σ_p − Σ is not PSD: min eigval = {eigvals.min():.2e}"
        )

        # Trace must increase — views *add* uncertainty, never reduce it
        assert np.trace(cov_p) >= np.trace(cov_4asset) - 1e-12

    def test_zero_uncertainty_numerical_stability(self, cov_4asset):
        """Ω = 0 (perfect confidence) must not crash."""
        pi = np.array([0.04, 0.05, 0.045, 0.055])
        P = np.array([[1, 0, 0, 0]])
        Q = np.array([0.08])
        Omega = np.zeros((1, 1))  # Zero uncertainty

        # Should not raise — the implementation clamps Ω⁻¹
        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)
        assert er.shape == (4,)
        assert cov_p.shape == (4, 4)
        # With zero uncertainty, er[0] should be ~Q[0]
        assert er[0] == pytest.approx(Q[0], rel=0.05)


# ---------------------------------------------------------------------------
# Integration: end-to-end Black-Litterman → MVO readiness
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_bl_output_feeds_mvo(self, cov_4asset, market_weights_4asset):
        """BL posterior returns and covariance are valid MVO inputs."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        P = np.array(
            [
                [1, -1, 0, 0],
                [0, 0, 1, -1],
            ]
        )
        Q = np.array([0.03, 0.02])
        Omega = build_views_uncertainty(cov_4asset, P, [70.0, 60.0])

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)

        # Covariance matrix must be symmetric positive semi-definite
        assert np.allclose(cov_p, cov_p.T)
        eigvals = np.linalg.eigvalsh(cov_p)
        assert np.all(eigvals >= -1e-12)

        # Returns vector must have the correct length
        assert len(er) == len(market_weights_4asset)

    def test_single_asset_views_no_crash(self, cov_4asset, market_weights_4asset):
        """K = 1 with an absolute view on one asset works end-to-end."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        P = np.array([[0, 1, 0, 0]])
        Q = np.array([0.06])
        Omega = build_views_uncertainty(cov_4asset, P, [65.0])

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)
        assert er.shape == (4,)
        # Posterior mean for asset 1 should shift toward Q
        assert abs(er[1] - Q[0]) < abs(pi[1] - Q[0])

    def test_idzorek_style_calibration(self, cov_4asset):
        """End-to-end: compute Π, calibrate Ω from confidences, blend."""
        w_mkt = np.array([0.40, 0.30, 0.20, 0.10])
        pi = compute_implied_returns(cov_4asset, w_mkt, risk_aversion=3.0)

        # Market-neutral view: asset 0 outruns asset 3 by 4 %
        P = np.array([[1, 0, 0, -1]])
        Q = np.array([0.04])

        # Investor is 75 % confident in this view
        Omega = build_views_uncertainty(cov_4asset, P, [75.0], tau=0.025)

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega, tau=0.025)

        # Posterior should reflect the view
        posterior_spread = er[0] - er[3]
        implied_spread = pi[0] - pi[3]
        assert posterior_spread > implied_spread
        assert posterior_spread < Q[0]  # Not fully to the view (shrinkage)

        # Covariance should be PSD
        assert np.all(np.linalg.eigvalsh(cov_p) >= -1e-12)


# ---------------------------------------------------------------------------
# Edge case mathematical properties
# ---------------------------------------------------------------------------


class TestBlendViewsEdgeCases:
    """Edge cases and mathematical invariants for blend_views."""

    def test_zero_views_returns_prior(self, cov_4asset, market_weights_4asset):
        """With K=0 views, posterior returns equal prior returns.

        Posterior covariance is Σ + τΣ (standard BL: uncertainty is added).
        """
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        P = np.zeros((0, 4))  # No views
        Q = np.array([])
        Omega = np.zeros((0, 0))
        tau = 0.05

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega, tau=tau)
        # Posterior returns = prior (no views to shift)
        np.testing.assert_allclose(er, pi, rtol=1e-9)
        # Posterior covariance = Σ + τΣ (uncertainty added)
        expected_cov = cov_4asset + tau * cov_4asset
        np.testing.assert_allclose(cov_p, expected_cov, rtol=1e-9)

    def test_extreme_confidence_dominates_view(self, cov_4asset, market_weights_4asset):
        """99.99% confidence → posterior return ~= view."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        P = np.array([[1, 0, 0, 0]])
        view_return = 0.10  # 10% expected for asset 0
        Q = np.array([view_return])
        Omega = build_views_uncertainty(cov_4asset, P, [99.99])

        er, _ = blend_views(pi, cov_4asset, P, Q, Omega)
        # Posterior for asset 0 should be very close to the view
        assert abs(er[0] - view_return) < 0.01

    def test_omega_scales_with_tau(self, cov_4asset):
        """Omega ∝ tau — doubling tau doubles Omega entries."""
        P = np.array([[1, -1, 0, 0]])
        conf = [70.0]
        omega_005 = build_views_uncertainty(cov_4asset, P, conf, tau=0.05)
        omega_010 = build_views_uncertainty(cov_4asset, P, conf, tau=0.10)

        np.testing.assert_allclose(omega_010, 2.0 * omega_005, rtol=1e-10)

    def test_confidences_near_zero_produce_large_omega(self):
        """Low confidence → large uncertainty → posterior stays close to prior."""
        rng = np.random.default_rng(42)
        cov = np.cov(rng.normal(0, 0.01, (500, 3)).T) * 252
        w_mkt = np.ones(3) / 3
        pi = compute_implied_returns(cov, w_mkt)

        P = np.array([[1, 0, 0]])
        Q = np.array([0.20])  # Very different from prior

        # Extremely low confidence (close to 0)
        omega_low = build_views_uncertainty(cov, P, [1.0], tau=0.05)
        er_low, _ = blend_views(pi, cov, P, Q, omega_low)

        # High confidence
        omega_high = build_views_uncertainty(cov, P, [90.0], tau=0.05)
        er_high, _ = blend_views(pi, cov, P, Q, omega_high)

        # Low confidence → smaller deviation from prior for the viewed asset
        deviation_low = abs(er_low[0] - pi[0])
        deviation_high = abs(er_high[0] - pi[0])
        assert deviation_low < deviation_high, (
            f"Low confidence should deviate less: {deviation_low:.6f} vs {deviation_high:.6f}"
        )

    def test_multiple_views_converge(self, cov_4asset, market_weights_4asset):
        """Multiple distinct views all contribute to posterior."""
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)

        # Three independent views
        P = np.array(
            [
                [1, 0, 0, 0],  # Asset 0 outperforms by 4%
                [0, 1, -1, 0],  # Asset 1 vs 2 spread
                [0, 0, 0, 1],  # Asset 3 has specific return
            ]
        )
        Q = np.array([0.04, 0.02, 0.03])
        Omega = build_views_uncertainty(cov_4asset, P, [80.0, 60.0, 70.0])

        er, cov_p = blend_views(pi, cov_4asset, P, Q, Omega)

        # Posterior returns should shift toward each view
        assert er[0] > pi[0]  # Asset 0 raised by view 1
        assert er[1] > er[2]  # Asset 1 > Asset 2 from view 2
        # Posterior covariance must be larger than prior (added uncertainty)
        assert np.all(np.diag(cov_p) > np.diag(cov_4asset))

    def test_consistent_with_capm_equilibrium(self):
        """Without views, BL reduces to CAPM equilibrium."""
        rng = np.random.default_rng(99)
        n = 5
        cov = np.cov(rng.normal(0, 0.01, (500, n)).T) * 252
        w_mkt = rng.dirichlet(np.ones(n))
        pi = compute_implied_returns(cov, w_mkt, risk_aversion=3.0)

        # CAPM: market portfolio is tangency → slope = Sharpe
        mkt_return = pi @ w_mkt
        mkt_var = w_mkt @ cov @ w_mkt
        mkt_sharpe = mkt_return / np.sqrt(mkt_var) if mkt_var > 0 else 0

        # The implied Sharpe should be positive (market risk premium)
        assert mkt_sharpe > 0, f"Market Sharpe should be positive, got {mkt_sharpe:.4f}"


class TestImpliedReturnsMath:
    """Mathematical properties of compute_implied_returns."""

    def test_linear_in_risk_aversion(self, cov_4asset, market_weights_4asset):
        """Π ∝ δ — doubling risk aversion doubles implied returns."""
        pi_2 = compute_implied_returns(
            cov_4asset, market_weights_4asset, risk_aversion=2.0
        )
        pi_4 = compute_implied_returns(
            cov_4asset, market_weights_4asset, risk_aversion=4.0
        )
        np.testing.assert_allclose(pi_4, 2.0 * pi_2, rtol=1e-10)

    def test_weights_sum_to_one(self, cov_4asset):
        """Market weights must be a valid simplex."""
        n = 4
        # Valid weights: Dirichlet sample
        rng = np.random.default_rng(55)
        w = rng.dirichlet(np.ones(n))
        pi = compute_implied_returns(cov_4asset, w)
        assert len(pi) == n
        assert not np.any(np.isnan(pi))

    def test_negative_risk_aversion_reverses_sign(self):
        """Negative δ flips the sign of implied returns."""
        rng = np.random.default_rng(42)
        cov = np.cov(rng.normal(0, 0.01, (100, 3)).T) * 252
        w = np.ones(3) / 3
        pi_pos = compute_implied_returns(cov, w, risk_aversion=2.5)
        pi_neg = compute_implied_returns(cov, w, risk_aversion=-2.5)
        np.testing.assert_allclose(pi_neg, -pi_pos, rtol=1e-10)

    def test_uniform_weights_with_identity_cov(self):
        """When Σ = I and weights uniform, Π values are identical."""
        n = 3
        cov = np.eye(n)
        w = np.ones(n) / n
        pi = compute_implied_returns(cov, w, risk_aversion=1.0)
        # All implied returns should be equal: 1.0 * ones/n
        np.testing.assert_allclose(pi, np.ones(n) / n, rtol=1e-10)


# ---------------------------------------------------------------------------
# Hardening: structural validation and strict-PSD guarantees
# ---------------------------------------------------------------------------


class TestHardeningValidation:
    """Malformed matrices must fail loudly with DataValidationError."""

    def test_blend_rejects_non_psd_cov(self, cov_4asset, market_weights_4asset):
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.03])
        Omega = np.diag([0.001])

        indefinite = np.array(
            [
                [2.0, 3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        with pytest.raises(DataValidationError, match="positive semi-definite"):
            blend_views(pi, indefinite, P, Q, Omega)

    def test_implied_returns_reject_non_psd_cov(self, market_weights_4asset):
        indefinite = np.array(
            [
                [2.0, 3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        with pytest.raises(DataValidationError, match="positive semi-definite"):
            compute_implied_returns(indefinite, market_weights_4asset)

    def test_blend_rejects_negative_omega_diagonal(
        self, cov_4asset, market_weights_4asset
    ):
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.03])
        Omega = np.diag([-0.001])  # negative uncertainty is invalid
        with pytest.raises(DataValidationError, match="non-negative"):
            blend_views(pi, cov_4asset, P, Q, Omega)

    def test_blend_rejects_nan_views(self, cov_4asset, market_weights_4asset):
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        P = np.array([[1, -1, 0, 0]])
        Q = np.array([np.nan])
        Omega = np.diag([0.001])
        with pytest.raises(DataValidationError, match="NaN"):
            blend_views(pi, cov_4asset, P, Q, Omega)


class TestHardeningStrictPSD:
    """BL outputs must be strictly positive definite even in degenerate regimes."""

    def test_near_singular_cov_completes_and_is_strictly_psd(
        self, market_weights_4asset
    ):
        """A singular (duplicated-column) covariance must not crash the
        (τΣ)⁻¹ inversion and must produce a strictly PD posterior."""
        rng = np.random.default_rng(7)
        base = rng.normal(0, 0.01, 500)
        returns = np.column_stack(
            [base, base, rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500)]
        )
        singular_cov = np.cov(returns, rowvar=False) * 252  # rank 3

        pi = compute_implied_returns(singular_cov, market_weights_4asset)
        P = np.array([[1, -1, 0, 0]])
        Q = np.array([0.03])
        Omega = build_views_uncertainty(singular_cov, P, [50.0])

        er, cov_p = blend_views(pi, singular_cov, P, Q, Omega)
        assert np.all(np.isfinite(er))
        assert np.all(np.linalg.eigvalsh(cov_p) > 0.0), (
            "Posterior covariance must be strictly positive definite"
        )

    def test_zero_views_posterior_strictly_psd(self, cov_4asset, market_weights_4asset):
        pi = compute_implied_returns(cov_4asset, market_weights_4asset)
        _, cov_p = blend_views(
            pi,
            cov_4asset,
            np.empty((0, 4)),
            np.empty(0),
            np.empty((0, 0)),
        )
        assert np.all(np.linalg.eigvalsh(cov_p) > 0.0)
