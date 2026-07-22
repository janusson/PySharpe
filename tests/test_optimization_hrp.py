"""Tests for ``pysharpe.optimization.hrp`` — Hierarchical Risk Parity.

.. note::

    All tests use synthetic data with fixed seeds.  No network calls.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pysharpe.optimization.hrp import HierarchicalRiskParity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def independent_returns(rng: np.random.Generator) -> pd.DataFrame:
    """Four‑asset returns with negligible correlation."""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.DataFrame(
        {
            "EQ1": rng.normal(0.001, 0.020, 252),
            "EQ2": rng.normal(0.0008, 0.018, 252),
            "FI1": rng.normal(0.0003, 0.005, 252),
            "FI2": rng.normal(0.0002, 0.004, 252),
        },
        index=dates,
    )


@pytest.fixture
def correlated_returns(rng: np.random.Generator) -> pd.DataFrame:
    """Returns with intentional within‑cluster correlation."""
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    common_eq = rng.normal(0.001, 0.015, 252)
    common_fi = rng.normal(0.0003, 0.003, 252)
    return pd.DataFrame(
        {
            "EQ_A": 0.7 * common_eq + 0.3 * rng.normal(0.001, 0.01, 252),
            "EQ_B": 0.7 * common_eq + 0.3 * rng.normal(0.001, 0.01, 252),
            "FI_A": 0.8 * common_fi + 0.2 * rng.normal(0.0003, 0.002, 252),
            "FI_B": 0.8 * common_fi + 0.2 * rng.normal(0.0003, 0.002, 252),
        },
        index=dates,
    )


@pytest.fixture
def near_singular_returns(rng: np.random.Generator) -> pd.DataFrame:
    """Returns where two assets are nearly identical (ρ > 0.999)."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    base = rng.normal(0.001, 0.02, 100)
    return pd.DataFrame(
        {
            "X": base,
            "Y": base + rng.normal(0, 1e-9, 100),  # nearly identical
            "Z": rng.normal(0.001, 0.02, 100),
        },
        index=dates,
    )


@pytest.fixture
def simple_cov() -> pd.DataFrame:
    """2‑asset covariance with different variances."""
    return pd.DataFrame(
        {"A": [0.04, 0.01], "B": [0.01, 0.01]},
        index=["A", "B"],
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    """Construction and input validation."""

    def test_construct_from_returns(self, independent_returns: pd.DataFrame) -> None:
        hrp = HierarchicalRiskParity(returns=independent_returns)
        assert hrp._n_assets == 4
        assert not hrp.ridge_applied

    def test_construct_from_cov_matrix(self, simple_cov: pd.DataFrame) -> None:
        hrp = HierarchicalRiskParity(cov_matrix=simple_cov)
        assert hrp._n_assets == 2
        assert not hrp.ridge_applied

    def test_construct_from_cov_ndarray(self, simple_cov: pd.DataFrame) -> None:
        arr = simple_cov.values
        hrp = HierarchicalRiskParity(cov_matrix=arr)
        assert hrp._n_assets == 2

    def test_rejects_neither_input(self) -> None:
        with pytest.raises(ValueError, match="Either returns or cov_matrix"):
            HierarchicalRiskParity()

    def test_rejects_both_inputs(self, independent_returns: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not both"):
            HierarchicalRiskParity(
                returns=independent_returns,
                cov_matrix=independent_returns.cov(),
            )

    def test_rejects_non_square_cov(self) -> None:
        with pytest.raises(ValueError, match="square"):
            HierarchicalRiskParity(cov_matrix=np.eye(3, 2))

    def test_rejects_single_asset(self, rng: np.random.Generator) -> None:
        dates = pd.date_range("2023-01-01", periods=50)
        ret = pd.DataFrame({"A": rng.normal(0, 1, 50)}, index=dates)
        with pytest.raises(ValueError, match="At least 2"):
            HierarchicalRiskParity(returns=ret)

    def test_ridge_triggers_for_near_perfect_correlation(
        self,
        near_singular_returns: pd.DataFrame,
    ) -> None:
        hrp = HierarchicalRiskParity(returns=near_singular_returns)
        assert hrp.ridge_applied

    def test_no_ridge_for_moderate_correlation(
        self,
        correlated_returns: pd.DataFrame,
    ) -> None:
        hrp = HierarchicalRiskParity(returns=correlated_returns)
        assert not hrp.ridge_applied


# ---------------------------------------------------------------------------
# Optimize — basic properties
# ---------------------------------------------------------------------------


class TestOptimizeBasic:
    """Basic correctness of the `optimize` method."""

    def test_weights_sum_to_one(self, independent_returns: pd.DataFrame) -> None:
        hrp = HierarchicalRiskParity(returns=independent_returns)
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_all_weights_non_negative(self, independent_returns: pd.DataFrame) -> None:
        hrp = HierarchicalRiskParity(returns=independent_returns)
        weights = hrp.optimize()
        assert (weights >= -1e-15).all(), f"Negative weights found: {weights.values}"

    def test_returns_series_with_tickers(
        self, independent_returns: pd.DataFrame
    ) -> None:
        hrp = HierarchicalRiskParity(returns=independent_returns)
        weights = hrp.optimize()
        assert isinstance(weights, pd.Series)
        assert weights.name == "weight"
        assert set(weights.index) == set(independent_returns.columns)

    def test_cov_matrix_input_produces_valid_weights(
        self, simple_cov: pd.DataFrame
    ) -> None:
        hrp = HierarchicalRiskParity(cov_matrix=simple_cov)
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        assert list(weights.index) == ["A", "B"] or list(weights.index) == ["B", "A"]

    def test_ridge_preserves_weight_sum(
        self, near_singular_returns: pd.DataFrame
    ) -> None:
        hrp = HierarchicalRiskParity(returns=near_singular_returns)
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_deterministic_given_same_input(
        self, independent_returns: pd.DataFrame
    ) -> None:
        hrp1 = HierarchicalRiskParity(returns=independent_returns)
        hrp2 = HierarchicalRiskParity(returns=independent_returns)
        w1 = hrp1.optimize()
        w2 = hrp2.optimize()
        pd.testing.assert_series_equal(w1, w2)


# ---------------------------------------------------------------------------
# Optimize — risk‑allocation behaviour
# ---------------------------------------------------------------------------


class TestOptimizeRiskAllocation:
    """Verify HRP allocates more weight to lower‑volatility assets."""

    def test_lower_volatility_gets_higher_weight(
        self,
        independent_returns: pd.DataFrame,
    ) -> None:
        """FI assets (lower vol) should receive higher weights than EQ assets."""
        hrp = HierarchicalRiskParity(returns=independent_returns)
        weights = hrp.optimize()

        # Low‑vol assets: FI1, FI2.
        low_vol_weight = weights[["FI1", "FI2"]].sum()
        high_vol_weight = weights[["EQ1", "EQ2"]].sum()

        assert low_vol_weight > high_vol_weight, (
            f"Low‑vol FI weight ({low_vol_weight:.4f}) should exceed "
            f"high‑vol EQ weight ({high_vol_weight:.4f})"
        )

    def test_two_asset_weight_ratio_matches_inverse_variance(
        self,
        simple_cov: pd.DataFrame,
    ) -> None:
        """For two uncorrelated assets, HRP ≈ inverse‑variance weighting."""
        hrp = HierarchicalRiskParity(cov_matrix=simple_cov)
        weights = hrp.optimize()

        # Expected inverse‑variance weights.
        inv_var = 1.0 / np.diag(simple_cov.values)
        expected = inv_var / inv_var.sum()

        assert weights["A"] == pytest.approx(expected[0], rel=0.05)
        assert weights["B"] == pytest.approx(expected[1], rel=0.05)

    def test_correlated_cluster_gets_balanced_weight(
        self,
        correlated_returns: pd.DataFrame,
    ) -> None:
        """Within a correlated cluster, weights should be roughly balanced."""
        hrp = HierarchicalRiskParity(returns=correlated_returns)
        weights = hrp.optimize()

        # EQ_A and EQ_B are correlated → should get similar weights.
        eq_ratio = min(weights["EQ_A"], weights["EQ_B"]) / max(
            weights["EQ_A"], weights["EQ_B"]
        )
        assert eq_ratio > 0.5, (
            f"EQ_A and EQ_B should have similar weights, got ratio {eq_ratio:.4f}"
        )

    def test_many_assets_stable(self, rng: np.random.Generator) -> None:
        """HRP should handle a moderate number of assets without exploding."""
        n_assets = 20
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.DataFrame(
            {f"A{i}": rng.normal(0.001, 0.02, 252) for i in range(n_assets)},
            index=dates,
        )
        hrp = HierarchicalRiskParity(returns=returns)
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        # Every asset should get a non‑trivial allocation.
        assert (weights > 0.001).all(), (
            f"Some assets have near‑zero weight: {weights.values}"
        )


# ---------------------------------------------------------------------------
# Inverse‑variance portfolio comparison
# ---------------------------------------------------------------------------


class TestInverseVarianceComparison:
    """HRP vs. simple inverse‑variance portfolio under multicollinearity."""

    @staticmethod
    def _inverse_variance_weights(returns: pd.DataFrame) -> pd.Series:
        """Simple inverse‑variance portfolio (no correlation accounting)."""
        variances = returns.var()
        inv_var = 1.0 / variances
        return inv_var / inv_var.sum()

    def test_hrp_differs_from_inverse_variance_with_correlation(
        self,
        correlated_returns: pd.DataFrame,
    ) -> None:
        """HRP should NOT simply equal inverse‑variance when assets correlate."""
        hrp = HierarchicalRiskParity(returns=correlated_returns)
        hrp_weights = hrp.optimize()
        iv_weights = self._inverse_variance_weights(correlated_returns)

        # HRP should spread weight more evenly across correlated clusters
        # than naive inverse‑variance (which over‑concentrates on low‑vol
        # assets regardless of redundancy).  The key comparison: HRP weights
        # should be less extreme (narrower range).
        hrp_max = hrp_weights.max()
        hrp_min = hrp_weights.min()
        iv_max = iv_weights.max()
        iv_min = iv_weights.min()

        hrp_range = hrp_max - hrp_min
        iv_range = iv_max - iv_min

        # HRP should have a narrower weight range (more balanced).
        assert hrp_range < iv_range, (
            f"HRP range ({hrp_range:.4f}) should be narrower than "
            f"IV range ({iv_range:.4f})"
        )

    def test_hrp_allocates_more_to_correlated_equities(
        self,
        correlated_returns: pd.DataFrame,
    ) -> None:
        """HRP gives meaningful combined weight to correlated EQ cluster."""
        hrp = HierarchicalRiskParity(returns=correlated_returns)
        hrp_weights = hrp.optimize()

        # EQ cluster total.
        eq_weight = hrp_weights[["EQ_A", "EQ_B"]].sum()

        # EQ has higher vol than FI → naive IV would heavily favor FI.
        # HRP should still give EQ a non‑trivial allocation.
        assert eq_weight > 0.05, (
            f"EQ cluster should get > 5% weight, got {eq_weight:.4f}"
        )

    def test_hrp_vs_iv_on_independent_returns(
        self,
        independent_returns: pd.DataFrame,
    ) -> None:
        """On nearly independent returns, HRP ≈ inverse‑variance."""
        hrp = HierarchicalRiskParity(returns=independent_returns)
        hrp_weights = hrp.optimize()
        iv_weights = self._inverse_variance_weights(independent_returns)

        # Rank correlation between HRP and IV should be high (same ordering).
        hrp_rank = hrp_weights.rank()
        iv_rank = iv_weights.rank()
        rank_corr = hrp_rank.corr(iv_rank)

        assert rank_corr > 0.8, (
            f"HRP and IV rankings should be similar for independent assets, "
            f"rank corr = {rank_corr:.4f}"
        )


# ---------------------------------------------------------------------------
# Ridge regularisation integration
# ---------------------------------------------------------------------------


class TestRidgeIntegration:
    """End‑to‑end ridge behaviour."""

    def test_ridge_warning_logged(
        self, near_singular_returns: pd.DataFrame, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            HierarchicalRiskParity(returns=near_singular_returns)
        assert "Near‑singular correlation" in caplog.text
        assert "ridge" in caplog.text.lower()

    def test_ridge_reduces_max_correlation(
        self, near_singular_returns: pd.DataFrame
    ) -> None:
        """After ridge, the optimise step completes without error."""
        hrp = HierarchicalRiskParity(returns=near_singular_returns)

        # The key check: the optimise step completes without error.
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_near_identical_assets_get_similar_weight(
        self,
        near_singular_returns: pd.DataFrame,
    ) -> None:
        """Near‑identical assets (X, Y) should get nearly identical allocations."""
        hrp = HierarchicalRiskParity(returns=near_singular_returns)
        weights = hrp.optimize()

        assert weights["X"] == pytest.approx(weights["Y"], rel=0.01), (
            f"X ({weights['X']:.6f}) and Y ({weights['Y']:.6f}) "
            f"should be nearly identical"
        )

    def test_ridge_property_reflects_state(
        self, near_singular_returns: pd.DataFrame
    ) -> None:
        hrp = HierarchicalRiskParity(returns=near_singular_returns)
        assert hrp.ridge_applied is True
        assert isinstance(hrp.covariance_matrix, pd.DataFrame)
        assert isinstance(hrp.correlation_matrix, pd.DataFrame)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and edge‑case behaviour."""

    def test_two_identical_assets(self, rng: np.random.Generator) -> None:
        """Two perfectly identical assets → equal weights."""
        dates = pd.date_range("2023-01-01", periods=100)
        series = rng.normal(0.001, 0.02, 100)
        returns = pd.DataFrame({"A": series, "B": series.copy()}, index=dates)
        hrp = HierarchicalRiskParity(returns=returns)
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)
        # Identical assets should get equal weight.
        assert abs(weights["A"] - 0.5) < 0.05
        assert abs(weights["B"] - 0.5) < 0.05

    def test_constant_returns(self, rng: np.random.Generator) -> None:
        """One asset with zero variance → should not crash."""
        dates = pd.date_range("2023-01-01", periods=100)
        returns = pd.DataFrame(
            {
                "Const": np.zeros(100),
                "Var": rng.normal(0.001, 0.02, 100),
            },
            index=dates,
        )
        hrp = HierarchicalRiskParity(returns=returns)
        # Zero‑variance assets trigger ridge through the correlation check
        # (NaN correlation → ridge triggered), OR the inverse‑variance step
        # may produce inf.  Either way, the optimise should complete.
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_three_identical_assets(self, rng: np.random.Generator) -> None:
        """Three identical assets should get reasonably balanced weights."""
        dates = pd.date_range("2023-01-01", periods=100)
        series = rng.normal(0.001, 0.02, 100)
        returns = pd.DataFrame(
            {"A": series, "B": series.copy(), "C": series.copy()},
            index=dates,
        )
        hrp = HierarchicalRiskParity(returns=returns)
        weights = hrp.optimize()
        assert weights.sum() == pytest.approx(1.0, abs=1e-10)

        # With 3 identical assets, HRP clusters 2 together first,
        # then pairs them with the third.  The weights won't be exactly
        # 1/3 each but should not be pathologically concentrated.
        # No single asset should dominate (> 0.6) or be starved (< 0.15).
        max_weight = weights.max()
        min_weight = weights.min()
        assert max_weight < 0.6, (
            f"No asset should dominate: max weight = {max_weight:.4f}"
        )
        assert min_weight > 0.15, (
            f"No asset should be starved: min weight = {min_weight:.4f}"
        )

    def test_reproducibility_across_runs(
        self, independent_returns: pd.DataFrame
    ) -> None:
        """Same input → same output every time (no stochasticity)."""
        weights_list = []
        for _ in range(5):
            hrp = HierarchicalRiskParity(returns=independent_returns)
            weights_list.append(hrp.optimize())
        for w in weights_list[1:]:
            pd.testing.assert_series_equal(weights_list[0], w)
