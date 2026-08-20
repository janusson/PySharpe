import numpy as np
import pandas as pd
import pytest
from conftest import _SAMPLING_SKIP_REASON, pymc_sampling_works

from pysharpe.exceptions import ExecutionConfigError
from pysharpe.optimization.base import OptimizationResult, PortfolioOptimizer
from pysharpe.optimization.bayesian import BayesianOptimizer
from pysharpe.optimization.sharpe_optimizer import (
    SharpeOptimizer,
    SharpeOptimizerConfig,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(12)
    dates = pd.date_range("2023-01-01", periods=100)
    data = pd.DataFrame(
        {
            "A": np.exp(rng.normal(0.001, 0.02, 100).cumsum()),
            "B": np.exp(rng.normal(0.002, 0.03, 100).cumsum()),
        },
        index=dates,
    )
    return data


@pytest.fixture
def steady_prices():
    """Monotonically increasing price series — ensures positive, deterministic returns."""
    dates = pd.date_range("2023-01-01", periods=100)
    return pd.DataFrame(
        {
            "A": 100.0 + np.arange(100) * 0.2,
            "B": 100.0 + np.arange(100) * 0.1,
        },
        index=dates,
    )


@pytest.fixture
def three_asset_prices():
    """Three-asset price series with distinct risk/return profiles."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2023-01-01", periods=200)
    returns = pd.DataFrame(index=dates)
    returns["A"] = rng.normal(0.001, 0.02, 200)  # High return, high vol
    returns["B"] = rng.normal(0.0005, 0.005, 200)  # Low return, low vol
    returns["C"] = rng.normal(0.0002, 0.01, 200)  # Mid
    prices = (1 + returns).cumprod() * 100
    prices.index.name = "Date"
    return prices


def test_sharpe_optimizer_mer_deduction_is_decimal_not_percentage(steady_prices):
    """mer_by_ticker values must be decimal fractions (0.01 = 1%), not percentage points."""
    equal_weights = np.array([0.5, 0.5])

    opt_no_mer = SharpeOptimizer(
        prices=steady_prices.copy(),
        config=SharpeOptimizerConfig(max_weight=1.0),
    )
    ret_no_mer, _, _ = opt_no_mer.calculate_portfolio_performance(equal_weights)

    opt_with_mer = SharpeOptimizer(
        prices=steady_prices.copy(),
        config=SharpeOptimizerConfig(
            max_weight=1.0,
            mer_by_ticker={"A": 0.01, "B": 0.01},
        ),
    )
    ret_with_mer, _, _ = opt_with_mer.calculate_portfolio_performance(equal_weights)

    # Equal weights × 0.01 each = 0.01 total deduction
    deduction = ret_no_mer - ret_with_mer
    assert abs(deduction - 0.01) < 1e-9, (
        f"Expected MER deduction of 0.01 (1% expressed as decimal), got {deduction:.8f}. "
        "mer_by_ticker should use decimal fractions, not percentage points."
    )


def test_sharpe_optimizer_implements_protocol(sample_data):
    from pysharpe.optimization.sharpe_optimizer import SharpeOptimizerConfig

    config = SharpeOptimizerConfig(max_weight=1.0)
    optimizer = SharpeOptimizer(prices=sample_data, config=config)
    assert isinstance(optimizer, PortfolioOptimizer)

    result = optimizer.optimize()
    assert isinstance(result, OptimizationResult)
    assert "A" in result.weights
    assert "B" in result.weights
    assert 0 <= result.expected_return
    assert 0 < result.volatility
    assert isinstance(result.sharpe_ratio, float)


# ---------------------------------------------------------------------------
# Tests for max_portfolio_mer constraint
# ---------------------------------------------------------------------------


def test_mer_constraint_respected(three_asset_prices):
    """The aggregate portfolio MER must not exceed max_portfolio_mer."""
    mer_mapping = {"A": 0.05, "B": 0.001, "C": 0.02}
    max_mer = 0.015  # 1.5% — forces optimizer away from high-MER asset A

    config = SharpeOptimizerConfig(
        max_weight=1.0,
        mer_by_ticker=mer_mapping,
        max_portfolio_mer=max_mer,
    )
    optimizer = SharpeOptimizer(prices=three_asset_prices, config=config)
    result = optimizer.optimize()

    port_mer = sum(result.weights.get(t, 0.0) * mer_mapping[t] for t in result.weights)
    assert port_mer <= max_mer + 1e-5, (
        f"Portfolio MER {port_mer:.6f} exceeded limit {max_mer}"
    )


def test_mer_constraint_with_max_weight(three_asset_prices):
    """MER constraint combined with per-asset max_weight bounds."""
    mer_mapping = {"A": 0.05, "B": 0.001, "C": 0.02}
    max_mer = 0.02

    config = SharpeOptimizerConfig(
        max_weight=0.40,
        mer_by_ticker=mer_mapping,
        max_portfolio_mer=max_mer,
    )
    optimizer = SharpeOptimizer(prices=three_asset_prices, config=config)
    result = optimizer.optimize()

    port_mer = sum(result.weights.get(t, 0.0) * mer_mapping[t] for t in result.weights)
    # Both constraints must hold
    assert port_mer <= max_mer + 1e-5
    for w in result.weights.values():
        assert w <= 0.40 + 1e-5


def test_infeasible_mer_constraint_raises(three_asset_prices):
    """When every individual MER exceeds max_portfolio_mer, raise ExecutionConfigError."""
    mer_mapping = {"A": 0.05, "B": 0.03, "C": 0.04}
    max_mer = 0.02  # All assets have MER > 0.02 — impossible

    config = SharpeOptimizerConfig(
        max_weight=1.0,
        mer_by_ticker=mer_mapping,
        max_portfolio_mer=max_mer,
    )
    optimizer = SharpeOptimizer(prices=three_asset_prices, config=config)

    with pytest.raises(ExecutionConfigError, match="infeasible"):
        optimizer.optimize()


def test_mer_constraint_not_enforced_when_not_set(three_asset_prices):
    """When max_portfolio_mer is None, no constraint is applied (backward compat)."""
    mer_mapping = {"A": 0.05, "B": 0.001, "C": 0.02}

    config = SharpeOptimizerConfig(
        max_weight=1.0,
        mer_by_ticker=mer_mapping,
        max_portfolio_mer=None,
    )
    optimizer = SharpeOptimizer(prices=three_asset_prices, config=config)
    result = optimizer.optimize()

    # Should complete successfully even though A (0.05) is high
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_mer_constraint_no_mer_mapping_no_error(three_asset_prices):
    """max_portfolio_mer with empty mer_by_ticker should not raise or constrain."""
    config = SharpeOptimizerConfig(
        max_weight=1.0,
        mer_by_ticker={},
        max_portfolio_mer=0.01,
    )
    optimizer = SharpeOptimizer(prices=three_asset_prices, config=config)
    result = optimizer.optimize()

    # Should complete normally
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(
    not pymc_sampling_works(),
    reason=_SAMPLING_SKIP_REASON,
)
def test_bayesian_optimizer_implements_protocol(sample_data):
    returns = sample_data.pct_change().dropna()
    optimizer = BayesianOptimizer(returns=returns, random_seed=42)
    assert isinstance(optimizer, PortfolioOptimizer)

    # Small sampling for speed
    optimizer.fit_returns_model(draws=50, tune=50, chains=1, cores=1)
    result = optimizer.optimize()

    assert isinstance(result, OptimizationResult)
    assert "A" in result.weights
    assert "B" in result.weights
    assert isinstance(result.expected_return, float)
    assert isinstance(result.volatility, float)
    assert isinstance(result.sharpe_ratio, float)


def test_shrinkage_expected_return_does_not_collapse_with_correlated_assets():
    """Highly correlated assets must still produce differentiated expected returns.

    When sample covariance is near-singular (common with broad-market ETFs),
    the old implementation fell back to returning the grand mean for every
    asset, forcing equal-weight allocations from the optimizer.  Ledoit-Wolf
    shrunk covariance must keep the inversion well-conditioned.
    """
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(99)
    n_days = 252
    # Two assets with 0.95 daily correlation
    base = rng.normal(0.0005, 0.01, n_days)
    a = base + rng.normal(0, 0.002, n_days)
    b = base + rng.normal(0, 0.003, n_days)

    prices = pd.DataFrame(
        {"VFV": 100 * (1 + a).cumprod(), "VDY": 100 * (1 + b).cumprod()},
        index=pd.date_range("2024-01-01", periods=n_days, freq="B"),
    )

    mu = shrinkage_expected_return(prices)

    # The two expected returns must NOT be identical.
    assert not np.isclose(mu["VFV"], mu["VDY"]), (
        f"Shrunk returns collapsed to identical values: VFV={mu['VFV']:.6f}, "
        f"VDY={mu['VDY']:.6f}.  This forces equal-weight allocations."
    )


def test_constant_expected_return_all_equal():
    """constant_expected_return produces identical values for every asset."""
    from pysharpe.optimization.expected_returns import constant_expected_return

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 * (1 + rng.normal(0.001, 0.02, 500)).cumprod(),
            "B": 100 * (1 + rng.normal(0.0005, 0.01, 500)).cumprod(),
            "C": 100 * (1 + rng.normal(0.002, 0.03, 500)).cumprod(),
        },
        index=dates,
    )

    mu = constant_expected_return(prices)

    # All assets should have identical expected returns (the grand mean)
    assert len(mu.unique()) == 1, (
        f"Expected identical returns for all assets, got: {mu.to_dict()}"
    )
    # Should match the manual calculation
    returns = prices.pct_change().dropna()
    grand_mean = returns.mean().mean() * 252
    assert abs(mu.iloc[0] - grand_mean) < 1e-10


def test_constant_expected_return_single_asset():
    """constant_expected_return works with a single asset."""
    from pysharpe.optimization.expected_returns import constant_expected_return

    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {"SINGLE": 100 * (1 + rng.normal(0.001, 0.015, 100)).cumprod()},
        index=dates,
    )

    mu = constant_expected_return(prices)
    assert isinstance(mu, pd.Series)
    assert len(mu) == 1
    assert mu.index[0] == "SINGLE"


def test_shrinkage_expected_return_single_asset_falls_back():
    """shrinkage_expected_return falls back to mean_historical_return for n < 2."""
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(77)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {"ONLY": 100 * (1 + rng.normal(0.001, 0.02, 100)).cumprod()},
        index=dates,
    )

    mu = shrinkage_expected_return(prices)
    # For single asset, should just be the mean historical return
    from pypfopt.expected_returns import mean_historical_return

    expected = mean_historical_return(prices)
    assert abs(mu.iloc[0] - expected.iloc[0]) < 1e-10


def test_shrinkage_expected_return_insufficient_data_falls_back():
    """shrinkage_expected_return with too few periods falls back to mean."""
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(13)
    # 5 assets, 6 periods → n_periods (5) < n_assets (5) + 2
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 + rng.normal(0, 1, 6).cumsum(),
            "B": 100 + rng.normal(0, 1, 6).cumsum(),
            "C": 100 + rng.normal(0, 1, 6).cumsum(),
            "D": 100 + rng.normal(0, 1, 6).cumsum(),
            "E": 100 + rng.normal(0, 1, 6).cumsum(),
        },
        index=dates,
    )

    mu = shrinkage_expected_return(prices)
    # Should still return a valid Series
    assert len(mu) == 5
    assert not mu.isna().any()


def test_shrinkage_expected_return_shrinks_toward_grand_mean():
    """Shrunk estimates should have lower cross-sectional dispersion than raw."""
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Create assets with clearly different mean returns
    returns = pd.DataFrame(
        {
            "High": rng.normal(0.0015, 0.02, n_days),  # ~38% annual
            "Mid": rng.normal(0.0008, 0.015, n_days),  # ~20% annual
            "Low": rng.normal(0.0002, 0.01, n_days),  # ~5% annual
        },
        index=dates,
    )
    prices = (1 + returns).cumprod() * 100

    mu_shrunk = shrinkage_expected_return(prices)
    mu_raw = returns.mean() * 252

    # The key property: shrunk estimates have lower cross-sectional variance
    # than raw estimates (shrinkage pulls extremes toward the center)
    shrunk_std = mu_shrunk.std()
    raw_std = mu_raw.std()
    assert shrunk_std < raw_std, (
        f"Shrunk std ({shrunk_std:.6f}) should be less than raw std ({raw_std:.6f})"
    )


def test_shrinkage_expected_return_respects_shrinkage_floor():
    """shrinkage_floor forces minimum shrinkage toward grand mean."""
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(42)
    # Use only 100 days with 2 very different assets — this keeps phi low
    # (not enough data to confidently differentiate), letting the floor take effect.
    n_days = 100
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Very different return profiles to prevent natural shrinkage from dominating
    returns_df = pd.DataFrame(
        {
            "A": rng.normal(0.003, 0.04, n_days),  # High return, high vol
            "B": rng.normal(-0.001, 0.01, n_days),  # Low/negative, low vol
        },
        index=dates,
    )
    prices = (1 + returns_df).cumprod() * 100

    # With floor=0.0, data-driven shrinkage only
    mu_no_floor = shrinkage_expected_return(prices, shrinkage_floor=0.0)
    # With floor=0.5, minimum 50% shrinkage — forces stronger contraction
    mu_floor = shrinkage_expected_return(prices, shrinkage_floor=0.5)

    # Cross-sectional dispersion should be lower with the floor
    std_no_floor = mu_no_floor.std()
    std_floor = mu_floor.std()
    assert std_floor <= std_no_floor + 1e-10, (
        f"Floor=0.5 should not increase dispersion. "
        f"std_no_floor={std_no_floor:.6f}, std_floor={std_floor:.6f}"
    )


def test_shrinkage_expected_return_floor_at_one():
    """shrinkage_floor=1.0 forces all estimates to the grand mean."""
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 * (1 + rng.normal(0.001, 0.02, n_days)).cumprod(),
            "B": 100 * (1 + rng.normal(0.002, 0.03, n_days)).cumprod(),
        },
        index=dates,
    )

    mu = shrinkage_expected_return(prices, shrinkage_floor=1.0)
    # With floor=1.0, all estimates should equal the grand mean
    assert np.isclose(mu["A"], mu["B"]), (
        f"Floor=1.0 should collapse to grand mean: A={mu['A']:.6f}, B={mu['B']:.6f}"
    )


def test_shrinkage_expected_return_all_equal_returns():
    """When all assets have identically equal returns, shrinkage is appropriate."""
    from pysharpe.optimization.expected_returns import shrinkage_expected_return

    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Identical returns for both assets
    base = rng.normal(0.001, 0.02, n_days)
    prices = pd.DataFrame(
        {
            "X": 100 * (1 + base).cumprod(),
            "Y": 100 * (1 + base).cumprod(),
        },
        index=dates,
    )

    mu = shrinkage_expected_return(prices)
    # With identical assets, shrunk returns should be very close to each other
    assert abs(mu["X"] - mu["Y"]) < 0.01, (
        f"Identical assets should have near-identical shrunk returns: "
        f"X={mu['X']:.6f}, Y={mu['Y']:.6f}"
    )
