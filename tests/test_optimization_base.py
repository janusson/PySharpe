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
    dates = pd.date_range("2023-01-01", periods=100)
    data = pd.DataFrame(
        {
            "A": np.exp(np.random.normal(0.001, 0.02, 100).cumsum()),
            "B": np.exp(np.random.normal(0.002, 0.03, 100).cumsum()),
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
    np.random.seed(99)
    dates = pd.date_range("2023-01-01", periods=200)
    returns = pd.DataFrame(index=dates)
    returns["A"] = np.random.normal(0.001, 0.02, 200)  # High return, high vol
    returns["B"] = np.random.normal(0.0005, 0.005, 200)  # Low return, low vol
    returns["C"] = np.random.normal(0.0002, 0.01, 200)  # Mid
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
