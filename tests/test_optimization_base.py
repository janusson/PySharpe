import numpy as np
import pandas as pd
import pytest

from pysharpe.optimization.base import OptimizationResult, PortfolioOptimizer
from pysharpe.optimization.bayesian import BayesianOptimizer
from pysharpe.optimization.sharpe_optimizer import SharpeOptimizer


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


def test_sharpe_optimizer_implements_protocol(sample_data):
    optimizer = SharpeOptimizer(prices=sample_data)
    assert isinstance(optimizer, PortfolioOptimizer)

    result = optimizer.optimize()
    assert isinstance(result, OptimizationResult)
    assert "A" in result.weights
    assert "B" in result.weights
    assert 0 <= result.expected_return
    assert 0 < result.volatility
    assert isinstance(result.sharpe_ratio, float)


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
