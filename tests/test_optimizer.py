"""Tests for the PortfolioOptimizer class."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from pysharpe.models import PortfolioAllocation, PortfolioPerformance
from pysharpe.optimization.optimizer import OptimizationResult, PortfolioOptimizer


@dataclass
class _DummyEfficientFrontier:
    cleaned_weights: dict

    def __post_init__(self) -> None:
        self.constraints = []
        self.bounds = None

    def add_constraint(self, constraint):  # noqa: D401 - testing stub
        self.constraints.append(constraint)

    def max_sharpe(self):  # noqa: D401 - testing stub
        return self.cleaned_weights

    def clean_weights(self):  # noqa: D401 - testing stub
        return self.cleaned_weights

    def portfolio_performance(self, *, verbose: bool = False):  # noqa: D401 - testing stub
        return 0.12, 0.08, 1.5


def _dummy_frontier_builder(cleaned_weights: dict):
    def _builder(self):
        return _DummyEfficientFrontier(cleaned_weights)

    return _builder


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104],
            "BBB": [200, 202, 201, 203, 205],
        },
        index=dates,
    )


def test_portfolio_optimizer_validates_non_empty_data():
    with pytest.raises(ValueError):
        PortfolioOptimizer(pd.DataFrame())


def test_portfolio_optimizer_returns_copies():
    prices = _sample_prices()
    optimizer = PortfolioOptimizer(prices)

    prices_pop = optimizer.prices
    prices_pop.iloc[0, 0] = -99
    assert prices.iloc[0, 0] == 100

    returns_first = optimizer.returns
    returns_second = optimizer.returns
    assert returns_first is not returns_second
    returns_first.iloc[0, 0] = -0.5
    assert returns_second.iloc[0, 0] != -0.5


def test_portfolio_optimizer_max_sharpe(monkeypatch):
    prices = _sample_prices()
    optimizer = PortfolioOptimizer(prices)

    weights = {"AAA": 0.4, "BBB": 0.6}
    monkeypatch.setattr(
        PortfolioOptimizer,
        "_build_efficient_frontier",
        _dummy_frontier_builder(weights),
    )

    result = optimizer.max_sharpe(weight_bounds=(0.0, 0.8))

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.allocation, PortfolioAllocation)
    assert result.allocation.weights == weights
    assert isinstance(result.performance, PortfolioPerformance)
    assert result.performance.sharpe_ratio == pytest.approx(1.5)


def test_portfolio_optimizer_rebalance_creates_new_instance():
    prices = _sample_prices()
    optimizer = PortfolioOptimizer(prices)

    start = prices.index[1]
    rebalanced = optimizer.rebalance(["AAA"], start=start, end=prices.index[-2])

    assert isinstance(rebalanced, PortfolioOptimizer)
    pd.testing.assert_index_equal(rebalanced.prices.index, prices.loc[start:prices.index[-2]].index)
    assert list(rebalanced.prices.columns) == ["AAA"]
