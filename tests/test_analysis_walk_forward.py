import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.backtest_engine import BacktestResult, WalkForwardBacktester
from pysharpe.optimization.base import OptimizationResult


class DummyOptimizer:
    def __init__(self, data: pd.DataFrame):
        self.assets = data.columns.tolist()

    def optimize(self) -> OptimizationResult:
        n = len(self.assets)
        weights = {asset: 1.0 / n for asset in self.assets}
        return OptimizationResult(
            weights=weights, expected_return=0.1, volatility=0.15, sharpe_ratio=0.5
        )


@pytest.fixture
def dummy_prices():
    dates = pd.date_range("2023-01-01", periods=100)
    data = pd.DataFrame(
        {
            "A": np.exp(np.random.normal(0.001, 0.02, 100).cumsum()),
            "B": np.exp(np.random.normal(0.002, 0.03, 100).cumsum()),
            "SPY": np.exp(np.random.normal(0.0005, 0.015, 100).cumsum()),
        },
        index=dates,
    )
    return data


def test_walk_forward_backtester_run(dummy_prices):
    backtester = WalkForwardBacktester(
        optimizer_factory=DummyOptimizer,
        train_window_days=30,
        test_window_days=10,
        initial_capital=10000.0,
    )
    result = backtester.run(dummy_prices)

    assert isinstance(result, BacktestResult)
    # The simulation starts out-of-sample on day 30, and runs up to day 100.
    assert len(result.portfolio_value) == 70  # Total days - train_window
    assert len(result.historical_weights) == 70
    assert "A" in result.historical_weights.columns
    assert "B" in result.historical_weights.columns


def test_compare_to_benchmark(dummy_prices):
    backtester = WalkForwardBacktester(
        optimizer_factory=DummyOptimizer,
        train_window_days=30,
        test_window_days=10,
        initial_capital=10000.0,
    )

    comparison = backtester.compare_to_benchmark(dummy_prices, benchmark_ticker="SPY")

    assert "Strategy" in comparison.columns
    assert "Benchmark" in comparison.columns
    assert len(comparison) == 70

    # First row of Strategy should equal initial_capital, or near it depending on how the backtester initializes the sub-window
    # Sub backtester initiates day 0 with initial_capital.
    # Here day 0 of the first test window is index 30 of dummy_prices
    assert pytest.approx(comparison["Strategy"].iloc[0], rel=1e-3) == 10000.0
    assert pytest.approx(comparison["Benchmark"].iloc[0], rel=1e-3) == 10000.0


def test_insufficient_data(dummy_prices):
    backtester = WalkForwardBacktester(
        optimizer_factory=DummyOptimizer,
        train_window_days=80,
        test_window_days=30,  # Total 110 > 100
        initial_capital=10000.0,
    )
    with pytest.raises(ValueError, match="Insufficient data"):
        backtester.run(dummy_prices)
