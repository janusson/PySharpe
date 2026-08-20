"""Tests for the historical backtest engine.

.. note::

    **Canadian ETF Context** — Backtests simulate CAD-denominated portfolios
    with drift-based, relative, calendar, and volatility-triggered
    rebalancing strategies.  All price data is synthetic; no live market
    data is used.  Test tickers are synthetic placeholders.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.backtest_engine import (
    BacktestResult,
    HistoricalBacktester,
    WalkForwardBacktester,
)
from pysharpe.optimization.base import OptimizationResult


def test_historical_backtester_initialization():
    prices = pd.DataFrame(
        {"AAPL": [150.0, 151.0], "MSFT": [250.0, 252.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    target_weights = {"AAPL": 0.6, "MSFT": 0.4}

    backtester = HistoricalBacktester(
        prices=prices, target_weights=target_weights, initial_capital=10000.0
    )

    assert backtester.initial_capital == 10000.0
    assert backtester.assets == ["AAPL", "MSFT"]
    assert np.allclose(backtester.targets, [0.6, 0.4])


def test_historical_backtester_no_overlapping_assets():
    prices = pd.DataFrame(
        {"TSLA": [200.0]}, index=pd.date_range("2024-01-01", periods=1)
    )

    target_weights = {"AAPL": 1.0}

    with pytest.raises(ValueError, match="No overlapping assets"):
        HistoricalBacktester(prices=prices, target_weights=target_weights)


def test_historical_backtester_zero_target_weight():
    prices = pd.DataFrame(
        {"AAPL": [150.0]}, index=pd.date_range("2024-01-01", periods=1)
    )

    target_weights = {"AAPL": 0.0}

    with pytest.raises(ValueError, match="Total target weight must be positive"):
        HistoricalBacktester(prices=prices, target_weights=target_weights)


def test_historical_backtester_empty_prices():
    prices = pd.DataFrame(columns=["AAPL"])
    target_weights = {"AAPL": 1.0}

    backtester = HistoricalBacktester(prices=prices, target_weights=target_weights)
    result = backtester.run()

    assert result.portfolio_value.empty
    assert result.historical_weights.empty
    assert result.rebalance_events.empty


def test_historical_backtester_run_no_rebalance():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 200.0], "MSFT": [100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    target_weights = {"AAPL": 0.5, "MSFT": 0.5}

    backtester = HistoricalBacktester(
        prices=prices, target_weights=target_weights, initial_capital=1000.0
    )

    result = backtester.run()

    # Initial capital = 1000. Targets = 50% / 50%
    # Day 0: AAPL=$500 (5 shares), MSFT=$500 (5 shares)
    # Day 1: AAPL=5*200=$1000, MSFT=5*100=$500. Total = $1500

    assert np.isclose(result.portfolio_value.iloc[0], 1000.0)
    assert np.isclose(result.portfolio_value.iloc[1], 1500.0)

    assert np.isclose(result.historical_weights.iloc[0]["AAPL"], 0.5)
    assert np.isclose(result.historical_weights.iloc[1]["AAPL"], 1000.0 / 1500.0)


def test_historical_backtester_run_absolute_drift_rebalance():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 200.0, 200.0], "MSFT": [100.0, 100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    target_weights = {"AAPL": 0.5, "MSFT": 0.5}

    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=target_weights,
        initial_capital=1000.0,
        abs_band=0.1,  # 10% drift triggers rebalance
    )

    result = backtester.run()

    # Day 0: AAPL=$500 (5 shares), MSFT=$500 (5 shares)
    # Day 1: AAPL=5*200=$1000, MSFT=5*100=$500. Total = $1500. Weight AAPL = 66.6%.
    # Drift = 16.6% > 10%
    # Rebalance triggered on Day 1.
    assert len(result.rebalance_events) == 1
    assert result.rebalance_events[0] == pd.Timestamp("2024-01-02")
    assert np.isclose(result.historical_weights.iloc[1]["AAPL"], 0.5)


def test_historical_backtester_run_relative_drift_rebalance():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 120.0, 120.0], "MSFT": [100.0, 100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    target_weights = {"AAPL": 0.5, "MSFT": 0.5}

    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=target_weights,
        initial_capital=1000.0,
        rel_band=0.08,  # 8% relative drift -> 0.5 * 1.08 = 0.54
    )

    result = backtester.run()

    # Day 0: 50% / 50%
    # Day 1: AAPL = 500 * 1.2 = 600. MSFT = 500. Total = 1100.
    # AAPL wt = 600/1100 = 54.54%
    # Relative drift: |54.54% - 50%| / 50% = 9.09% > 8%. Rebalance triggered.
    assert len(result.rebalance_events) == 1
    assert result.rebalance_events[0] == pd.Timestamp("2024-01-02")
    assert np.isclose(result.historical_weights.iloc[1]["AAPL"], 0.5)


def test_historical_backtester_run_calendar_rebalance():
    prices = pd.DataFrame(
        {"AAPL": [100.0, 120.0, 150.0, 150.0], "MSFT": [100.0, 100.0, 100.0, 100.0]},
        index=pd.date_range("2024-01-29", periods=4),
    )

    target_weights = {"AAPL": 0.5, "MSFT": 0.5}

    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=target_weights,
        initial_capital=1000.0,
        rebalance_freq="M",  # Monthly end rebalance
    )

    result = backtester.run()

    # "2024-01-29" (M), "2024-01-30" (T), "2024-01-31" (W), "2024-02-01" (Th)
    # The monthly calendar boundary is Jan 31st, plus the final day always triggers.
    assert len(result.rebalance_events) == 2
    assert result.rebalance_events[0] == pd.Timestamp("2024-01-31")
    assert result.rebalance_events[1] == pd.Timestamp("2024-02-01")


def test_historical_backtester_run_vol_threshold_rebalance():
    # We need at least 21 days to compute a 20-day realized volatility
    dates = pd.date_range("2024-01-01", periods=25)
    # Asset price that is very volatile on the 21st day (index 20)
    prices = pd.DataFrame(
        {"AAPL": [100.0] * 20 + [150.0, 100.0, 150.0, 100.0, 150.0]}, index=dates
    )

    target_weights = {"AAPL": 1.0}

    # Set a low volatility threshold so it's easily triggered by the price spike
    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=target_weights,
        initial_capital=1000.0,
        vol_threshold=0.10,  # 10% annualized vol
    )

    result = backtester.run()

    # The price spike at t=20 will cause log returns to be non-zero for the first time
    # in the 20-day window [t-20 : t+1].
    # At t=20, window_values = prices[0:21]. returns[0:20] = [0, 0, ..., log(150/100)]
    # This will trigger a rebalance if rolling_vol > 0.10
    assert len(result.rebalance_events) > 0
    assert pd.Timestamp("2024-01-21") in result.rebalance_events


class _DummyOptimizer:
    def __init__(self, data: pd.DataFrame):
        self.assets = data.columns.tolist()

    def optimize(self) -> OptimizationResult:
        n = len(self.assets)
        weights = {asset: 1.0 / n for asset in self.assets}
        return OptimizationResult(
            weights=weights, expected_return=0.1, volatility=0.15, sharpe_ratio=0.5
        )


@pytest.fixture
def _dummy_prices():
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


def test_walk_forward_backtester_run(_dummy_prices):
    backtester = WalkForwardBacktester(
        optimizer_factory=_DummyOptimizer,
        train_window_days=30,
        test_window_days=10,
        initial_capital=10000.0,
    )
    result = backtester.run(_dummy_prices)

    assert isinstance(result, BacktestResult)
    assert len(result.portfolio_value) == 70
    assert len(result.historical_weights) == 70
    assert "A" in result.historical_weights.columns
    assert "B" in result.historical_weights.columns


def test_walk_forward_compare_to_benchmark(_dummy_prices):
    backtester = WalkForwardBacktester(
        optimizer_factory=_DummyOptimizer,
        train_window_days=30,
        test_window_days=10,
        initial_capital=10000.0,
    )

    comparison = backtester.compare_to_benchmark(_dummy_prices, benchmark_ticker="SPY")

    assert "Strategy" in comparison.columns
    assert "Benchmark" in comparison.columns
    assert len(comparison) == 70
    assert pytest.approx(comparison["Strategy"].iloc[0], rel=1e-3) == 10000.0
    assert pytest.approx(comparison["Benchmark"].iloc[0], rel=1e-3) == 10000.0


def test_walk_forward_insufficient_data(_dummy_prices):
    backtester = WalkForwardBacktester(
        optimizer_factory=_DummyOptimizer,
        train_window_days=80,
        test_window_days=30,
        initial_capital=10000.0,
    )
    with pytest.raises(ValueError, match="Insufficient data"):
        backtester.run(_dummy_prices)


# ---------------------------------------------------------------------------
# Walk-forward with missing price data (NaNs)
# ---------------------------------------------------------------------------


class TestWalkForwardMissingData:
    """NaNs in the middle of a fold must not crash the engine or leak
    information across the train/test boundary."""

    def _prices(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n)
        return pd.DataFrame(
            {
                "A": np.exp(rng.normal(0.001, 0.02, n).cumsum()),
                "B": np.exp(rng.normal(0.002, 0.03, n).cumsum()),
            },
            index=dates,
        )

    def test_nan_in_middle_of_series(self):
        """NaN prices for one asset inside the data range: rows are dropped
        listwise and the backtest completes with finite values."""
        prices = self._prices(120, seed=7)
        prices.iloc[55:60, 0] = np.nan  # A missing mid-series

        backtester = WalkForwardBacktester(
            optimizer_factory=_DummyOptimizer,
            train_window_days=30,
            test_window_days=10,
            initial_capital=10000.0,
        )
        result = backtester.run(prices)

        # Walk-forward output covers only the out-of-sample portion
        # (rows after the first training window): 120 − 5 dropped − 30 train.
        expected_days = 120 - 5 - 30
        assert len(result.portfolio_value) == expected_days
        assert np.all(np.isfinite(result.portfolio_value.values))
        assert np.all(np.isfinite(result.historical_weights.values))
        # Dropped dates must not appear in the result index.
        missing_dates = prices.index[55:60]
        assert not set(missing_dates).intersection(result.portfolio_value.index)

    def test_nan_inside_a_specific_fold(self):
        """NaNs confined to one asset within what would be a single test
        window: the engine drops them and still simulates every window."""
        prices = self._prices(110, seed=11)
        # Fold boundaries with train=30/test=10: test windows start at 30,
        # 40, 50, ... Place NaNs inside the 40–50 test window.
        prices.iloc[42:45, 1] = np.nan

        backtester = WalkForwardBacktester(
            optimizer_factory=_DummyOptimizer,
            train_window_days=30,
            test_window_days=10,
            initial_capital=10000.0,
        )
        result = backtester.run(prices)

        assert len(result.portfolio_value) == 107 - 30
        assert np.all(np.isfinite(result.portfolio_value.values))
        assert len(result.rebalance_events) >= 4

    def test_nan_at_start_of_series(self):
        """An asset with delayed listing (NaN at the start) is handled."""
        prices = self._prices(100, seed=13)
        prices.iloc[:8, 1] = np.nan

        result = WalkForwardBacktester(
            optimizer_factory=_DummyOptimizer,
            train_window_days=30,
            test_window_days=10,
        ).run(prices)

        assert len(result.portfolio_value) == 92 - 30
        assert np.all(np.isfinite(result.portfolio_value.values))

    def test_fully_missing_asset_is_dropped_not_fatal(self, caplog):
        """An asset with NO prices at all is excluded with a warning instead
        of wiping out the entire backtest."""
        prices = self._prices(100, seed=17)
        prices["NeverListed"] = np.nan

        with caplog.at_level(
            logging.WARNING, logger="pysharpe.analysis.backtest_engine"
        ):
            result = WalkForwardBacktester(
                optimizer_factory=_DummyOptimizer,
                train_window_days=30,
                test_window_days=10,
            ).run(prices)

        assert any("no price coverage" in r.message for r in caplog.records)
        assert len(result.portfolio_value) == 100 - 30
        assert "NeverListed" not in result.historical_weights.columns
        assert np.all(np.isfinite(result.portfolio_value.values))

    def test_all_assets_missing_raises(self):
        prices = self._prices(100, seed=19)
        prices[:] = np.nan
        with pytest.raises(ValueError, match="No usable price data"):
            WalkForwardBacktester(
                optimizer_factory=_DummyOptimizer,
                train_window_days=30,
                test_window_days=10,
            ).run(prices)

    def test_insufficient_rows_after_drop_raises(self):
        prices = self._prices(60, seed=23)
        prices.iloc[:40, 0] = np.nan  # leaves only 20 shared rows
        with pytest.raises(ValueError, match="Insufficient data"):
            WalkForwardBacktester(
                optimizer_factory=_DummyOptimizer,
                train_window_days=30,
                test_window_days=10,
            ).run(prices)

    def test_nan_handling_is_deterministic(self):
        prices = self._prices(100, seed=29)
        prices.iloc[30:35, 0] = np.nan

        r1 = WalkForwardBacktester(
            optimizer_factory=_DummyOptimizer, train_window_days=30, test_window_days=10
        ).run(prices)
        r2 = WalkForwardBacktester(
            optimizer_factory=_DummyOptimizer, train_window_days=30, test_window_days=10
        ).run(prices)
        pd.testing.assert_series_equal(r1.portfolio_value, r2.portfolio_value)

    def test_historical_backtester_nan_mid_series(self):
        """HistoricalBacktester also survives NaN gaps via listwise drops."""
        prices = self._prices(30, seed=31)
        prices.iloc[10:12, 0] = np.nan
        result = HistoricalBacktester(
            prices=prices,
            target_weights={"A": 0.5, "B": 0.5},
            initial_capital=10000.0,
            abs_band=0.05,
        ).run()
        assert np.all(np.isfinite(result.portfolio_value.values))
        assert len(result.portfolio_value) == 28
