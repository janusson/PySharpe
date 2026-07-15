"""Tests for the historical backtest engine.

.. note::

    **Canadian ETF Context** — Backtests simulate CAD-denominated portfolios
    with drift-based, relative, calendar, and volatility-triggered
    rebalancing strategies.  All price data is synthetic; no live market
    data is used.  Test tickers are synthetic placeholders.
"""
import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.backtest_engine import HistoricalBacktester


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
