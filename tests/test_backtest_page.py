"""Tests for the backtesting tab and its helper functions.

.. note::

    **Canadian ETF Context** — Backtests use synthetic CAD-denominated price
    series and standard Canadian rebalancing frequencies (monthly/ME).
    Historical backtesting validates drift-based and calendar rebalancing
    strategies for broad-market ETF portfolios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe import metrics
from pysharpe.analysis.backtest_engine import BacktestResult, HistoricalBacktester


def _monotone_prices(n=60):
    """Monotonically increasing prices — deterministic, positive returns."""
    dates = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {"A": 100.0 + np.arange(n) * 0.5, "B": 200.0 + np.arange(n) * 0.3},
        index=dates,
    )


def test_backtest_result_shape():
    prices = _monotone_prices()
    bt = HistoricalBacktester(prices, {"A": 0.6, "B": 0.4}, initial_capital=1000.0)
    result = bt.run()
    assert isinstance(result, BacktestResult)
    assert len(result.portfolio_value) == len(prices)
    assert set(result.historical_weights.columns) == {"A", "B"}
    assert result.portfolio_value.iloc[0] == pytest.approx(1000.0)


def test_cagr_positive_on_growing_portfolio():
    prices = _monotone_prices()
    bt = HistoricalBacktester(prices, {"A": 0.6, "B": 0.4}, initial_capital=1000.0)
    result = bt.run()
    assert metrics.cagr(result.portfolio_value) > 0


def test_maximum_drawdown_zero_on_monotone():
    prices = _monotone_prices()
    bt = HistoricalBacktester(prices, {"A": 0.6, "B": 0.4}, initial_capital=1000.0)
    result = bt.run()
    mdd = metrics.maximum_drawdown(result.portfolio_value)
    assert mdd >= -0.01  # effectively zero on monotone series


def test_monthly_rebalance_produces_rebalance_events():
    prices = _monotone_prices(n=120)
    bt = HistoricalBacktester(
        prices, {"A": 0.6, "B": 0.4}, initial_capital=1000.0, rebalance_freq="ME"
    )
    result = bt.run()
    assert len(result.rebalance_events) > 1


def test_parse_manual_weights_valid():
    """Test the weight-parsing helper that the UI uses."""
    from pysharpe.app.backtest import parse_weight_input

    weights = parse_weight_input("A=0.6, B=0.4")
    assert weights == pytest.approx({"A": 0.6, "B": 0.4})


def test_parse_manual_weights_normalised():
    from pysharpe.app.backtest import parse_weight_input

    weights = parse_weight_input("A=3, B=1")
    assert weights["A"] == pytest.approx(0.75)
    assert weights["B"] == pytest.approx(0.25)


def test_parse_manual_weights_invalid_raises():
    from pysharpe.app.backtest import parse_weight_input

    with pytest.raises(ValueError):
        parse_weight_input("not valid input")
