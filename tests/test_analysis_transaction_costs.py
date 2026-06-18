import pandas as pd
import pytest

from pysharpe.analysis.backtest_engine import HistoricalBacktester


@pytest.fixture
def prices():
    dates = pd.date_range("2023-01-01", periods=3)
    # Simple prices: day 0: $10, day 1: $10 (drift), day 2: $20
    data = pd.DataFrame({"A": [10.0, 10.0, 20.0], "B": [10.0, 20.0, 20.0]}, index=dates)
    return data


def test_initial_fees(prices):
    targets = {"A": 0.5, "B": 0.5}
    initial_capital = 1000.0

    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=targets,
        initial_capital=initial_capital,
        fee_per_trade=5.0,  # $5 fixed per asset traded
        slippage_pct=0.01,  # 1% variable
    )
    result = backtester.run()

    # Expected initial fees:
    # 2 assets traded > 0
    # fixed = 2 * 5.0 = 10.0
    # slippage = 1000.0 * 0.01 = 10.0
    # total fees = 20.0
    # starting value = 980.0
    assert pytest.approx(result.portfolio_value.iloc[0]) == 980.0


def test_rebalance_fees(prices):
    targets = {"A": 0.5, "B": 0.5}
    initial_capital = 1000.0

    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=targets,
        initial_capital=initial_capital,
        abs_band=0.01,  # Force rebalance easily
        fee_per_trade=2.0,
        slippage_pct=0.02,
    )
    result = backtester.run()

    # Day 0:
    # fixed: 2 * 2 = 4
    # slip: 1000 * 0.02 = 20
    # start value: 976.0
    # shares: A=488/10=48.8, B=488/10=48.8
    assert pytest.approx(result.portfolio_value.iloc[0]) == 976.0

    # Day 1:
    # A prices=10, B prices=20
    # value A = 488, value B = 976 -> total = 1464
    # Weights: A=33.3%, B=66.7%
    # This breaches abs_band (0.01).
    # Ideal target A: 1464 * 0.5 = 732 -> 73.2 shares
    # Ideal target B: 1464 * 0.5 = 732 -> 36.6 shares
    # Shares traded A = abs(73.2 - 48.8) = 24.4 -> $244
    # Shares traded B = abs(36.6 - 48.8) = 12.2 -> $244
    # dollars traded = 488
    # fixed: 2 trades * 2.0 = 4.0
    # slip: 488 * 0.02 = 9.76
    # total fees = 13.76
    # portfolio value day 1 = 1464 - 13.76 = 1450.24

    assert pytest.approx(result.portfolio_value.iloc[1]) == 1450.24
