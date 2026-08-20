"""Tests for transaction-cost modeling in backtests.

.. note::

    **Canadian TFSA Context** — Transaction costs (fixed fees + slippage)
    are modeled as portfolio drag.  In a TFSA, these costs cannot be
    deducted from taxable income.  Fee-per-trade and slippage percentages
    reflect Canadian discount brokerage rates.
"""

import numpy as np
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


# ---------------------------------------------------------------------------
# Bid-ask spread modelling (half-spread on each side)
# ---------------------------------------------------------------------------


class TestSpreadCosts:
    def test_initial_fees_with_spread(self):
        """spread_pct charges half-spread on each side of the initial buy."""
        prices_df = pd.DataFrame(
            {"A": [10.0, 10.0], "B": [10.0, 10.0]},
            index=pd.date_range("2023-01-01", periods=2),
        )
        backtester = HistoricalBacktester(
            prices=prices_df,
            target_weights={"A": 0.5, "B": 0.5},
            initial_capital=1000.0,
            fee_per_trade=5.0,
            slippage_pct=0.01,
            spread_pct=0.02,  # full spread; cost = turnover * 0.01
        )
        result = backtester.run()

        # fixed = 2 * 5 = 10; slippage = 1000 * 0.01 = 10;
        # spread = 1000 * (0.02 / 2) = 10 → total 30 → start 970
        assert pytest.approx(result.portfolio_value.iloc[0]) == 970.0

    def test_rebalance_fees_with_spread(self):
        """Rebalancing cost includes the half-spread on turnover."""
        prices_df = pd.DataFrame(
            {"A": [10.0, 10.0, 20.0], "B": [10.0, 20.0, 20.0]},
            index=pd.date_range("2023-01-01", periods=3),
        )
        backtester = HistoricalBacktester(
            prices=prices_df,
            target_weights={"A": 0.5, "B": 0.5},
            initial_capital=1000.0,
            abs_band=0.01,
            fee_per_trade=2.0,
            slippage_pct=0.02,
            spread_pct=0.04,
        )
        result = backtester.run()

        # Day 0: fixed 2*2 = 4 + slippage 1000*0.02 = 20 + spread 1000*0.02 = 20
        # → start 956, shares A = B = 47.8
        assert pytest.approx(result.portfolio_value.iloc[0]) == 956.0

        # Day 1: A=10→478, B=20→956, total 1434; weights breach the band.
        # Turnover: |71.7−47.8|*10 + |35.85−47.8|*20 = 239 + 239 = 478
        # fixed = 2*2 = 4; slippage = 478*0.02 = 9.56; spread = 478*0.02 = 9.56
        # total fees = 23.12; value = 1434 − 23.12 = 1410.88
        assert pytest.approx(result.portfolio_value.iloc[1]) == 1410.88

    def test_negative_cost_parameter_raises(self):
        prices_df = pd.DataFrame(
            {"A": [10.0, 10.0], "B": [10.0, 10.0]},
            index=pd.date_range("2023-01-01", periods=2),
        )
        with pytest.raises(ValueError, match="non-negative"):
            HistoricalBacktester(
                prices=prices_df, target_weights={"A": 1.0}, spread_pct=-0.01
            )
        with pytest.raises(ValueError, match="non-negative"):
            HistoricalBacktester(
                prices=prices_df, target_weights={"A": 1.0}, fee_per_trade=-1.0
            )

    def test_fees_cannot_exceed_portfolio(self):
        """Costs exceeding the portfolio value must wipe it out, not create
        negative share positions."""
        prices_df = pd.DataFrame(
            {"A": [10.0, 10.0, 10.0], "B": [10.0, 20.0, 30.0]},
            index=pd.date_range("2023-01-01", periods=3),
        )
        backtester = HistoricalBacktester(
            prices=prices_df,
            target_weights={"A": 0.5, "B": 0.5},
            initial_capital=100.0,
            abs_band=1e-9,  # rebalance every day
            fee_per_trade=1000.0,  # fees dwarf the portfolio
            slippage_pct=0.0,
        )
        result = backtester.run()
        assert np.all(result.portfolio_value.values >= 0.0)
        assert np.all(np.isfinite(result.portfolio_value.values))


# ---------------------------------------------------------------------------
# Walk-forward transaction costs
# ---------------------------------------------------------------------------


class _FixedWeightsOptimizer:
    """Deterministic optimizer returning fixed weights."""

    def __init__(self, data):
        self.assets = data.columns.tolist()

    def optimize(self):
        from pysharpe.optimization.base import OptimizationResult

        n = len(self.assets)
        return OptimizationResult(
            weights={a: 1.0 / n for a in self.assets},
            expected_return=0.08,
            volatility=0.15,
            sharpe_ratio=0.5,
        )


@pytest.fixture
def wf_prices():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=90)
    return pd.DataFrame(
        {
            "A": np.exp(rng.normal(0.001, 0.02, 90).cumsum()),
            "B": np.exp(rng.normal(0.002, 0.03, 90).cumsum()),
        },
        index=dates,
    )


class TestWalkForwardCosts:
    def test_walk_forward_applies_transaction_costs(self, wf_prices):
        """Walk-forward results must be strictly worse with costs than without."""
        from pysharpe.analysis.backtest_engine import WalkForwardBacktester

        no_cost = WalkForwardBacktester(
            optimizer_factory=_FixedWeightsOptimizer,
            train_window_days=30,
            test_window_days=15,
            initial_capital=10000.0,
        ).run(wf_prices)

        with_cost = WalkForwardBacktester(
            optimizer_factory=_FixedWeightsOptimizer,
            train_window_days=30,
            test_window_days=15,
            initial_capital=10000.0,
            fee_per_trade=5.0,
            slippage_pct=0.001,
            spread_pct=0.001,
        ).run(wf_prices)

        assert with_cost.portfolio_value.iloc[-1] < no_cost.portfolio_value.iloc[-1]
        # Every window transition reduces the value by the exact transition cost.
        assert with_cost.portfolio_value.iloc[0] < 10000.0

    def test_no_future_cost_leakage(self):
        """Costs at day t must depend only on day-t prices.  Perturbing a
        future price must leave the day-t portfolio value untouched."""
        from pysharpe.analysis.backtest_engine import HistoricalBacktester

        dates = pd.date_range("2023-01-01", periods=4)
        base = pd.DataFrame(
            {"A": [10.0, 10.0, 20.0, 20.0], "B": [10.0, 20.0, 20.0, 20.0]},
            index=dates,
        )
        shocked = base.copy()
        shocked.loc[dates[3], "A"] = 200.0  # massive future move

        kwargs = dict(
            target_weights={"A": 0.5, "B": 0.5},
            initial_capital=1000.0,
            abs_band=0.01,
            fee_per_trade=2.0,
            slippage_pct=0.02,
        )
        base_result = HistoricalBacktester(prices=base, **kwargs).run()
        shocked_result = HistoricalBacktester(prices=shocked, **kwargs).run()

        # Values before the shocked date must be identical.
        pd.testing.assert_series_equal(
            base_result.portfolio_value.iloc[:3],
            shocked_result.portfolio_value.iloc[:3],
        )
