"""Tests for ACB tracking (Adjusted Cost Base) for Canadian TFSA portfolios.

Tax-Loss Harvesting (TLH) is categorically prohibited in TFSA accounts
per Canadian treaty guidelines and is not tested.

Foreign withholding tax on US dividends (e.g., VFV, QQC) is modeled
as a strict yield reduction — see :mod:`tests.test_tax_location` for FWT
drag verification.
"""

from __future__ import annotations

from datetime import date

import pytest

from pysharpe.execution.tax_tracker import (
    ACBPosition,
    ACBTracker,
    TradeRecord,
)

# ---------------------------------------------------------------------------
# ACBTracker — initialization
# ---------------------------------------------------------------------------


class TestACBTrackerInit:
    def test_empty_initialization(self):
        tracker = ACBTracker()
        assert tracker.positions == {}
        assert tracker.trades == []

    def test_initial_positions_valid(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        assert len(tracker.positions) == 1
        pos = tracker.get_position("VCN.TO")
        assert pos.total_shares == 100.0
        assert pos.total_cost == 3500.0
        assert pos.acb_per_share == 35.0

    def test_initial_positions_negative_shares_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ACBTracker(initial_positions={"VCN.TO": (-10.0, 1000.0)})

    def test_initial_positions_negative_cost_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ACBTracker(initial_positions={"VCN.TO": (100.0, -100.0)})

    def test_initial_positions_shares_no_cost_raises(self):
        with pytest.raises(ValueError, match="cost.*positive"):
            ACBTracker(initial_positions={"VCN.TO": (100.0, 0.0)})

    def test_initial_positions_zero_shares_with_cost_raises(self):
        with pytest.raises(ValueError, match="cost.*zero"):
            ACBTracker(initial_positions={"VCN.TO": (0.0, 500.0)})

    def test_multiple_initial_positions(self):
        tracker = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 3500.0),
                "VUN.TO": (50.0, 2500.0),
            }
        )
        assert len(tracker.positions) == 2
        assert tracker.get_acb_per_share("VCN.TO") == 35.0
        assert tracker.get_acb_per_share("VUN.TO") == 50.0


# ---------------------------------------------------------------------------
# ACBTracker — buys
# ---------------------------------------------------------------------------


class TestACBTrackerBuys:
    def test_single_buy(self):
        tracker = ACBTracker()
        pos = tracker.record_buy("VCN.TO", 10.0, 50.0, date(2024, 1, 15))
        assert pos.total_shares == 10.0
        assert pos.total_cost == 500.0
        assert pos.acb_per_share == 50.0
        assert len(tracker.trades) == 1

    def test_multiple_buys_weighted_average(self):
        """ACB per share should be the weighted average across all purchases."""
        tracker = ACBTracker()
        tracker.record_buy("VCN.TO", 10.0, 50.0, date(2024, 1, 15))
        tracker.record_buy("VCN.TO", 20.0, 40.0, date(2024, 2, 15))

        pos = tracker.get_position("VCN.TO")
        assert pos.total_shares == 30.0
        # Total cost = 10*50 + 20*40 = 500 + 800 = 1300
        assert pos.total_cost == 1300.0
        # ACB/sh = 1300/30 = 43.333...
        assert pos.acb_per_share == pytest.approx(43.333333, rel=1e-6)

    def test_buy_negative_shares_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VCN.TO", -5.0, 50.0, date(2024, 1, 15))

    def test_buy_zero_shares_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VCN.TO", 0.0, 50.0, date(2024, 1, 15))

    def test_buy_negative_price_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VCN.TO", 10.0, -5.0, date(2024, 1, 15))

    def test_buy_zero_price_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VCN.TO", 10.0, 0.0, date(2024, 1, 15))

    def test_buy_adds_to_existing_position(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (10.0, 500.0)})
        tracker.record_buy("VCN.TO", 10.0, 40.0, date(2024, 2, 1))
        pos = tracker.get_position("VCN.TO")
        assert pos.total_shares == 20.0
        assert pos.total_cost == 900.0
        assert pos.acb_per_share == 45.0

    def test_trade_record_fields(self):
        tracker = ACBTracker()
        tracker.record_buy("VCN.TO", 10.0, 50.0, date(2024, 1, 15))
        trade = tracker.trades[0]
        assert trade.ticker == "VCN.TO"
        assert trade.date == date(2024, 1, 15)
        assert trade.action == "BUY"
        assert trade.shares == 10.0
        assert trade.price_per_share == 50.0
        assert trade.total_amount == 500.0


# ---------------------------------------------------------------------------
# ACBTracker — sells
# ---------------------------------------------------------------------------


class TestACBTrackerSells:
    def test_partial_sell_realizes_gain(self):
        """Selling part of a position at a price above ACB should produce a gain."""
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 3500.0)}  # ACB/sh = 35.0
        )
        gain = tracker.record_sell("VCN.TO", 50.0, 45.0, date(2024, 3, 1))
        # ACB of sold shares = 50 * 35 = 1750; proceeds = 50 * 45 = 2250
        # gain = 2250 - 1750 = 500
        assert gain == 500.0
        pos = tracker.get_position("VCN.TO")
        assert pos.total_shares == 50.0
        assert pos.total_cost == pytest.approx(1750.0)
        assert pos.acb_per_share == pytest.approx(35.0)

    def test_partial_sell_realizes_loss(self):
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 3500.0)}  # ACB/sh = 35.0
        )
        loss = tracker.record_sell("VCN.TO", 50.0, 30.0, date(2024, 3, 1))
        # ACB of sold = 50 * 35 = 1750; proceeds = 1500
        # loss = 1500 - 1750 = -250
        assert loss == pytest.approx(-250.0)
        pos = tracker.get_position("VCN.TO")
        assert pos.total_shares == 50.0
        assert pos.acb_per_share == pytest.approx(35.0)

    def test_full_sell(self):
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (10.0, 500.0)}  # ACB/sh = 50
        )
        gain = tracker.record_sell("VCN.TO", 10.0, 60.0, date(2024, 3, 1))
        assert gain == 100.0
        pos = tracker.get_position("VCN.TO")
        assert pos.total_shares == 0.0
        assert pos.total_cost == 0.0
        assert pos.acb_per_share == 0.0

    def test_sell_more_than_held_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (10.0, 500.0)})
        with pytest.raises(ValueError, match="only 10.0 held"):
            tracker.record_sell("VCN.TO", 15.0, 60.0, date(2024, 3, 1))

    def test_sell_negative_shares_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (10.0, 500.0)})
        with pytest.raises(ValueError, match="positive"):
            tracker.record_sell("VCN.TO", -5.0, 60.0, date(2024, 3, 1))

    def test_sell_zero_shares_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (10.0, 500.0)})
        with pytest.raises(ValueError, match="positive"):
            tracker.record_sell("VCN.TO", 0.0, 60.0, date(2024, 3, 1))

    def test_sell_negative_price_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (10.0, 500.0)})
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record_sell("VCN.TO", 5.0, -10.0, date(2024, 3, 1))

    def test_sell_empty_position_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="no shares or zero cost basis"):
            tracker.record_sell("VCN.TO", 10.0, 50.0, date(2024, 3, 1))

    def test_sell_records_trade(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        tracker.record_sell("VCN.TO", 20.0, 40.0, date(2024, 3, 1))
        trade = tracker.trades[0]
        assert trade.action == "SELL"
        assert trade.shares == 20.0
        assert trade.price_per_share == 40.0
        assert trade.total_amount == 800.0


# ---------------------------------------------------------------------------
# ACBTracker — Return of Capital
# ---------------------------------------------------------------------------


class TestACBTrackerROC:
    def test_roc_reduces_acb(self):
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 3500.0)}  # ACB/sh = 35.0
        )
        pos, gain = tracker.record_roc("VCN.TO", 100.0, date(2024, 6, 15))
        assert pos.total_cost == 3400.0
        assert pos.total_shares == 100.0
        assert pos.acb_per_share == 34.0
        assert gain == 0.0
        assert len(tracker.trades) == 1

    def test_roc_excess_becomes_capital_gain(self):
        """ROC exceeding ACB resets cost to zero and realizes a capital gain."""
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 50.0)})
        pos, gain = tracker.record_roc("VCN.TO", 200.0, date(2024, 6, 15))
        assert pos.total_cost == 0.0
        assert pos.acb_per_share == 0.0
        assert gain == 150.0  # 200 ROC - 50 ACB = 150 realized gain

    def test_roc_on_empty_position_logs_warning(self, caplog):
        import logging

        tracker = ACBTracker()
        with caplog.at_level(logging.WARNING):
            pos, gain = tracker.record_roc("VCN.TO", 100.0, date(2024, 6, 15))
        assert "no shares are held" in caplog.text
        assert pos.total_shares == 0.0
        assert pos.total_cost == 0.0
        assert gain == 0.0

    def test_roc_negative_amount_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        with pytest.raises(ValueError, match="positive"):
            tracker.record_roc("VCN.TO", -50.0, date(2024, 6, 15))

    def test_roc_zero_amount_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        with pytest.raises(ValueError, match="positive"):
            tracker.record_roc("VCN.TO", 0.0, date(2024, 6, 15))

    def test_roc_trade_record(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        pos, gain = tracker.record_roc("VCN.TO", 100.0, date(2024, 6, 15))
        assert gain == 0.0
        trade = tracker.trades[0]
        assert trade.action == "ROC"
        assert trade.shares == 0.0
        assert trade.price_per_share == 0.0
        assert trade.total_amount == 100.0
        assert trade.realized_gain_loss == 0.0

    def test_roc_trade_record_with_excess_gain(self):
        """TradeRecord captures the realized gain when ROC exceeds ACB."""
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 50.0)})
        pos, gain = tracker.record_roc("VCN.TO", 200.0, date(2024, 6, 15))
        assert gain == 150.0
        trade = tracker.trades[0]
        assert trade.action == "ROC"
        assert trade.shares == 0.0
        assert trade.price_per_share == 0.0
        assert trade.total_amount == 200.0
        assert trade.realized_gain_loss == 150.0


# ---------------------------------------------------------------------------
# ACBTracker — unrealized gain/loss
# ---------------------------------------------------------------------------


class TestACBTrackerUnrealizedGL:
    def test_unrealized_gain(self):
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 3500.0)}  # ACB/sh = 35
        )
        gl = tracker.get_unrealized_gain_loss("VCN.TO", 45.0)
        # Market value = 100 * 45 = 4500; ACB = 3500 → gain of 1000
        assert gl == 1000.0

    def test_unrealized_loss(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        gl = tracker.get_unrealized_gain_loss("VCN.TO", 30.0)
        # Market = 3000; ACB = 3500 → loss of 500
        assert gl == -500.0

    def test_unrealized_no_shares(self):
        tracker = ACBTracker()
        gl = tracker.get_unrealized_gain_loss("VCN.TO", 50.0)
        assert gl == 0.0

    def test_unrealized_negative_price_raises(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        with pytest.raises(ValueError, match="non-negative"):
            tracker.get_unrealized_gain_loss("VCN.TO", -10.0)

    def test_unrealized_gain_loss_pct(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        pct = tracker.get_unrealized_gain_loss_pct("VCN.TO", 45.0)
        assert pct == pytest.approx(1000.0 / 3500.0)

    def test_unrealized_pct_no_cost(self):
        tracker = ACBTracker()
        pct = tracker.get_unrealized_gain_loss_pct("VCN.TO", 50.0)
        assert pct == 0.0


# ---------------------------------------------------------------------------
# ACBTracker — getters and positions property
# ---------------------------------------------------------------------------


class TestACBTrackerGetters:
    def test_get_position_creates_if_missing(self):
        tracker = ACBTracker()
        pos = tracker.get_position("VCN.TO")
        assert pos.ticker == "VCN.TO"
        assert pos.total_shares == 0.0
        assert pos.total_cost == 0.0
        # Should now be in positions
        assert "VCN.TO" in tracker.positions

    def test_get_acb(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        assert tracker.get_acb("VCN.TO") == 3500.0

    def test_get_acb_per_share(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        assert tracker.get_acb_per_share("VCN.TO") == 35.0

    def test_get_shares(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        assert tracker.get_shares("VCN.TO") == 100.0

    def test_positions_property_returns_copy(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        positions = tracker.positions
        positions["NEW"] = ACBPosition(ticker="NEW")
        assert "NEW" not in tracker.positions

    def test_trades_property_returns_copy(self):
        tracker = ACBTracker()
        tracker.record_buy("VCN.TO", 10.0, 50.0, date(2024, 1, 1))
        trades = tracker.trades
        trades.append(
            TradeRecord(
                ticker="FAKE",
                date=date(2024, 1, 1),
                action="BUY",
                shares=1.0,
                price_per_share=1.0,
                total_amount=1.0,
            )
        )
        assert len(tracker.trades) == 1


# ---------------------------------------------------------------------------
# ACBTracker — summary
# ---------------------------------------------------------------------------


class TestACBTrackerSummary:
    def test_summary_empty(self):
        tracker = ACBTracker()
        df = tracker.summary()
        assert df.empty

    def test_summary_without_prices(self):
        tracker = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 3500.0),
                "VUN.TO": (50.0, 2500.0),
            }
        )
        df = tracker.summary()
        assert len(df) == 2
        assert set(df.columns) == {
            "ticker",
            "total_shares",
            "total_cost",
            "acb_per_share",
        }
        assert df.loc[df["ticker"] == "VCN.TO", "acb_per_share"].iloc[0] == 35.0

    def test_summary_with_prices(self):
        tracker = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 3500.0),  # ACB/sh = 35
                "VUN.TO": (50.0, 2500.0),  # ACB/sh = 50
            }
        )
        df = tracker.summary(current_prices={"VCN.TO": 40.0, "VUN.TO": 45.0})
        assert "market_value" in df.columns
        assert "unrealized_gain_loss" in df.columns

        vcn = df[df["ticker"] == "VCN.TO"].iloc[0]
        assert vcn["market_value"] == 4000.0
        assert vcn["unrealized_gain_loss"] == 500.0

        vun = df[df["ticker"] == "VUN.TO"].iloc[0]
        assert vun["market_value"] == 2250.0
        assert vun["unrealized_gain_loss"] == -250.0

    def test_summary_missing_price(self):
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        df = tracker.summary(current_prices={"OTHER": 50.0})
        assert "market_value" not in df.columns


# ---------------------------------------------------------------------------
# ACBTracker — fractional share handling
# ---------------------------------------------------------------------------


class TestACBTrackerFractional:
    def test_fractional_buy_and_sell(self):
        tracker = ACBTracker()
        tracker.record_buy("VCN.TO", 1.3333, 50.0, date(2024, 1, 15))
        assert tracker.get_shares("VCN.TO") == pytest.approx(1.3333)
        assert tracker.get_acb("VCN.TO") == pytest.approx(66.665)

        gain = tracker.record_sell("VCN.TO", 0.3333, 60.0, date(2024, 2, 1))
        expected_acb_sold = 0.3333 * 50.0
        expected_proceeds = 0.3333 * 60.0
        assert gain == pytest.approx(expected_proceeds - expected_acb_sold)
        assert tracker.get_shares("VCN.TO") == pytest.approx(1.0)

    def test_complete_round_trip(self):
        """Buy, sell all, then buy again. ACB should reflect only the new pool."""
        tracker = ACBTracker()
        tracker.record_buy("VCN.TO", 10.0, 50.0, date(2024, 1, 1))
        tracker.record_sell("VCN.TO", 10.0, 60.0, date(2024, 2, 1))
        assert tracker.get_shares("VCN.TO") == 0.0
        assert tracker.get_acb("VCN.TO") == 0.0

        # Buy again — new cost pool
        tracker.record_buy("VCN.TO", 10.0, 70.0, date(2024, 3, 1))
        assert tracker.get_acb_per_share("VCN.TO") == 70.0

    def test_floating_point_cleanup_on_full_sale(self):
        """After a full sale, residual floating-point noise should be zeroed."""
        tracker = ACBTracker(initial_positions={"VCN.TO": (1e-13, 1e-13)})
        # Force sell more than held to trigger edge case (but shares > 1e-13
        # isn't > 1e-12 threshold). Actually let's test the normal path.
        tracker.record_buy("VCN.TO", 10.0, 35.123456789, date(2024, 1, 1))
        tracker.record_sell("VCN.TO", 10.0, 40.0, date(2024, 2, 1))
        assert tracker.get_shares("VCN.TO") == 0.0
        assert tracker.get_acb("VCN.TO") == 0.0


# ---------------------------------------------------------------------------
# ACBTracker — edge-case sequences
# ---------------------------------------------------------------------------


class TestACBTrackerEdgeCases:
    def test_zero_cost_after_roc_then_buy(self):
        """After ROC drives cost to zero, new buys should restart pooling."""
        tracker = ACBTracker(initial_positions={"VCN.TO": (10.0, 500.0)})
        pos, gain = tracker.record_roc("VCN.TO", 600.0, date(2024, 1, 15))
        assert tracker.get_acb("VCN.TO") == 0.0
        assert gain == 100.0  # 600 ROC - 500 ACB = 100 realized gain

        tracker.record_buy("VCN.TO", 10.0, 40.0, date(2024, 2, 1))
        assert tracker.get_acb_per_share("VCN.TO") == pytest.approx(
            (0.0 + 400.0) / 20.0
        )
        assert tracker.get_shares("VCN.TO") == 20.0

    def test_many_small_buys(self):
        """Weighted-average should handle many small purchases correctly."""
        tracker = ACBTracker()
        for i in range(1, 101):
            tracker.record_buy("VCN.TO", 0.1, float(i), date(2024, 1, 1))
        # 100 buys of 0.1 shares at prices 1..100
        total_shares = 10.0
        total_cost = sum(0.1 * i for i in range(1, 101))
        expected_acb = total_cost / total_shares
        assert tracker.get_shares("VCN.TO") == pytest.approx(total_shares)
        assert tracker.get_acb("VCN.TO") == pytest.approx(total_cost)
        assert tracker.get_acb_per_share("VCN.TO") == pytest.approx(expected_acb)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_trade_record_fields(self):
        tr = TradeRecord(
            ticker="VCN.TO",
            date=date(2024, 1, 15),
            action="BUY",
            shares=10.0,
            price_per_share=35.0,
            total_amount=350.0,
        )
        assert tr.ticker == "VCN.TO"
        assert tr.total_amount == 350.0

    def test_acb_position_attributes(self):
        pos = ACBPosition(ticker="VCN.TO", total_shares=100.0, total_cost=3500.0)
        assert pos.acb_per_share == 35.0

    def test_acb_position_zero_shares(self):
        pos = ACBPosition(ticker="VCN.TO", total_shares=0.0, total_cost=0.0)
        assert pos.acb_per_share == 0.0
