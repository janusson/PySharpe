"""Tests for ACB tracking and Tax-Loss Harvesting engine.

.. warning::

    **Canadian TFSA Constraint** — PySharpe operates under the assumption
    that all portfolios are held within a Canadian Tax-Free Savings Account
    (TFSA).  Capital gains are tax-exempt and losses cannot be claimed.

    **Tax-Loss Harvesting (TLH) is categorically prohibited** in TFSA,
    FHSA, and RRSP accounts per Canadian treaty guidelines.  The TLH
    tests in this file (:class:`TestTLHEngineIdentify`,
    :class:`TestTLHEngineGeneratePairs`, :class:`TestTLHEngineSuperficialLoss`,
    :class:`TestEndToEndTLHScenario`, and related) exercise the deprecated
    TLH engine for non-registered account validation only.

    The :class:`TestTLHAccountGuardrails` class verifies that the TLH entry
    points correctly reject TFSA, FHSA, and RRSP account types.

    **ACB tracking** (:class:`ACBTracker`, :class:`TradeRecord`,
    :class:`ACBPosition`) is legitimate in any account type for cost-basis
    bookkeeping and is retained in the test suite.

    **Foreign withholding tax** on US dividends (e.g., VFV, QQC) is modeled
    as a strict yield reduction — see :mod:`tests.test_tax_location` for FWT
    drag verification.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pysharpe.execution.tax_tracker import (
    ACBPosition,
    ACBTracker,
    TLHEngine,
    TLHRebalanceResult,
    TLHTrade,
    TradeRecord,
    _load_switch_fund_map,
    _settlement_date,
    _superficial_window,
    analyze_tlh_opportunities,
    format_tlh_rebalance_result,
)
from pysharpe.optimization.tax_location import AccountType

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
# Helper functions
# ---------------------------------------------------------------------------


class TestSettlementDate:
    def test_settlement_date_t1(self):
        assert _settlement_date(date(2024, 6, 3)) == date(2024, 6, 4)

    def test_settlement_date_month_end(self):
        assert _settlement_date(date(2024, 1, 31)) == date(2024, 2, 1)


class TestSuperficialWindow:
    def test_window_bounds(self):
        start, end = _superficial_window(date(2024, 6, 15))
        assert start == date(2024, 5, 16)
        assert end == date(2024, 7, 15)

    def test_window_around_year_end(self):
        start, end = _superficial_window(date(2024, 1, 2))
        assert start == date(2023, 12, 3)
        assert end == date(2024, 2, 1)


class TestLoadSwitchFundMap:
    def test_loads_valid_json(self, tmp_path):
        path = tmp_path / "switch_fund_map.json"
        path.write_text(
            '{"VCN.TO": ["XIC.TO", "ZCN.TO"], "XIC.TO": ["VCN.TO"]}',
            encoding="utf-8",
        )
        result = _load_switch_fund_map(path)
        assert result == {"VCN.TO": ["XIC.TO", "ZCN.TO"], "XIC.TO": ["VCN.TO"]}

    def test_missing_file_returns_empty(self):
        result = _load_switch_fund_map(Path("/nonexistent/path.json"))
        assert result == {}

    def test_invalid_json_returns_empty(self, tmp_path, caplog):
        import logging

        path = tmp_path / "bad.json"
        path.write_text("{not valid json}", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            result = _load_switch_fund_map(path)
        assert result == {}
        assert "Failed to load" in caplog.text

    def test_non_object_json_returns_empty(self, tmp_path, caplog):
        import logging

        path = tmp_path / "array.json"
        path.write_text('[{"key": "val"}]', encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            result = _load_switch_fund_map(path)
        assert result == {}
        assert "JSON object" in caplog.text

    def test_skips_invalid_entries(self, tmp_path, caplog):
        import logging

        path = tmp_path / "mixed.json"
        path.write_text(
            '{"VCN.TO": ["XIC.TO"], "bad_key": "not_a_list", "int_key": 42}',
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            result = _load_switch_fund_map(path)
        # "VCN.TO" -> ["XIC.TO"] is valid; the others have non-list values
        # and are skipped. Only "VCN.TO" survives validation.
        assert "VCN.TO" in result
        assert "bad_key" not in result
        assert "int_key" not in result


# ---------------------------------------------------------------------------
# TLHEngine — initialization
# ---------------------------------------------------------------------------


class TestTLHEngineInit:
    def test_init_with_dict(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 3500.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 30.0},
        )
        assert engine.switch_map == {"VCN.TO": ["XIC.TO"]}
        assert engine.prices == {"VCN.TO": 30.0}

    def test_init_with_path(self, tmp_path):
        path = tmp_path / "switch.json"
        path.write_text('{"VCN.TO": ["XIC.TO"]}', encoding="utf-8")
        acb = ACBTracker()
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map=path,
            current_prices={"VCN.TO": 30.0},
        )
        assert engine.switch_map == {"VCN.TO": ["XIC.TO"]}

    def test_init_with_none(self):
        acb = ACBTracker()
        engine = TLHEngine(acb_tracker=acb, switch_fund_map=None)
        assert engine.switch_map == {}

    def test_init_with_empty_prices(self):
        acb = ACBTracker()
        engine = TLHEngine(acb_tracker=acb)
        assert engine.prices == {}


# ---------------------------------------------------------------------------
# TLHEngine — identify_tlh_opportunities
# ---------------------------------------------------------------------------


class TestTLHEngineIdentify:
    def test_identifies_loss_positions(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),  # ACB/sh = 40
                "VUN.TO": (50.0, 3000.0),  # ACB/sh = 60
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 30.0, "VUN.TO": 65.0},
        )
        df = engine.identify_tlh_opportunities()

        # VCN: loss of (3000 - 4000) = -1000 → eligible
        # VUN: gain of (3250 - 3000) = 250 → NOT eligible
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "VCN.TO"
        assert df.iloc[0]["unrealized_gain_loss"] == -1000.0
        assert df.iloc[0]["loss_pct"] == 0.25
        assert df.iloc[0]["has_switch_fund"]

    def test_no_losses_returns_empty(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 3000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 40.0},  # gain
        )
        df = engine.identify_tlh_opportunities()
        assert df.empty

    def test_min_loss_dollars_filter(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),
                "VUN.TO": (50.0, 3000.0),
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 39.0, "VUN.TO": 59.0},
        )
        # VCN loss = 3900 - 4000 = -100; VUN loss = 2950 - 3000 = -50
        df = engine.identify_tlh_opportunities(min_loss_dollars=75.0)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "VCN.TO"

    def test_min_loss_pct_filter(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),  # ACB/sh 40, price 39 → 2.5% loss
                "VUN.TO": (50.0, 3000.0),  # ACB/sh 60, price 54 → 10% loss
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 39.0, "VUN.TO": 54.0},
        )
        df = engine.identify_tlh_opportunities(min_loss_pct=0.05)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "VUN.TO"

    def test_exclude_tickers(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),
                "VUN.TO": (50.0, 3000.0),
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0, "VUN.TO": 50.0},
        )
        df = engine.identify_tlh_opportunities(exclude_tickers=["VUN.TO"])
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "VCN.TO"

    def test_skips_tickers_without_price(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        engine = TLHEngine(acb_tracker=acb, current_prices={})
        df = engine.identify_tlh_opportunities()
        assert df.empty

    def test_skips_tickers_with_zero_shares(self):
        acb = ACBTracker()
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        df = engine.identify_tlh_opportunities()
        assert df.empty

    def test_sorted_by_worst_loss_first(self):
        acb = ACBTracker(
            initial_positions={
                "A": (100.0, 5000.0),  # price 40 → loss 1000
                "B": (100.0, 5000.0),  # price 30 → loss 2000
                "C": (100.0, 5000.0),  # price 45 → loss 500
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"A": 40.0, "B": 30.0, "C": 45.0},
        )
        df = engine.identify_tlh_opportunities()
        assert list(df["ticker"]) == ["B", "A", "C"]


# ---------------------------------------------------------------------------
# TLHEngine — generate_tlh_pairs
# ---------------------------------------------------------------------------


class TestTLHEngineGeneratePairs:
    def test_generates_pairs_for_each_switch_fund(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO", "ZCN.TO"]},
            current_prices={"VCN.TO": 30.0},
        )
        proposals = engine.generate_tlh_pairs("VCN.TO")
        assert len(proposals) == 2
        buy_tickers = {p.buy_ticker for p in proposals}
        assert buy_tickers == {"XIC.TO", "ZCN.TO"}

        for p in proposals:
            assert p.sell_ticker == "VCN.TO"
            assert p.sell_shares == 100.0
            assert p.sell_proceeds == 3000.0
            assert p.unrealized_loss == -1000.0

    def test_no_loss_returns_empty(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 3000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 40.0},  # gain
        )
        assert engine.generate_tlh_pairs("VCN.TO") == []

    def test_no_switch_funds_returns_empty(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        assert engine.generate_tlh_pairs("VCN.TO") == []

    def test_no_price_returns_empty(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={},
        )
        assert engine.generate_tlh_pairs("VCN.TO") == []

    def test_no_shares_returns_empty(self):
        acb = ACBTracker()
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 30.0},
        )
        assert engine.generate_tlh_pairs("VCN.TO") == []


# ---------------------------------------------------------------------------
# TLHEngine — generate_all_tlh_pairs
# ---------------------------------------------------------------------------


class TestTLHEngineGenerateAll:
    def test_generates_all_eligible_pairs(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),
                "VUN.TO": (50.0, 3000.0),
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={
                "VCN.TO": ["XIC.TO"],
                "VUN.TO": ["XUU.TO", "ZSP.TO"],
            },
            current_prices={"VCN.TO": 30.0, "VUN.TO": 50.0},
        )
        df = engine.generate_all_tlh_pairs()
        # VCN: 1 pair; VUN: 2 pairs → 3 total
        assert len(df) == 3
        assert set(df["sell_ticker"]) == {"VCN.TO", "VUN.TO"}
        assert set(df["buy_ticker"]) == {"XIC.TO", "XUU.TO", "ZSP.TO"}

    def test_empty_when_no_opportunities(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 3000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 40.0},
        )
        df = engine.generate_all_tlh_pairs()
        assert df.empty
        assert list(df.columns) == [
            "sell_ticker",
            "buy_ticker",
            "sell_shares",
            "sell_proceeds",
            "unrealized_loss",
            "superficial_loss_risk",
            "superficial_loss_amount",
            "note",
        ]

    def test_respects_min_filters(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),  # loss = 1000 at price 30
                "SMALL": (10.0, 200.0),  # loss = 50 at price 15
            }
        )
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"], "SMALL": ["BIG"]},
            current_prices={"VCN.TO": 30.0, "SMALL": 15.0},
        )
        df = engine.generate_all_tlh_pairs(min_loss_dollars=100.0)
        assert len(df) == 1
        assert df.iloc[0]["sell_ticker"] == "VCN.TO"


# ---------------------------------------------------------------------------
# TLHEngine — check_superficial_loss
# ---------------------------------------------------------------------------


class TestTLHEngineSuperficialLoss:
    def test_no_buys_in_window(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        # No buys of VCN in the ledger
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        is_sl, amount = engine.check_superficial_loss("VCN.TO", date(2024, 6, 15))
        assert is_sl is False
        assert amount == 0.0

    def test_buy_within_window_triggers_superficial_loss(self):
        acb = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 4000.0)}  # ACB/sh = 40
        )
        # Record a buy within 30 days (same ticker).
        # Weighted-average ACB: (4000 + 20*35) / 120 = 4700 / 120 = 39.1667
        acb.record_buy("VCN.TO", 20.0, 35.0, date(2024, 6, 10))
        # Disposition on June 15 -> settlement June 17 -> window May 18 - Jul 17
        # Buy on June 10 is within window

        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        is_sl, amount = engine.check_superficial_loss("VCN.TO", date(2024, 6, 15))
        assert is_sl is True
        # Total loss = 120*30 - 4700 = 3600 - 4700 = -1100
        # Shares bought in window = 20; shares held = 120
        # Superficial proportion = 20/120 = 1/6
        # Superficial amount = 1100 * 1/6 = 183.33
        assert amount == pytest.approx(1100.0 * 20.0 / 120.0)

    def test_buy_outside_window_no_trigger(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        # Buy 60 days before disposition
        acb.record_buy("VCN.TO", 10.0, 40.0, date(2024, 3, 1))

        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        is_sl, amount = engine.check_superficial_loss("VCN.TO", date(2024, 6, 15))
        # Settlement June 17; window May 18 – Jul 17
        # Buy March 1 is outside
        assert is_sl is False

    def test_no_price_returns_risky_flag(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        acb.record_buy("VCN.TO", 10.0, 40.0, date(2024, 6, 14))

        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={},  # no price
        )
        is_sl, amount = engine.check_superficial_loss("VCN.TO", date(2024, 6, 15))
        assert is_sl is True
        assert amount == 0.0  # cannot compute

    def test_no_shares_held_returns_false(self):
        acb = ACBTracker()
        acb.record_buy("VCN.TO", 10.0, 40.0, date(2024, 6, 14))
        acb.record_sell("VCN.TO", 10.0, 40.0, date(2024, 6, 15))
        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        is_sl, _ = engine.check_superficial_loss("VCN.TO", date(2024, 6, 16))
        assert is_sl is False

    def test_full_buy_back_total_superficial(self):
        """If all shares were bought within the window, the entire loss is
        superficial."""
        acb = ACBTracker()
        # All shares bought within 30 days before disposition
        acb.record_buy("VCN.TO", 100.0, 40.0, date(2024, 6, 10))
        # Position: 100 sh, ACB = 4000, current price = 30 → loss 1000

        engine = TLHEngine(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
        )
        is_sl, amount = engine.check_superficial_loss("VCN.TO", date(2024, 6, 15))
        assert is_sl is True
        # Entire loss is superficial: 100/100 = 1.0 * 1000
        assert amount == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# TLHEngine — is_within_superficial_window
# ---------------------------------------------------------------------------


class TestTLHEngineIsWithinWindow:
    def test_buy_within_window(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        # Buy 5 days before disposition date → settlement = disposition + 2
        acb.record_buy("VCN.TO", 10.0, 40.0, date(2024, 6, 10))

        engine = TLHEngine(acb_tracker=acb)
        result = engine.is_within_superficial_window("VCN.TO", date(2024, 6, 15))
        assert result is True

    def test_buy_outside_window(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        acb.record_buy("VCN.TO", 10.0, 40.0, date(2024, 1, 1))

        engine = TLHEngine(acb_tracker=acb)
        result = engine.is_within_superficial_window("VCN.TO", date(2024, 6, 15))
        assert result is False

    def test_other_ticker_also_checked(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        acb.record_buy("XIC.TO", 10.0, 30.0, date(2024, 6, 14))

        engine = TLHEngine(acb_tracker=acb)
        result = engine.is_within_superficial_window(
            "VCN.TO", date(2024, 6, 15), other_ticker="XIC.TO"
        )
        assert result is True

    def test_empty_trade_ledger(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        engine = TLHEngine(acb_tracker=acb)
        result = engine.is_within_superficial_window("VCN.TO", date(2024, 6, 15))
        assert result is False


# ---------------------------------------------------------------------------
# TLHEngine — superficial loss in generate_tlh_pairs
# ---------------------------------------------------------------------------


class TestTLHEnginePairsWithSuperficialLoss:
    def test_superficial_loss_flag_in_pairs(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        # Buy within window to trigger superficial loss
        acb.record_buy("VCN.TO", 50.0, 35.0, date(2024, 6, 14))

        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 30.0},
        )
        proposals = engine.generate_tlh_pairs("VCN.TO", as_of_date=date(2024, 6, 15))
        assert len(proposals) == 1
        p = proposals[0]
        assert p.superficial_loss_risk is True
        assert p.superficial_loss_amount > 0
        assert "SUPERFICIAL LOSS" in p.note

    def test_no_superficial_loss_when_no_recent_buy(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        # Buy well outside the window
        acb.record_buy("VCN.TO", 10.0, 40.0, date(2023, 1, 1))

        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 30.0},
        )
        proposals = engine.generate_tlh_pairs("VCN.TO", as_of_date=date(2024, 6, 15))
        p = proposals[0]
        assert p.superficial_loss_risk is False
        assert p.superficial_loss_amount == 0.0
        assert p.note == ""


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

    def test_tlh_trade_defaults(self):
        trade = TLHTrade(
            sell_ticker="VCN.TO",
            buy_ticker="XIC.TO",
            sell_shares=100.0,
            sell_proceeds=3000.0,
            unrealized_loss=-1000.0,
        )
        assert trade.superficial_loss_risk is False
        assert trade.superficial_loss_amount == 0.0
        assert trade.note == ""


# ---------------------------------------------------------------------------
# analyze_tlh_opportunities (top-level integration)
# ---------------------------------------------------------------------------


class TestAnalyzeTLHOpportunities:
    def test_full_analysis(self):
        acb = ACBTracker(
            initial_positions={
                "VCN.TO": (100.0, 4000.0),
                "VUN.TO": (50.0, 3000.0),
            }
        )
        result = analyze_tlh_opportunities(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0, "VUN.TO": 65.0},
            account_type=AccountType.NON_REG,
            switch_fund_map={"VCN.TO": ["XIC.TO", "ZCN.TO"]},
            portfolio_name="test_portfolio",
            min_loss_dollars=100.0,
        )
        assert result.portfolio_name == "test_portfolio"
        assert not result.tlh_trades.empty
        assert len(result.tlh_trades) == 2  # VCN → XIC and VCN → ZCN
        assert "VCN.TO" in result.acb_summary["ticker"].values
        assert result.superficial_loss_flags == []

    def test_analysis_with_superficial_loss_flag(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        acb.record_buy("VCN.TO", 10.0, 35.0, date(2024, 6, 14))

        result = analyze_tlh_opportunities(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
            account_type=AccountType.NON_REG,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            as_of_date=date(2024, 6, 15),
        )
        assert len(result.superficial_loss_flags) == 1
        assert "SUPERFICIAL LOSS" in result.superficial_loss_flags[0]

    def test_empty_analysis(self):
        acb = ACBTracker()
        result = analyze_tlh_opportunities(
            acb_tracker=acb,
            current_prices={},
            account_type=AccountType.NON_REG,
            portfolio_name="empty",
        )
        assert result.tlh_trades.empty
        assert result.acb_summary.empty
        assert result.superficial_loss_flags == []

    def test_switch_fund_map_from_path(self, tmp_path):
        path = tmp_path / "switch.json"
        path.write_text('{"VCN.TO": ["XIC.TO"]}', encoding="utf-8")

        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        result = analyze_tlh_opportunities(
            acb_tracker=acb,
            current_prices={"VCN.TO": 30.0},
            account_type=AccountType.NON_REG,
            switch_fund_map=path,
        )
        assert len(result.tlh_trades) == 1


# ---------------------------------------------------------------------------
# TLH Account Guardrails — tax-sheltered account prohibition
# ---------------------------------------------------------------------------


class TestTLHAccountGuardrails:
    """Ensure TLH entry points reject tax-sheltered account types."""

    @pytest.fixture
    def _tracker_and_prices(self):
        """Minimal non-empty inputs so we test the guard, not empty-edge paths."""
        tracker = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        prices = {"VCN.TO": 30.0}
        return tracker, prices

    @pytest.mark.parametrize(
        "account_type",
        [AccountType.TFSA, AccountType.FHSA, AccountType.RRSP],
    )
    def test_raises_for_tax_sheltered_accounts(self, _tracker_and_prices, account_type):
        """TFSA, FHSA, and RRSP must all raise ValueError."""
        tracker, prices = _tracker_and_prices
        with pytest.raises(ValueError, match="structurally prohibited"):
            analyze_tlh_opportunities(
                acb_tracker=tracker,
                current_prices=prices,
                account_type=account_type,
            )

    @pytest.mark.parametrize(
        "account_type",
        [AccountType.NON_REG, AccountType.LIRA, AccountType.RRIF],
    )
    def test_allows_taxable_and_locked_in_accounts(
        self, _tracker_and_prices, account_type
    ):
        """NON_REG, LIRA, and RRIF must NOT raise (proceed to TLH analysis)."""
        tracker, prices = _tracker_and_prices
        # Should not raise
        result = analyze_tlh_opportunities(
            acb_tracker=tracker,
            current_prices=prices,
            account_type=account_type,
        )
        assert isinstance(result, TLHRebalanceResult)


# ---------------------------------------------------------------------------
# format_tlh_rebalance_result
# ---------------------------------------------------------------------------


class TestFormatTLHRebalanceResult:
    def test_formats_complete_result(self):
        acb = ACBTracker(initial_positions={"VCN.TO": (100.0, 4000.0)})
        engine = TLHEngine(
            acb_tracker=acb,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 30.0},
        )
        tlh_trades = engine.generate_all_tlh_pairs()
        acb_summary = acb.summary(current_prices={"VCN.TO": 30.0})
        result = TLHRebalanceResult(
            portfolio_name="test",
            tlh_trades=tlh_trades,
            acb_summary=acb_summary,
            superficial_loss_flags=[],
        )
        output = format_tlh_rebalance_result(result)
        assert "Tax-Loss Harvesting Analysis" in output
        assert "test" in output
        assert "ACB Summary" in output
        assert "Proposed TLH Switch Trades" in output
        assert "VCN.TO" in output
        assert "XIC.TO" in output

    def test_formats_empty_result(self):
        result = TLHRebalanceResult(
            portfolio_name="empty",
            tlh_trades=pd.DataFrame(),
            acb_summary=pd.DataFrame(),
        )
        output = format_tlh_rebalance_result(result)
        assert "no positions tracked" in output
        assert "No eligible tax-loss harvesting opportunities" in output

    def test_formats_with_superficial_warnings(self):
        acb_summary = pd.DataFrame(
            {
                "ticker": ["VCN.TO"],
                "total_shares": [100.0],
                "total_cost": [4000.0],
                "acb_per_share": [40.0],
            }
        )
        tlh_trades = pd.DataFrame(
            {
                "sell_ticker": ["VCN.TO"],
                "buy_ticker": ["XIC.TO"],
                "sell_shares": [100.0],
                "sell_proceeds": [3000.0],
                "unrealized_loss": [-1000.0],
                "superficial_loss_risk": [True],
                "superficial_loss_amount": [200.0],
                "note": ["WARNING: superficial loss risk"],
            }
        )
        result = TLHRebalanceResult(
            portfolio_name="test",
            tlh_trades=tlh_trades,
            acb_summary=acb_summary,
            superficial_loss_flags=["VCN.TO: WARNING: superficial loss risk"],
        )
        output = format_tlh_rebalance_result(result)
        assert "Superficial Loss Warnings" in output
        assert "VCN.TO" in output


# ---------------------------------------------------------------------------
# End-to-end: typical Canadian non-registered TLH scenario
# ---------------------------------------------------------------------------


class TestEndToEndTLHScenario:
    """Simulate a realistic TLH scenario for a Canadian non-registered account."""

    def test_typical_canadian_etf_tlh(self):
        """Buy VCN.TO at high price, market drops, propose TLH switch to XIC.TO."""
        tracker = ACBTracker()

        # Investor buys VCN.TO over several months
        tracker.record_buy("VCN.TO", 50.0, 45.0, date(2024, 1, 15))
        tracker.record_buy("VCN.TO", 50.0, 44.0, date(2024, 2, 15))
        tracker.record_buy("VCN.TO", 50.0, 46.0, date(2024, 3, 15))

        # ACB: (50*45 + 50*44 + 50*46) / 150 = (2250 + 2200 + 2300) / 150
        # = 6750 / 150 = 45.00
        assert tracker.get_acb_per_share("VCN.TO") == pytest.approx(45.0)
        assert tracker.get_shares("VCN.TO") == 150.0

        # Also buy VUN.TO which is slightly up
        tracker.record_buy("VUN.TO", 100.0, 55.0, date(2024, 1, 15))

        # Market drops in November — prime TLH season
        current_prices = {"VCN.TO": 40.0, "VUN.TO": 58.0, "XIC.TO": 38.0}
        switch_map = {"VCN.TO": ["XIC.TO", "ZCN.TO"], "VUN.TO": ["XUU.TO"]}

        engine = TLHEngine(
            acb_tracker=tracker,
            switch_fund_map=switch_map,
            current_prices=current_prices,
        )

        # Identify opportunities
        opportunities = engine.identify_tlh_opportunities(
            min_loss_dollars=200.0, as_of_date=date(2024, 11, 15)
        )
        # VCN: market = 150*40 = 6000, ACB = 6750 → loss = -750 → eligible
        # VUN: market = 100*58 = 5800, ACB = 5500 → gain = 300 → not eligible
        assert len(opportunities) == 1
        assert opportunities.iloc[0]["ticker"] == "VCN.TO"
        assert opportunities.iloc[0]["unrealized_gain_loss"] == -750.0

        # Generate switch trades
        proposals = engine.generate_tlh_pairs("VCN.TO", as_of_date=date(2024, 11, 15))
        assert len(proposals) == 2

        # No buys within the 61-day window, so no superficial loss
        for p in proposals:
            assert p.superficial_loss_risk is False
            assert p.superficial_loss_amount == 0.0

        # Execute the TLH: sell VCN, buy XIC (manually)
        gain_loss = tracker.record_sell("VCN.TO", 150.0, 40.0, date(2024, 11, 20))
        # ACB of sold = 150 * 45 = 6750; proceeds = 150 * 40 = 6000
        # loss = 6000 - 6750 = -750
        assert gain_loss == pytest.approx(-750.0)
        assert tracker.get_shares("VCN.TO") == 0.0

        # Buy replacement (XIC.TO) with the proceeds
        xic_price = current_prices["XIC.TO"]
        xic_shares = 6000.0 / xic_price
        tracker.record_buy("XIC.TO", xic_shares, xic_price, date(2024, 11, 20))

        # Verify new position
        assert tracker.get_shares("XIC.TO") == pytest.approx(6000.0 / 38.0)
        assert tracker.get_shares("VCN.TO") == 0.0

    def test_superficial_loss_prevents_tlh(self):
        """If investor bought VCN within the window, the loss is superficial."""
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 4500.0)}  # ACB/sh = 45
        )
        # Buy more VCN within 30 days before proposed disposition
        tracker.record_buy("VCN.TO", 20.0, 40.0, date(2024, 11, 10))

        engine = TLHEngine(
            acb_tracker=tracker,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            current_prices={"VCN.TO": 40.0},
        )

        proposals = engine.generate_tlh_pairs("VCN.TO", as_of_date=date(2024, 11, 20))
        assert len(proposals) == 1
        p = proposals[0]
        # The 20 shares bought in the window create a superficial loss risk
        assert p.superficial_loss_risk is True
        # Weighted-average ACB: (4500 + 20*40) / 120 = 5300/120 = 44.1667
        # Loss = 120*40 - 5300 = 4800 - 5300 = -500
        # Superficial proportion = 20/120 = 1/6, amount = 500/6 = 83.33
        assert p.superficial_loss_amount == pytest.approx(500.0 * 20.0 / 120.0)

        # The full end-to-end analysis should flag this
        result = analyze_tlh_opportunities(
            acb_tracker=tracker,
            current_prices={"VCN.TO": 40.0},
            account_type=AccountType.NON_REG,
            switch_fund_map={"VCN.TO": ["XIC.TO"]},
            as_of_date=date(2024, 11, 20),
        )
        assert len(result.superficial_loss_flags) == 1

    def test_return_of_capital_adjusts_acb_correctly(self):
        """ROC distributions should lower ACB, potentially creating deeper
        unrealized losses and more TLH opportunity."""
        tracker = ACBTracker(
            initial_positions={"VCN.TO": (100.0, 4000.0)}  # ACB/sh = 40
        )

        # Stock drops to 35
        gl_before = tracker.get_unrealized_gain_loss("VCN.TO", 35.0)
        # Market = 3500, ACB = 4000 → loss = -500
        assert gl_before == -500.0

        # ROC distribution of $2/share
        pos, gain = tracker.record_roc("VCN.TO", 200.0, date(2024, 6, 15))
        assert gain == 0.0
        assert tracker.get_acb_per_share("VCN.TO") == 38.0

        # Now the unrealized loss is different
        gl_after = tracker.get_unrealized_gain_loss("VCN.TO", 35.0)
        # Market = 3500, ACB = 3800 → loss = -300
        assert gl_after == pytest.approx(-300.0)

        # Note: ROC reduces ACB, which reduces capital loss.  However in a
        # falling market, holding through ROC can still produce losses.
        # The tracker correctly models this CRA-required behavior.
