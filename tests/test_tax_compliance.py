"""Tests for the Canadian tax compliance interlock.

Covers ACB tracking with commission support and CRA superficial loss
rule enforcement across multi-account structures (TFSA, RRSP, Non-Reg).

All tests use synthetic data only — no network calls.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pysharpe.guardrails.tax_compliance import (
    ACBTracker,
    SuperficialLossGuardrail,
    TransactionRecord,
    build_default_identical_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _d(days_offset: int, base: date | None = None) -> date:
    """Return a date offset from *base* (default 2024-06-15)."""
    if base is None:
        base = date(2024, 6, 15)
    return base + timedelta(days=days_offset)


def _buy(
    ticker: str,
    trade_date: date,
    shares: float = 10.0,
    price: float = 100.0,
    account: str = "TFSA",
    commission: float = 0.0,
) -> TransactionRecord:
    return TransactionRecord(
        ticker=ticker,
        date=trade_date,
        account_type=account,
        action="BUY",
        shares=shares,
        price_per_share=price,
        total_amount=shares * price + commission,
        commission=commission,
    )


def _sell(
    ticker: str,
    trade_date: date,
    shares: float = 10.0,
    price: float = 90.0,
    account: str = "NON_REG",
    commission: float = 0.0,
    acb_per_share: float = 100.0,
) -> TransactionRecord:
    acb_of_sold = shares * acb_per_share
    proceeds = shares * price
    gain_loss = proceeds - acb_of_sold - commission
    return TransactionRecord(
        ticker=ticker,
        date=trade_date,
        account_type=account,
        action="SELL",
        shares=shares,
        price_per_share=price,
        total_amount=proceeds,
        commission=commission,
        realized_gain_loss=gain_loss,
    )


# ===========================================================================
# ACB Tracker — commission support
# ===========================================================================


class TestACBTrackerCommission:
    """Commission-aware ACB tracking for Non-Registered accounts."""

    # -- Initialization -------------------------------------------------------

    def test_empty_initialization(self):
        tracker = ACBTracker()
        assert tracker.positions == {}
        assert tracker.trades == []

    def test_initial_positions(self):
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})
        pos = tracker.get_position("VFV.TO")
        assert pos.total_shares == 100.0
        assert pos.total_cost == 5000.0
        assert pos.acb_per_share == 50.0

    def test_initial_negative_shares_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ACBTracker({"VFV.TO": (-10.0, 500.0)})

    def test_initial_negative_cost_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ACBTracker({"VFV.TO": (10.0, -100.0)})

    def test_initial_zero_cost_with_shares_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ACBTracker({"VFV.TO": (10.0, 0.0)})

    # -- Buy with commission --------------------------------------------------

    def test_buy_no_commission(self):
        tracker = ACBTracker()
        pos = tracker.record_buy("VFV.TO", 10.0, 50.0, _d(0))
        assert pos.total_shares == 10.0
        assert pos.total_cost == 500.0
        assert pos.acb_per_share == 50.0

    def test_buy_with_commission_adds_to_acb(self):
        """Commission is added to the cost base on purchase."""
        tracker = ACBTracker()
        pos = tracker.record_buy("VFV.TO", 10.0, 50.0, _d(0), commission=9.99)
        # Total ACB = 10*50 + 9.99 = 509.99
        assert pos.total_cost == pytest.approx(509.99)
        assert pos.acb_per_share == pytest.approx(50.999)

    def test_multiple_buys_weighted_average_with_commission(self):
        """ACB per share reflects weighted average including commissions."""
        tracker = ACBTracker()
        tracker.record_buy("VFV.TO", 10.0, 50.0, _d(0), commission=5.0)
        tracker.record_buy("VFV.TO", 20.0, 40.0, _d(5), commission=10.0)

        pos = tracker.get_position("VFV.TO")
        # Total cost = 10*50+5 + 20*40+10 = 505 + 810 = 1315
        expected_cost = (10 * 50 + 5) + (20 * 40 + 10)
        assert pos.total_cost == pytest.approx(expected_cost)
        # ACB/sh = 1315/30 ≈ 43.8333
        assert pos.acb_per_share == pytest.approx(1315 / 30)

    def test_buy_negative_commission_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record_buy("VFV.TO", 10.0, 50.0, _d(0), commission=-5.0)

    def test_buy_negative_shares_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VFV.TO", -5.0, 50.0, _d(0))

    def test_buy_zero_shares_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VFV.TO", 0.0, 50.0, _d(0))

    def test_buy_negative_price_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.record_buy("VFV.TO", 10.0, -10.0, _d(0))

    # -- Sell with commission -------------------------------------------------

    def test_sell_with_commission_reduces_gain(self):
        """Sales commission reduces realized gain dollar-for-dollar."""
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})  # ACB/sh = 50
        gain = tracker.record_sell("VFV.TO", 50.0, 60.0, _d(10), commission=9.99)
        # ACB of sold = 50*50 = 2500; proceeds = 50*60 = 3000
        # gain = 3000 - 2500 - 9.99 = 490.01
        expected = 3000.0 - 2500.0 - 9.99
        assert gain == pytest.approx(expected)

    def test_sell_commission_turns_gain_to_loss(self):
        """A small gain can become a loss when commission exceeds it."""
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})  # ACB/sh = 50
        # Sell 1 share at 50.05 → gain of $0.05, but $9.99 commission → loss
        gain = tracker.record_sell("VFV.TO", 1.0, 50.05, _d(10), commission=9.99)
        expected = 50.05 - 50.0 - 9.99
        assert gain == pytest.approx(expected)
        assert gain < 0

    def test_sell_no_commission(self):
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})
        gain = tracker.record_sell("VFV.TO", 10.0, 45.0, _d(10))
        # ACB of sold = 10*50 = 500; proceeds = 10*45 = 450
        # loss = 450 - 500 = -50
        assert gain == pytest.approx(-50.0)

    def test_acb_per_share_unchanged_after_partial_sell(self):
        """Per-unit ACB does not change after a partial disposition."""
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})  # ACB/sh = 50
        acb_before = tracker.get_acb_per_share("VFV.TO")
        tracker.record_sell("VFV.TO", 30.0, 60.0, _d(10))
        acb_after = tracker.get_acb_per_share("VFV.TO")
        assert acb_after == pytest.approx(acb_before)

    def test_full_sell_resets_position(self):
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})
        tracker.record_sell("VFV.TO", 100.0, 60.0, _d(10))
        pos = tracker.get_position("VFV.TO")
        assert pos.total_shares == 0.0
        assert pos.total_cost == 0.0
        assert pos.acb_per_share == 0.0

    def test_sell_more_than_held_raises(self):
        tracker = ACBTracker({"VFV.TO": (10.0, 500.0)})
        with pytest.raises(ValueError, match="only 10.0 held"):
            tracker.record_sell("VFV.TO", 15.0, 60.0, _d(0))

    def test_sell_negative_commission_raises(self):
        tracker = ACBTracker({"VFV.TO": (10.0, 500.0)})
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record_sell("VFV.TO", 5.0, 60.0, _d(0), commission=-1.0)

    def test_sell_empty_position_raises(self):
        tracker = ACBTracker()
        with pytest.raises(ValueError, match="no shares held"):
            tracker.record_sell("VFV.TO", 5.0, 50.0, _d(0))

    # -- Trade record integrity -----------------------------------------------

    def test_buy_trade_record_fields(self):
        tracker = ACBTracker()
        tracker.record_buy("VFV.TO", 10.0, 50.0, _d(0), commission=5.0)
        txn = tracker.trades[0]
        assert txn.ticker == "VFV.TO"
        assert txn.action == "BUY"
        assert txn.shares == 10.0
        assert txn.price_per_share == 50.0
        assert txn.commission == 5.0
        assert txn.total_amount == pytest.approx(505.0)
        assert txn.realized_gain_loss == 0.0

    def test_sell_trade_record_fields(self):
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})
        tracker.record_sell("VFV.TO", 50.0, 45.0, _d(10), commission=9.99)
        txn = tracker.trades[0]
        assert txn.ticker == "VFV.TO"
        assert txn.action == "SELL"
        assert txn.shares == 50.0
        assert txn.commission == 9.99
        assert txn.total_amount == pytest.approx(2250.0)  # 50*45

    # -- Unrealized gain/loss -------------------------------------------------

    def test_unrealized_gain(self):
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})  # ACB/sh = 50
        ugl = tracker.get_unrealized_gain_loss("VFV.TO", 60.0)
        assert ugl == pytest.approx(1000.0)  # 100*60 - 5000

    def test_unrealized_loss(self):
        tracker = ACBTracker({"VFV.TO": (100.0, 5000.0)})
        ugl = tracker.get_unrealized_gain_loss("VFV.TO", 40.0)
        assert ugl == pytest.approx(-1000.0)

    def test_unrealized_zero_shares(self):
        tracker = ACBTracker()
        assert tracker.get_unrealized_gain_loss("VFV.TO", 50.0) == 0.0


# ===========================================================================
# Superficial Loss Guardrail — detection
# ===========================================================================


class TestSuperficialLossDetection:
    """Detection of CRA superficial loss rule violations."""

    # -- No violation cases ---------------------------------------------------

    def test_no_violation_when_sell_is_gain(self):
        """Selling at a gain does not trigger superficial loss rules."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-5), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=110.0, acb_per_share=100.0)
        # Realized gain, not a loss
        assert sell.realized_gain_loss > 0

        proposed = [sell]
        violations = guardrail.detect_violations(proposed)
        assert len(violations) == 0

    def test_no_violation_when_buy_is_non_reg(self):
        """Buying back identical property in Non-Reg does not trigger
        superficial loss (the denied loss is added to ACB instead)."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-5), account="NON_REG"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        assert sell.realized_gain_loss < 0

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 0

    def test_no_violation_outside_window(self):
        """A buy more than 30 days away does not trigger the rule."""
        guardrail = SuperficialLossGuardrail(window_days=30)
        # Buy 35 days before
        guardrail.record_transaction(_buy("VFV.TO", _d(-35), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        assert sell.realized_gain_loss < 0

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 0

    def test_no_violation_different_ticker(self):
        """A buy of a completely different ticker does not trigger the rule."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VDY.TO", _d(-5), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 0

    # -- Violation cases ------------------------------------------------------

    def test_superficial_loss_detected_tfsa_buy_before(self):
        """TFSA purchase within 30 days before Non-Reg sale at loss."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-10), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        assert sell.realized_gain_loss < 0

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1
        v = violations[0]
        assert v.sell_trade.ticker == "VFV.TO"
        assert v.conflicting_buy.account_type == "TFSA"
        assert v.days_delta == 10
        assert "Superficial loss" in v.message

    def test_superficial_loss_detected_tfsa_buy_after(self):
        """Proposed TFSA buy within 30 days after Non-Reg sale at loss."""
        guardrail = SuperficialLossGuardrail()
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        buy = _buy("VFV.TO", _d(15), account="TFSA")

        # Both in proposed slate
        violations = guardrail.detect_violations([sell, buy])
        assert len(violations) >= 1

    def test_superficial_loss_rrsp_buy(self):
        """RRSP purchase within window triggers superficial loss too."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-5), account="RRSP"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1

    def test_historical_sell_proposed_buy_violation(self):
        """A historical Non-Reg sale at loss conflicts with a proposed
        TFSA buy within 30 days."""
        guardrail = SuperficialLossGuardrail()
        hist_sell = _sell("VFV.TO", _d(-10), price=90.0, acb_per_share=100.0)
        guardrail.record_transaction(hist_sell)

        proposed_buy = _buy("VFV.TO", _d(0), account="TFSA")
        violations = guardrail.detect_violations([proposed_buy])
        assert len(violations) == 1
        v = violations[0]
        assert v.sell_trade.ticker == "VFV.TO"
        assert v.conflicting_buy.account_type == "TFSA"

    # -- Identical property ---------------------------------------------------

    def test_identical_property_vfv_and_voo(self):
        """VFV.TO and VOO are identical property (both S&P 500)."""
        guardrail = SuperficialLossGuardrail()
        assert guardrail.are_identical("VFV.TO", "VOO")
        assert guardrail.are_identical("VOO", "VFV.TO")

    def test_identical_property_different_indices(self):
        """VDY.TO (dividend) is NOT identical to VFV.TO (S&P 500)."""
        guardrail = SuperficialLossGuardrail()
        assert not guardrail.are_identical("VFV.TO", "VDY.TO")

    def test_superficial_loss_identical_property_different_ticker(self):
        """Selling VFV.TO at loss conflicts with VOO buy in TFSA."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VOO", _d(-5), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1
        v = violations[0]
        assert v.conflicting_buy.ticker == "VOO"
        assert "VFV.TO" in v.identical_tickers
        assert "VOO" in v.identical_tickers

    def test_get_identical_tickers_includes_self(self):
        guardrail = SuperficialLossGuardrail()
        identical = guardrail.get_identical_tickers("VFV.TO")
        assert "VFV.TO" in identical

    # -- Multiple violations --------------------------------------------------

    def test_multiple_violations(self):
        """A single sale can conflict with multiple buys."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-10), account="TFSA"))
        guardrail.record_transaction(_buy("VOO", _d(-5), account="RRSP"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 2

    def test_qqc_identical_property(self):
        """QQC.TO and QQQ are identical (both NASDAQ-100)."""
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("QQQ", _d(-3), account="TFSA"))
        sell = _sell("QQC.TO", _d(0), price=80.0, acb_per_share=100.0)

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1


# ===========================================================================
# Superficial Loss Guardrail — trade slate validation
# ===========================================================================


class TestValidateTradeSlate:
    """Validation and automatic re-routing of violating trade slates."""

    def test_clean_slate_passes_through(self):
        """A slate with no violations is returned unchanged."""
        guardrail = SuperficialLossGuardrail()
        propsed = [
            _buy("VFV.TO", _d(0), account="TFSA"),
            _buy("VDY.TO", _d(0), account="RRSP"),
        ]
        validated, violations = guardrail.validate_trade_slate(propsed)
        assert len(violations) == 0
        assert len(validated) == 2

    def test_violating_buy_blocked(self):
        """A TFSA buy that conflicts with a Non-Reg sale is blocked."""
        guardrail = SuperficialLossGuardrail()
        hist_sell = _sell("VFV.TO", _d(-10), price=90.0, acb_per_share=100.0)
        guardrail.record_transaction(hist_sell)

        proposed = [
            _buy("VFV.TO", _d(0), account="TFSA"),
            _buy("VDY.TO", _d(0), account="TFSA"),
        ]
        validated, violations = guardrail.validate_trade_slate(proposed)
        assert len(violations) == 1
        # VFV buy should be blocked; VDY buy should remain
        assert len(validated) == 1
        assert validated[0].ticker == "VDY.TO"

    def test_reroute_to_non_identical_asset(self):
        """Blocked cash is re-routed to the next-best non-identical asset."""
        guardrail = SuperficialLossGuardrail()
        hist_sell = _sell("VFV.TO", _d(-10), price=90.0, acb_per_share=100.0)
        guardrail.record_transaction(hist_sell)

        proposed = [
            _buy("VFV.TO", _d(0), account="TFSA", price=100.0, shares=10.0),
        ]

        ranking = {"VDY.TO": 0.85, "QQC.TO": 0.70, "VFV.TO": 0.90, "VOO": 0.88}
        validated, violations = guardrail.validate_trade_slate(proposed, ranking)
        assert len(violations) == 1
        # VFV buy blocked, rerouted to VDY (highest non-identical)
        rerouted = [t for t in validated if t.action == "BUY"]
        tickers = {t.ticker for t in rerouted}
        # Should NOT contain VFV or VOO (identical group)
        assert "VFV.TO" not in tickers
        assert "VOO" not in tickers

    def test_reroute_with_no_alternatives(self):
        """When all assets are identical property, no re-route is possible."""
        guardrail = SuperficialLossGuardrail()
        hist_sell = _sell("VFV.TO", _d(-10), price=90.0, acb_per_share=100.0)
        guardrail.record_transaction(hist_sell)

        proposed = [
            _buy("VFV.TO", _d(0), account="TFSA"),
        ]
        # Only VFV-related tickers in ranking (all identical)
        ranking = {"VFV.TO": 0.90, "VOO": 0.88, "SPY": 0.85}
        validated, violations = guardrail.validate_trade_slate(proposed, ranking)
        assert len(violations) == 1
        # VFV buy blocked, no alternatives to re-route to
        buys = [
            t
            for t in validated
            if t.is_buy and t.ticker not in {"VFV.TO", "VOO", "SPY"}
        ]
        # Either empty or a low-confidence placeholder
        assert len(buys) <= 1

    def test_violation_message_content(self):
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-15), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)

        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1
        msg = violations[0].message
        assert "VFV.TO" in msg
        assert "TFSA" in msg
        assert "15 days" in msg
        assert "superficial" in msg.lower()


# ===========================================================================
# Superficial Loss Guardrail — history and query
# ===========================================================================


class TestGuardrailHistory:
    def test_record_and_retrieve(self):
        guardrail = SuperficialLossGuardrail()
        txn = _buy("VFV.TO", _d(0), account="TFSA")
        guardrail.record_transaction(txn)
        assert len(guardrail.history) == 1
        assert guardrail.history[0].ticker == "VFV.TO"

    def test_record_multiple(self):
        guardrail = SuperficialLossGuardrail()
        txns = [
            _buy("VFV.TO", _d(0), account="TFSA"),
            _buy("VDY.TO", _d(5), account="RRSP"),
        ]
        guardrail.record_transactions(txns)
        assert len(guardrail.history) == 2

    def test_get_trades_in_window(self):
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-20)))
        guardrail.record_transaction(_buy("VFV.TO", _d(-5)))
        guardrail.record_transaction(_buy("VFV.TO", _d(10)))
        guardrail.record_transaction(_buy("VFV.TO", _d(40)))

        window = guardrail.get_trades_in_window(_d(0))
        # 4 trades, all within ±30 of day 0
        assert len(window) == 3  # -20, -5, 10 (40 is outside)

    def test_get_trades_in_window_with_ticker_filter(self):
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(-5)))
        guardrail.record_transaction(_buy("VDY.TO", _d(-5)))

        window = guardrail.get_trades_in_window(_d(0), ticker="VFV.TO")
        assert len(window) == 1
        assert window[0].ticker == "VFV.TO"

    def test_clear_history(self):
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(0)))
        guardrail.clear_history()
        assert len(guardrail.history) == 0

    def test_summary(self):
        guardrail = SuperficialLossGuardrail()
        guardrail.record_transaction(_buy("VFV.TO", _d(0), account="TFSA"))
        guardrail.record_transaction(_buy("VDY.TO", _d(5), account="RRSP"))

        summary = guardrail.summary()
        assert summary["total_transactions"] == 2
        assert summary["unique_tickers"] == 2
        assert "TFSA" in summary["accounts"]
        assert "RRSP" in summary["accounts"]


# ===========================================================================
# Default identical-property map
# ===========================================================================


class TestIdenticalMap:
    def test_build_default_map(self):
        m = build_default_identical_map()
        assert "VFV.TO" in m
        assert "VOO" in m["VFV.TO"]
        assert "QQC.TO" in m
        assert "XIU.TO" in m
        assert "XBB.TO" in m

    def test_all_values_are_frozensets(self):
        m = build_default_identical_map()
        for v in m.values():
            assert isinstance(v, frozenset)


# ===========================================================================
# TransactionRecord properties
# ===========================================================================


class TestTransactionRecord:
    def test_is_buy(self):
        assert _buy("VFV.TO", _d(0)).is_buy
        assert not _buy("VFV.TO", _d(0)).is_sell

    def test_is_sell(self):
        assert _sell("VFV.TO", _d(0)).is_sell
        assert not _sell("VFV.TO", _d(0)).is_buy

    def test_is_loss(self):
        sell_gain = _sell("VFV.TO", _d(0), price=110.0, acb_per_share=100.0)
        sell_loss = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        assert not sell_gain.is_loss
        assert sell_loss.is_loss

    def test_is_tax_sheltered(self):
        assert _buy("VFV.TO", _d(0), account="TFSA").is_tax_sheltered
        assert _buy("VFV.TO", _d(0), account="RRSP").is_tax_sheltered
        assert not _buy("VFV.TO", _d(0), account="NON_REG").is_tax_sheltered

    def test_is_non_reg(self):
        assert _buy("VFV.TO", _d(0), account="NON_REG").is_non_reg
        assert not _buy("VFV.TO", _d(0), account="TFSA").is_non_reg


# ===========================================================================
# Integration: S&P 500 loss-harvesting + TFSA dividend reinvestment
# ===========================================================================


class TestIntegrationSP500LossHarvesting:
    """Simulate an S&P 500 loss-harvesting scenario in Non-Reg while TFSA
    dividend reinvestment triggers within 15 days."""

    def test_tfsa_drip_triggers_superficial_loss(self):
        """Full scenario:
        1. Investor holds VFV.TO in both Non-Reg and TFSA.
        2. Day 0: Non-Reg sells VFV.TO at a loss ($90 vs ACB $100).
        3. Day 12: TFSA dividend reinvestment buys VFV.TO (within 30 days).
        4. Guardrail must detect the superficial loss violation.
        """
        guardrail = SuperficialLossGuardrail(window_days=30)

        # --- Day -60: Initial purchases (outside window, not relevant) ---
        guardrail.record_transaction(
            _buy("VFV.TO", _d(-60), account="NON_REG", price=100.0, shares=50.0)
        )
        guardrail.record_transaction(
            _buy("VFV.TO", _d(-60), account="TFSA", price=100.0, shares=20.0)
        )

        # --- Day 0: Loss harvest in Non-Reg ---
        # Sell 25 shares at $90 (ACB $100 → $10 loss/share, $250 total loss)
        harvest_sell = _sell(
            "VFV.TO",
            _d(0),
            price=90.0,
            acb_per_share=100.0,
            shares=25.0,
            account="NON_REG",
            commission=9.99,
        )
        assert harvest_sell.realized_gain_loss < 0  # It's a loss

        guardrail.record_transaction(harvest_sell)

        # --- Day 12: TFSA dividend reinvestment ---
        # $50 dividend buys 0.55 shares at $90
        drip_buy = _buy(
            "VFV.TO",
            _d(12),
            account="TFSA",
            price=90.0,
            shares=0.55,
            commission=0.0,
        )

        # Validate the proposed DRIP buy
        validated, violations = guardrail.validate_trade_slate([drip_buy])
        assert len(violations) == 1, (
            "TFSA DRIP within 15 days of Non-Reg loss sale should trigger "
            "superficial loss"
        )
        v = violations[0]
        assert v.sell_trade.ticker == "VFV.TO"
        assert v.conflicting_buy.account_type == "TFSA"
        assert v.days_delta <= 12
        assert v.sell_trade.realized_gain_loss < 0

        # The DRIP buy should be blocked
        assert len(validated) == 0, (
            "TFSA DRIP buy should be blocked to avoid superficial loss"
        )

    def test_cash_rebalance_rerouted_away_from_identical(self):
        """When a TFSA cash rebalance would buy VFV.TO within 15 days of a
        Non-Reg VFV sale at loss, the cash is re-routed to the next-best
        non-identical underweight asset (e.g., VDY.TO)."""
        guardrail = SuperficialLossGuardrail(window_days=30)

        # Historical Non-Reg loss sale on day -12
        hist_sell = _sell(
            "VFV.TO",
            _d(-12),
            price=85.0,
            acb_per_share=100.0,
            shares=30.0,
            account="NON_REG",
        )
        guardrail.record_transaction(hist_sell)

        # Proposed trades for today's rebalance
        proposed = [
            _buy("VFV.TO", _d(0), account="TFSA", price=86.0, shares=10.0),
            _buy("VDY.TO", _d(0), account="TFSA", price=40.0, shares=5.0),
            _buy("QQC.TO", _d(0), account="TFSA", price=70.0, shares=3.0),
        ]

        # VDY is more underweight than QQC → higher opportunity score
        ranking = {"VDY.TO": 0.92, "QQC.TO": 0.75, "XIU.TO": 0.60, "VFV.TO": 0.50}

        validated, violations = guardrail.validate_trade_slate(proposed, ranking)
        assert len(violations) >= 1

        # VFV buy should be blocked
        vfv_buys = [t for t in validated if t.ticker == "VFV.TO"]
        assert len(vfv_buys) == 0, "VFV.TO buy in TFSA should be blocked"

        # Remaining buys should still be there
        remaining_tickers = {t.ticker for t in validated if t.is_buy}
        assert "VDY.TO" in remaining_tickers
        assert "QQC.TO" in remaining_tickers

    def test_non_reg_buyback_does_not_trigger_guardrail(self):
        """Buying back VFV in the Non-Reg account itself within 30 days
        does NOT trigger the guardrail — the superficial loss rule still
        applies (loss denied, added to ACB), but the guardrail only blocks
        tax-sheltered purchases."""
        guardrail = SuperficialLossGuardrail()

        hist_sell = _sell(
            "VFV.TO",
            _d(-10),
            price=90.0,
            acb_per_share=100.0,
            shares=20.0,
            account="NON_REG",
        )
        guardrail.record_transaction(hist_sell)

        # Buy back in Non-Reg (not a guardrail violation — handled by ACB)
        proposed = [_buy("VFV.TO", _d(0), account="NON_REG", price=91.0, shares=15.0)]
        validated, violations = guardrail.validate_trade_slate(proposed)

        # No violation because the buy is in Non-Reg (not tax-sheltered)
        assert len(violations) == 0
        assert len(validated) == 1

    def test_tfsa_sale_does_not_trigger_superficial_loss(self):
        """A sale in TFSA at a loss is not a superficial loss trigger
        because TFSA losses cannot be claimed anyway. Only Non-Reg sales
        at a loss are relevant."""
        guardrail = SuperficialLossGuardrail()

        # TFSA sale at loss
        tfsa_sell = _sell(
            "VFV.TO",
            _d(-5),
            price=90.0,
            acb_per_share=100.0,
            shares=10.0,
            account="TFSA",
        )
        guardrail.record_transaction(tfsa_sell)

        proposed = [_buy("VFV.TO", _d(0), account="RRSP")]
        validated, violations = guardrail.validate_trade_slate(proposed)

        # TFSA sale doesn't trigger superficial loss concerns
        # (Only Non-Reg sales are relevant since only those produce
        # claimable capital losses)
        assert len(violations) == 0


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_proposed_slate(self):
        guardrail = SuperficialLossGuardrail()
        validated, violations = guardrail.validate_trade_slate([])
        assert len(validated) == 0
        assert len(violations) == 0

    def test_exactly_30_days_boundary(self):
        """A buy exactly 30 days before the sale is within the window."""
        guardrail = SuperficialLossGuardrail(window_days=30)
        guardrail.record_transaction(
            _buy("VFV.TO", _d(-30), account="TFSA")  # exactly 30 days before
        )
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1

    def test_exactly_31_days_outside_window(self):
        """31 days is outside the default 30-day window."""
        guardrail = SuperficialLossGuardrail(window_days=30)
        guardrail.record_transaction(_buy("VFV.TO", _d(-31), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        violations = guardrail.detect_violations([sell])
        assert len(violations) == 0

    def test_custom_window_days(self):
        """The window can be configured (e.g., 60 days for cautious investors)."""
        guardrail = SuperficialLossGuardrail(window_days=60)
        guardrail.record_transaction(_buy("VFV.TO", _d(-45), account="TFSA"))
        sell = _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0)
        violations = guardrail.detect_violations([sell])
        assert len(violations) == 1

    def test_invalid_window_days_raises(self):
        with pytest.raises(ValueError, match="window_days"):
            SuperficialLossGuardrail(window_days=0)

    def test_proposed_sell_conflicts_with_proposed_buy(self):
        """Both sell and buy are in the same proposed slate."""
        guardrail = SuperficialLossGuardrail()
        proposed = [
            _sell("VFV.TO", _d(0), price=90.0, acb_per_share=100.0, account="NON_REG"),
            _buy("VFV.TO", _d(5), account="TFSA"),
        ]
        violations = guardrail.detect_violations(proposed)
        assert len(violations) >= 1
