"""Tests for mark-to-market valuation of ledger holdings (CAD).

Conventions under test:

* prices are per share in base currency (CAD) and are never derived from a
  transaction's historical ``fx_rate``;
* portfolio value includes cash;
* a missing price is unknown, reported, and excluded — never valued at zero;
* a holding without a cost basis is rejected rather than reported as a gain.
"""

from __future__ import annotations

from datetime import date

import pytest

import pysharpe
from pysharpe.bookkeeping import (
    Account,
    Currency,
    PortfolioLedger,
    Transaction,
    TransactionKind,
    value_portfolio,
)
from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import AccountType

TFSA = AccountType.TFSA
NON_REG = AccountType.NON_REG
TFSA_ACCOUNT = Account.from_registration(TFSA)
NON_REG_ACCOUNT = Account.from_registration(NON_REG)


def _ledger(amount: float = 10_000.0, *, account: AccountType = TFSA):
    ledger = PortfolioLedger()
    ledger.record(
        Transaction(
            date=date(2024, 1, 1),
            account=account,
            kind=TransactionKind.CONTRIBUTION,
            amount=amount,
        )
    )
    return ledger


def _convert(
    ledger: PortfolioLedger,
    cad: float = 6750.0,
    usd: float = 5000.0,
    day: int = 1,
) -> None:
    """Fund a USD balance with an explicit CAD→USD conversion."""
    ledger.record(
        Transaction(
            date=date(2024, 1, day),
            account=TFSA,
            kind=TransactionKind.FX_CONVERSION,
            amount=cad,
            destination_currency=Currency.USD,
            destination_amount=usd,
        )
    )


def _buy(
    ledger: PortfolioLedger,
    ticker: str = "VFV.TO",
    quantity: float = 10.0,
    price: float = 100.0,
    day: int = 2,
    account: AccountType = TFSA,
    **kwargs,
) -> None:
    ledger.record(
        Transaction(
            date=date(2024, 1, day),
            account=account,
            kind=TransactionKind.BUY,
            ticker=ticker,
            quantity=quantity,
            price_per_share=price,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Core valuation
# ---------------------------------------------------------------------------


def test_empty_ledger_has_an_empty_valuation():
    valuation = PortfolioLedger().valuation({"VFV.TO": 110.0})

    assert valuation.positions == ()
    assert valuation.cash == {}
    assert valuation.total_cash == 0.0
    assert valuation.total_market_value == 0.0
    assert valuation.total_book_cost == 0.0
    assert valuation.is_complete
    assert valuation.allocation_by_ticker() == {}
    assert valuation.allocation_by_account() == {}


def test_single_position_is_marked_to_market():
    ledger = _ledger(5000.0)
    _buy(ledger)

    valuation = ledger.valuation({"VFV.TO": 110.0})

    (position,) = valuation.positions
    assert position.market_value == 1100.0
    assert position.book_cost == 1000.0
    assert position.unrealized_gain_loss == 100.0
    assert position.unrealized_gain_loss_pct == pytest.approx(0.1)
    # Cash is part of portfolio value.
    assert valuation.total_cash == 4000.0
    assert valuation.total_market_value == 5100.0
    assert valuation.is_complete


def test_market_price_is_not_derived_from_the_trade_rate():
    """A USD position is valued at the caller's CAD price, not the book rate.

    ACB is fixed at 675 CAD (10 shares at USD 50, booked at 1.35).  A CAD price
    of 70 is a *current* market observation and is used as given, while the USD
    cash left over is marked at the separate current rate the caller supplies.
    """
    ledger = _ledger()
    _convert(ledger)
    _buy(ledger, quantity=10, price=50.0, currency=Currency.USD, fx_rate=1.35)

    valuation = ledger.valuation({"VFV.TO": 70.0}, {Currency.USD: 1.35})

    (position,) = valuation.positions
    assert position.book_cost == pytest.approx(675.0)
    assert position.market_value == 700.0
    assert position.unrealized_gain_loss == pytest.approx(25.0)
    # 3,250 CAD + 4,500 USD at the supplied current rate.
    assert valuation.total_cash == pytest.approx(3250.0 + 4500.0 * 1.35)


def test_commission_loaded_cost_basis_drives_unrealized_gain():
    ledger = _ledger()
    _buy(ledger, quantity=10, price=100.0, commission=10.0)

    valuation = ledger.valuation({"VFV.TO": 110.0})

    assert valuation.total_book_cost == 1010.0
    assert valuation.total_unrealized_gain_loss == 90.0


def test_fractional_shares_are_valued_exactly():
    ledger = _ledger()
    _buy(ledger, quantity=0.5, price=100.0)

    valuation = ledger.valuation({"VFV.TO": 120.0})

    assert valuation.total_market_value == 60.0 + 9950.0
    assert valuation.total_unrealized_gain_loss == 10.0


def test_zero_price_values_the_holding_at_zero_and_realizes_the_whole_cost():
    ledger = _ledger()
    _buy(ledger)

    valuation = ledger.valuation({"VFV.TO": 0.0})

    (position,) = valuation.positions
    assert position.market_value == 0.0
    assert position.unrealized_gain_loss == -1000.0
    assert position.unrealized_gain_loss_pct == -1.0


def test_closed_position_is_not_valued():
    ledger = _ledger()
    _buy(ledger)
    ledger.record(
        Transaction(
            date=date(2024, 1, 9),
            account=TFSA,
            kind=TransactionKind.SELL,
            ticker="VFV.TO",
            quantity=10.0,
            price_per_share=120.0,
        )
    )

    valuation = ledger.valuation({"VFV.TO": 130.0})

    assert valuation.positions == ()
    assert valuation.total_market_value == 10_200.0  # 9000 cash + 1200 proceeds


# ---------------------------------------------------------------------------
# Unknown prices stay unknown
# ---------------------------------------------------------------------------


def test_missing_price_is_unknown_not_zero():
    ledger = _ledger()
    _buy(ledger, ticker="VFV.TO")
    _buy(ledger, ticker="VUN.TO", quantity=10.0, price=50.0, day=3)

    valuation = ledger.valuation({"VFV.TO": 110.0})

    assert valuation.unpriced_tickers == ("VUN.TO",)
    assert not valuation.is_complete
    assert [position.ticker for position in valuation.positions] == ["VFV.TO"]
    # Only the priced holding is counted; VUN.TO is not valued at zero.
    # Cash: 10_000 - 1_000 (VFV) - 500 (VUN) = 8_500.
    assert valuation.total_market_value == 1100.0 + 8500.0
    assert valuation.total_book_cost == 1000.0


def test_nan_price_is_unknown_not_zero():
    ledger = _ledger()
    _buy(ledger)

    valuation = ledger.valuation({"VFV.TO": float("nan")})

    assert valuation.unpriced_tickers == ("VFV.TO",)
    assert valuation.positions == ()
    assert not valuation.is_complete


def test_price_for_an_unheld_ticker_is_ignored():
    ledger = _ledger()
    _buy(ledger)

    valuation = ledger.valuation({"VFV.TO": 110.0, "VEE.TO": float("nan")})

    assert valuation.is_complete
    assert valuation.unpriced_tickers == ()


def test_unpriced_security_shrinks_allocation_denominators_but_is_not_dropped():
    ledger = _ledger()
    _buy(ledger, ticker="VFV.TO")
    _buy(ledger, ticker="VUN.TO", day=3)

    valuation = ledger.valuation({"VFV.TO": 100.0})

    # 1000 priced + 8000 cash; VUN.TO's 1000 ACB is untouched.
    assert valuation.allocation_by_ticker() == {"VFV.TO": 1000.0 / 9000.0}


@pytest.mark.parametrize("price", [-1.0, float("inf"), "110.0", True])
def test_unusable_prices_are_rejected(price: object):
    ledger = _ledger()
    _buy(ledger)

    with pytest.raises(DataValidationError, match="price for VFV.TO"):
        ledger.valuation({"VFV.TO": price})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Per-account structure and allocation
# ---------------------------------------------------------------------------


def test_same_ticker_in_two_accounts_is_valued_separately():
    ledger = PortfolioLedger()
    for account, price in ((TFSA, 100.0), (NON_REG, 200.0)):
        ledger.record(
            Transaction(
                date=date(2024, 1, 1),
                account=account,
                kind=TransactionKind.CONTRIBUTION,
                amount=5000.0,
            )
        )
        _buy(ledger, quantity=10.0, price=price, account=account)

    valuation = ledger.valuation({"VFV.TO": 250.0})

    assert [position.book_cost for position in valuation.positions] == [
        2000.0,
        1000.0,
    ]
    assert valuation.total_unrealized_gain_loss == (2500.0 - 2000.0) + (2500.0 - 1000.0)
    # Combined for allocation: 5_000 CAD of VFV.TO out of 12_000 total
    # (5_000 of positions plus 7_000 of cash).
    assert valuation.allocation_by_ticker() == {"VFV.TO": 5000.0 / 12_000.0}


def test_account_values_include_cash():
    ledger = PortfolioLedger()
    ledger.record(
        Transaction(
            date=date(2024, 1, 1),
            account=TFSA,
            kind=TransactionKind.CONTRIBUTION,
            amount=5000.0,
        )
    )
    ledger.record(
        Transaction(
            date=date(2024, 1, 1),
            account=NON_REG,
            kind=TransactionKind.CONTRIBUTION,
            amount=1000.0,
        )
    )
    _buy(ledger)

    valuation = ledger.valuation({"VFV.TO": 110.0})

    assert valuation.account_values() == {TFSA_ACCOUNT: 5100.0, NON_REG_ACCOUNT: 1000.0}
    assert valuation.allocation_by_account() == {
        TFSA_ACCOUNT: 5100.0 / 6100.0,
        NON_REG_ACCOUNT: 1000.0 / 6100.0,
    }
    assert sum(valuation.allocation_by_account().values()) == pytest.approx(1.0)


def test_allocations_are_empty_when_the_portfolio_is_worth_nothing():
    ledger = _ledger(1000.0)
    _buy(ledger, quantity=10.0, price=100.0)

    valuation = ledger.valuation({"VFV.TO": 0.0})

    assert valuation.total_market_value == 0.0
    assert valuation.allocation_by_account() == {}
    assert valuation.allocation_by_ticker() == {}


def test_positions_are_sorted_by_account_then_ticker():
    ledger = PortfolioLedger()
    for account in (TFSA, NON_REG):
        ledger.record(
            Transaction(
                date=date(2024, 1, 1),
                account=account,
                kind=TransactionKind.CONTRIBUTION,
                amount=10_000.0,
            )
        )
        _buy(ledger, ticker="VFV.TO", account=account)
        _buy(ledger, ticker="VDY.TO", quantity=1.0, price=10.0, day=3, account=account)

    valuation = ledger.valuation({"VFV.TO": 100.0, "VDY.TO": 10.0})

    assert [(p.account, p.ticker) for p in valuation.positions] == [
        (NON_REG_ACCOUNT, "VDY.TO"),
        (NON_REG_ACCOUNT, "VFV.TO"),
        (TFSA_ACCOUNT, "VDY.TO"),
        (TFSA_ACCOUNT, "VFV.TO"),
    ]


def test_valuation_does_not_mutate_the_ledger():
    ledger = _ledger()
    _buy(ledger)
    before = (ledger.positions, ledger.cash_balances, ledger.transactions)

    ledger.valuation({"VFV.TO": 110.0})
    ledger.valuation({})

    assert (ledger.positions, ledger.cash_balances, ledger.transactions) == before


# ---------------------------------------------------------------------------
# The pure function
# ---------------------------------------------------------------------------


def test_value_portfolio_requires_cost_basis_for_every_holding():
    with pytest.raises(DataValidationError, match="no cost basis for VFV.TO"):
        value_portfolio(
            {(TFSA_ACCOUNT, "VFV.TO"): 10.0}, {TFSA_ACCOUNT: 0.0}, {"VFV.TO": 110.0}, {}
        )


def test_zero_cost_basis_reports_full_market_value_as_a_gain():
    valuation = value_portfolio(
        {(TFSA_ACCOUNT, "VFV.TO"): 10.0},
        {TFSA_ACCOUNT: 0.0},
        {"VFV.TO": 110.0},
        {(TFSA_ACCOUNT, "VFV.TO"): 0.0},
    )

    (position,) = valuation.positions
    assert position.unrealized_gain_loss == 1100.0
    assert position.unrealized_gain_loss_pct == 0.0


@pytest.mark.parametrize("cash", [float("nan"), float("inf"), "1000"])
def test_non_finite_cash_is_rejected(cash: object):
    with pytest.raises(DataValidationError, match="cash for TFSA"):
        value_portfolio({}, {TFSA_ACCOUNT: cash}, {}, {})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Reporting surface
# ---------------------------------------------------------------------------


def test_to_frame_reports_one_row_per_position():
    ledger = _ledger()
    _buy(ledger, ticker="VFV.TO")
    _buy(ledger, ticker="VDY.TO", quantity=2.0, price=10.0, day=3)

    frame = ledger.valuation({"VFV.TO": 110.0, "VDY.TO": 12.0}).to_frame()

    assert list(frame.columns) == [
        "account",
        "registration",
        "ticker",
        "quantity",
        "market_price",
        "market_value",
        "book_cost",
        "unrealized_gain_loss",
    ]
    assert frame["account"].tolist() == ["TFSA", "TFSA"]
    assert frame["ticker"].tolist() == ["VDY.TO", "VFV.TO"]
    assert frame["market_value"].tolist() == [24.0, 1100.0]
    assert frame["unrealized_gain_loss"].tolist() == [4.0, 100.0]


def test_empty_frame_keeps_its_columns():
    frame = PortfolioLedger().valuation({}).to_frame()
    assert frame.empty
    assert "market_value" in frame.columns


def test_valuation_symbols_are_lazily_exported():
    names = {"PositionValuation", "Valuation", "value_portfolio"}

    assert names.issubset(pysharpe.__all__)
    assert all(getattr(pysharpe, name) is not None for name in names)
