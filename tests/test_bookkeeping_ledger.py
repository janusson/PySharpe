"""Tests for multi-account portfolio bookkeeping (journal, holdings, cash).

All values are hand-computed from the documented conventions:

* base currency is CAD;
* ``fx_rate`` is CAD per one unit of the transaction currency;
* monetary fields are non-negative and direction comes from the kind;
* replay order is ``(date, insertion order)``.
"""

from __future__ import annotations

from datetime import date

import pytest

import pysharpe
from pysharpe.bookkeeping import (
    BASE_CURRENCY,
    Account,
    Currency,
    PortfolioLedger,
    Position,
    Transaction,
    TransactionKind,
)
from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import AccountType

TFSA = AccountType.TFSA
NON_REG = AccountType.NON_REG
TFSA_ACCOUNT = Account.from_registration(TFSA)
NON_REG_ACCOUNT = Account.from_registration(NON_REG)


def contribution(amount: float, day: int = 1, account: AccountType = TFSA):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.CONTRIBUTION,
        amount=amount,
    )


def buy(
    ticker: str = "VFV.TO",
    quantity: float = 10.0,
    price: float = 100.0,
    day: int = 2,
    account: AccountType = TFSA,
    **kwargs,
):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.BUY,
        ticker=ticker,
        quantity=quantity,
        price_per_share=price,
        **kwargs,
    )


def sell(
    ticker: str = "VFV.TO",
    quantity: float = 10.0,
    price: float = 100.0,
    day: int = 3,
    account: AccountType = TFSA,
    **kwargs,
):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.SELL,
        ticker=ticker,
        quantity=quantity,
        price_per_share=price,
        **kwargs,
    )


def convert(
    source_amount: float = 6750.0,
    destination_amount: float = 5000.0,
    day: int = 1,
    account: AccountType = TFSA,
    currency: Currency = Currency.CAD,
    destination_currency: Currency = Currency.USD,
    fx_rate: float | None = None,
    commission: float = 0.0,
):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.FX_CONVERSION,
        amount=source_amount,
        currency=currency,
        fx_rate=fx_rate,
        destination_currency=destination_currency,
        destination_amount=destination_amount,
        commission=commission,
    )


# ---------------------------------------------------------------------------
# Conventions on the record itself
# ---------------------------------------------------------------------------


def test_cad_is_the_base_currency():
    assert BASE_CURRENCY is Currency.CAD


def test_buy_gross_and_cash_flow_are_derived_from_quantity_and_price():
    txn = buy(quantity=10, price=100.0)
    assert txn.gross_amount_base == 1000.0
    assert txn.cash_flow_base == -1000.0
    assert txn.position_delta == 10.0


def test_sell_reverses_the_share_sign_not_the_amount_sign():
    txn = sell(quantity=4, price=120.0)
    assert txn.gross_amount_base == 480.0
    assert txn.cash_flow_base == 480.0
    assert txn.position_delta == -4.0


def test_cash_only_kinds_carry_no_shares():
    txn = Transaction(
        date=date(2024, 1, 5),
        account=TFSA,
        kind=TransactionKind.DIVIDEND,
        ticker="VDY.TO",
        amount=12.5,
    )
    assert txn.position_delta == 0.0
    assert txn.cash_flow_base == 12.5


def test_usd_transaction_requires_an_explicit_rate():
    with pytest.raises(DataValidationError, match="fx_rate is required for USD"):
        buy(currency=Currency.USD)


def test_usd_transaction_converts_at_the_recorded_rate():
    txn = buy(quantity=10, price=50.0, currency=Currency.USD, fx_rate=1.35)
    assert txn.gross_amount_base == pytest.approx(675.0)
    assert txn.cash_flow_base == pytest.approx(-675.0)
    assert txn.rate_to_base == pytest.approx(1.35)


def test_cad_transaction_rejects_a_non_unit_rate():
    with pytest.raises(DataValidationError, match="implicit rate of 1.0"):
        buy(fx_rate=1.35)


def test_cad_transaction_accepts_an_explicit_unit_rate():
    assert buy(fx_rate=1.0).rate_to_base == 1.0


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_foreign_rate_must_be_positive_and_finite(rate: float):
    with pytest.raises(DataValidationError):
        buy(currency=Currency.USD, fx_rate=rate)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantity": 0.0}, "requires a positive quantity"),
        ({"price": 0.0}, "requires a positive price_per_share"),
        ({"amount": 5.0}, "derives its cash amount"),
        ({"ticker": None}, "BUY requires a ticker"),
        ({"commission": -1.0}, "commission must be non-negative"),
    ],
)
def test_traded_kinds_reject_inconsistent_records(kwargs: dict, message: str):
    with pytest.raises(DataValidationError, match=message):
        buy(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"amount": 0.0}, "requires a positive amount"),
        ({"quantity": 1.0}, "moves no shares"),
        ({"commission": 1.0}, "record trading cost as a FEE"),
        ({"ticker": "VFV.TO"}, "cannot name a ticker"),
        ({"fx_rate": float("nan")}, "fx_rate must be finite"),
    ],
)
def test_cash_kinds_reject_inconsistent_records(kwargs: dict, message: str):
    fields = {
        "date": date(2024, 1, 5),
        "account": TFSA,
        "kind": TransactionKind.FEE,
        "amount": 9.99,
        **kwargs,
    }
    with pytest.raises(DataValidationError, match=message):
        Transaction(**fields)


def test_return_of_capital_requires_a_ticker():
    with pytest.raises(DataValidationError, match="requires a ticker"):
        Transaction(
            date=date(2024, 1, 5),
            account=TFSA,
            kind=TransactionKind.RETURN_OF_CAPITAL,
            amount=10.0,
        )


def test_account_labels_accept_member_names_and_reject_aliases():
    assert buy(account="tfsa").account == TFSA_ACCOUNT
    assert (
        Transaction(
            date=date(2024, 1, 2),
            account=TFSA,
            kind="buy",
            ticker="VFV.TO",
            quantity=1,
            price_per_share=10.0,
        ).kind
        is TransactionKind.BUY
    )
    with pytest.raises(DataValidationError, match="account must be an AccountType"):
        buy(account="Non-Reg")


def test_ticker_is_trimmed_and_blank_becomes_none():
    assert buy(ticker="  VFV.TO ").ticker == "VFV.TO"


def test_unknown_kind_is_rejected():
    with pytest.raises(DataValidationError, match="kind must be one of"):
        Transaction(
            date=date(2024, 1, 2),
            account=TFSA,
            kind="TRANSFER",
            amount=1.0,
        )


def test_non_numeric_amount_is_rejected():
    with pytest.raises(DataValidationError, match="amount must be a number"):
        contribution("5000")


# ---------------------------------------------------------------------------
# Replay into holdings and cash
# ---------------------------------------------------------------------------


def test_empty_ledger_has_no_state():
    ledger = PortfolioLedger()
    assert ledger.transactions == ()
    assert ledger.positions == {}
    assert ledger.cash_balances == {}
    assert ledger.total_commissions == 0.0
    assert all(total == 0.0 for total in ledger.totals_by_kind.values())
    assert ledger.holdings() == []


def test_contribution_then_buy_settles_cash_and_holdings():
    ledger = PortfolioLedger()
    ledger.extend([contribution(5000.0), buy(quantity=10, price=100.0)])
    assert ledger.positions == {(TFSA_ACCOUNT, "VFV.TO"): 10.0}
    assert ledger.cash_balances == {(TFSA_ACCOUNT, Currency.CAD): 4000.0}
    assert ledger.holdings() == [
        Position(account=TFSA_ACCOUNT, ticker="VFV.TO", quantity=10.0)
    ]


def test_partial_sale_keeps_remaining_shares_and_adds_proceeds():
    ledger = PortfolioLedger()
    ledger.extend([contribution(5000.0), buy(), sell(quantity=4, price=120.0)])
    assert ledger.positions[(TFSA_ACCOUNT, "VFV.TO")] == 6.0
    assert ledger.cash(TFSA_ACCOUNT) == 4480.0


def test_selling_the_whole_position_removes_it():
    ledger = PortfolioLedger()
    ledger.extend([contribution(1000.0), buy(quantity=10), sell(quantity=10)])
    assert ledger.positions == {}
    assert ledger.cash(TFSA_ACCOUNT) == 1000.0


def test_fractional_shares_are_tracked_exactly():
    ledger = PortfolioLedger()
    ledger.extend([contribution(1000.0), buy(quantity=0.5, price=100.0)])
    assert ledger.positions[(TFSA_ACCOUNT, "VFV.TO")] == 0.5
    assert ledger.cash(TFSA_ACCOUNT) == 950.0


def test_commissions_are_added_to_purchases_and_deducted_from_proceeds():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(2000.0),
            buy(quantity=10, price=100.0, commission=5.0),
            sell(quantity=10, price=100.0, commission=5.0),
        ]
    )
    assert ledger.cash(TFSA_ACCOUNT) == 1990.0
    assert ledger.total_commissions == 10.0


def test_accounts_are_independent():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(2000.0, account=TFSA),
            contribution(2000.0, account=NON_REG),
            buy(quantity=10, account=TFSA),
            buy(quantity=10, account=NON_REG),
            sell(quantity=10, account=TFSA),
        ]
    )
    assert ledger.positions == {(NON_REG_ACCOUNT, "VFV.TO"): 10.0}
    # The TFSA sale at its purchase price returns the original outflow.
    assert ledger.cash_balances == {
        (TFSA_ACCOUNT, Currency.CAD): 2000.0,
        (NON_REG_ACCOUNT, Currency.CAD): 1000.0,
    }


def test_holdings_are_sorted_by_account_then_ticker():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(10_000.0, account=TFSA),
            contribution(10_000.0, account=NON_REG),
            buy(ticker="VDY.TO", quantity=1, price=10.0, account=TFSA),
            buy(ticker="VFV.TO", quantity=1, price=10.0, account=TFSA),
            buy(ticker="VFV.TO", quantity=1, price=10.0, account=NON_REG),
        ]
    )
    assert ledger.holdings() == [
        Position(account=NON_REG_ACCOUNT, ticker="VFV.TO", quantity=1.0),
        Position(account=TFSA_ACCOUNT, ticker="VDY.TO", quantity=1.0),
        Position(account=TFSA_ACCOUNT, ticker="VFV.TO", quantity=1.0),
    ]


def test_cash_events_are_totalled_per_kind():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(1000.0),
            Transaction(
                date=date(2024, 1, 5),
                account=TFSA,
                kind=TransactionKind.DIVIDEND,
                ticker="VDY.TO",
                amount=20.0,
            ),
            Transaction(
                date=date(2024, 1, 6),
                account=TFSA,
                kind=TransactionKind.INTEREST,
                amount=5.0,
            ),
            Transaction(
                date=date(2024, 1, 7),
                account=TFSA,
                kind=TransactionKind.RETURN_OF_CAPITAL,
                ticker="ZWC.TO",
                amount=15.0,
            ),
            Transaction(
                date=date(2024, 1, 8),
                account=TFSA,
                kind=TransactionKind.FEE,
                amount=2.5,
            ),
            Transaction(
                date=date(2024, 1, 9),
                account=TFSA,
                kind=TransactionKind.WITHDRAWAL,
                amount=100.0,
            ),
        ]
    )
    assert ledger.cash(TFSA_ACCOUNT) == 937.5
    totals = ledger.totals_by_kind
    assert totals[TransactionKind.CONTRIBUTION] == 1000.0
    assert totals[TransactionKind.DIVIDEND] == 20.0
    assert totals[TransactionKind.INTEREST] == 5.0
    assert totals[TransactionKind.RETURN_OF_CAPITAL] == 15.0
    assert totals[TransactionKind.FEE] == 2.5
    assert totals[TransactionKind.WITHDRAWAL] == 100.0
    assert totals[TransactionKind.BUY] == 0.0


def test_return_of_capital_leaves_the_share_count_unchanged():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(1000.0),
            buy(quantity=10),
            Transaction(
                date=date(2024, 1, 9),
                account=TFSA,
                kind=TransactionKind.RETURN_OF_CAPITAL,
                ticker="VFV.TO",
                amount=25.0,
            ),
        ]
    )
    assert ledger.positions[(TFSA_ACCOUNT, "VFV.TO")] == 10.0
    assert ledger.cash(TFSA_ACCOUNT) == 25.0


def test_usd_purchase_without_usd_cash_is_rejected_even_with_cad_on_hand():
    # A CAD balance never funds a USD trade: nothing converts automatically.
    ledger = PortfolioLedger()
    ledger.record(contribution(1000.0))
    before = ledger.positions, ledger.cash_balances, ledger.transactions

    with pytest.raises(DataValidationError, match="TFSA USD cash would fall to"):
        ledger.record(buy(quantity=10, price=50.0, currency=Currency.USD, fx_rate=1.35))

    assert (ledger.positions, ledger.cash_balances, ledger.transactions) == before
    assert ledger.cash(TFSA_ACCOUNT) == 1000.0
    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == 0.0


# ---------------------------------------------------------------------------
# Invariants and ordering
# ---------------------------------------------------------------------------


def test_overselling_is_rejected_and_state_is_unchanged():
    ledger = PortfolioLedger()
    ledger.extend([contribution(1000.0), buy(quantity=10)])
    before = ledger.positions, ledger.cash_balances, ledger.transactions

    with pytest.raises(DataValidationError, match="only 10 held"):
        ledger.record(sell(quantity=11))

    assert (ledger.positions, ledger.cash_balances, ledger.transactions) == before


def test_negative_cash_is_rejected_by_default():
    ledger = PortfolioLedger()
    with pytest.raises(DataValidationError, match="CAD cash would fall to -50.00"):
        ledger.record(
            Transaction(
                date=date(2024, 1, 5),
                account=TFSA,
                kind=TransactionKind.FEE,
                amount=50.0,
            )
        )
    assert ledger.transactions == ()
    assert ledger.cash_balances == {}


def test_negative_cash_can_be_allowed_explicitly():
    ledger = PortfolioLedger(allow_negative_cash=True)
    ledger.record(
        Transaction(
            date=date(2024, 1, 5),
            account=TFSA,
            kind=TransactionKind.FEE,
            amount=50.0,
        )
    )
    assert ledger.cash(TFSA_ACCOUNT) == -50.0


def test_batch_is_replayed_in_date_order_not_insertion_order():
    # Inserted sell-first the batch would oversell; sorted by date it is valid.
    ledger = PortfolioLedger()
    ledger.extend(
        [
            sell(quantity=4, price=120.0, day=9),
            contribution(2000.0, day=1),
            buy(day=5),
        ]
    )
    assert ledger.positions[(TFSA_ACCOUNT, "VFV.TO")] == 6.0
    assert ledger.cash(TFSA_ACCOUNT) == 1480.0
    assert [txn.date.day for txn in ledger.ordered_transactions] == [1, 5, 9]


def test_rejected_transaction_is_not_journaled():
    ledger = PortfolioLedger()
    with pytest.raises(DataValidationError, match="cash would fall"):
        ledger.record(buy(day=3))
    assert ledger.transactions == ()


def test_same_day_events_settle_in_insertion_order():
    ledger = PortfolioLedger()
    with pytest.raises(DataValidationError):
        ledger.record(buy(day=2))

    ledger.extend([contribution(2000.0, day=2), buy(day=2)])
    assert ledger.cash(TFSA_ACCOUNT) == 1000.0
    assert [txn.kind for txn in ledger.ordered_transactions] == [
        TransactionKind.CONTRIBUTION,
        TransactionKind.BUY,
    ]


def test_replay_result_does_not_depend_on_insertion_order():
    events = [contribution(2000.0, day=1), buy(day=5), sell(quantity=4, day=9)]

    forward = PortfolioLedger()
    forward.extend(events)

    backward = PortfolioLedger()
    backward.extend(reversed(events))

    assert forward.positions == backward.positions
    assert forward.cash_balances == backward.cash_balances
    assert forward.ordered_transactions == backward.ordered_transactions


def test_failed_batch_discards_every_transaction_in_it():
    ledger = PortfolioLedger()
    ledger.extend([contribution(1000.0), buy(quantity=10)])
    before = ledger.transactions, ledger.positions, ledger.cash_balances

    with pytest.raises(DataValidationError):
        ledger.extend([sell(quantity=1), sell(quantity=100, day=4)])

    assert (ledger.transactions, ledger.positions, ledger.cash_balances) == before


def test_extending_with_an_empty_batch_is_a_no_op():
    ledger = PortfolioLedger()
    ledger.extend([])
    assert ledger.transactions == ()
    assert ledger.positions == {}


def test_non_transaction_input_is_rejected():
    ledger = PortfolioLedger()
    with pytest.raises(DataValidationError, match="accepts Transaction records"):
        ledger.extend([contribution(100.0), "BUY VFV.TO"])  # type: ignore[list-item]
    assert ledger.transactions == ()


# ---------------------------------------------------------------------------
# Cost basis and realized gains (CAD)
# ---------------------------------------------------------------------------


def _funded_ledger(amount: float = 10_000.0) -> PortfolioLedger:
    ledger = PortfolioLedger()
    ledger.record(contribution(amount))
    return ledger


def test_commission_is_loaded_into_the_purchase_cost():
    """CRA: acquisition fees are added to the ACB, not expensed."""
    ledger = _funded_ledger()
    ledger.record(buy(quantity=100, price=10.0, commission=9.99))

    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(1009.99)
    assert ledger.book_cost_per_share(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(10.0999)
    # Cash and cost basis are related but not equal: the pool holds the fee.
    assert ledger.cash(TFSA_ACCOUNT) == pytest.approx(8990.01)


def test_second_purchase_repools_the_weighted_average():
    ledger = _funded_ledger()
    ledger.extend(
        [
            buy(quantity=100, price=10.0, commission=9.99, day=2),
            buy(quantity=100, price=12.0, commission=9.99, day=3),
        ]
    )

    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(2219.98)
    assert ledger.book_cost_per_share(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(11.0999)


def test_partial_sale_realizes_proceeds_less_acb_less_commission():
    """Selling half leaves the per-share ACB unchanged (ITA s. 47(1))."""
    ledger = _funded_ledger()
    ledger.extend(
        [
            buy(quantity=100, price=10.0, commission=9.99, day=2),
            buy(quantity=100, price=12.0, commission=9.99, day=3),
            sell(quantity=100, price=15.0, commission=9.99, day=4),
        ]
    )

    # proceeds 1500.00 - commission 9.99 - ACB sold 1109.99
    assert ledger.realized_gain_loss(TFSA) == pytest.approx(380.02)
    assert ledger.realized_gain_loss(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(380.02)
    assert ledger.positions[(TFSA_ACCOUNT, "VFV.TO")] == 100.0
    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(1109.99)
    assert ledger.book_cost_per_share(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(11.0999)


def test_return_of_capital_reduces_acb_without_realizing_a_gain():
    ledger = _funded_ledger()
    ledger.extend(
        [
            buy(quantity=100, price=10.0, commission=9.99),
            Transaction(
                date=date(2024, 2, 15),
                account=TFSA,
                kind=TransactionKind.RETURN_OF_CAPITAL,
                ticker="VFV.TO",
                amount=100.0,
            ),
        ]
    )

    assert ledger.positions[(TFSA_ACCOUNT, "VFV.TO")] == 100.0
    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(909.99)
    assert ledger.realized_gain_loss(TFSA) == 0.0


def test_return_of_capital_beyond_acb_is_realized():
    """ROC above the remaining ACB realizes the excess and zeroes the pool."""
    ledger = _funded_ledger()
    ledger.extend(
        [
            buy(quantity=100, price=10.0, commission=9.99),
            Transaction(
                date=date(2024, 2, 15),
                account=TFSA,
                kind=TransactionKind.RETURN_OF_CAPITAL,
                ticker="VFV.TO",
                amount=1200.0,
            ),
        ]
    )

    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == 0.0
    assert ledger.realized_gain_loss(TFSA) == pytest.approx(190.01)


def test_foreign_purchase_books_cad_cost_at_the_acquisition_rate():
    ledger = _funded_ledger()
    ledger.extend(
        [
            convert(),
            buy(quantity=10, price=50.0, day=2, currency=Currency.USD, fx_rate=1.35),
        ]
    )

    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(675.0)
    assert ledger.book_cost_per_share(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(67.5)


def test_foreign_sale_converts_proceeds_at_the_disposition_rate():
    """The currency effect lands in the realized gain, not in the book cost.

    ACB is fixed at the acquisition rate (675 CAD for 10 shares).  The sale
    settles at 1.40, so proceeds are 840 CAD and the gain is 165 CAD — it
    includes the CAD/USD move.
    """
    ledger = _funded_ledger()
    ledger.extend(
        [
            convert(),
            buy(quantity=10, price=50.0, day=2, currency=Currency.USD, fx_rate=1.35),
            sell(quantity=10, price=60.0, day=3, currency=Currency.USD, fx_rate=1.40),
        ]
    )

    assert ledger.realized_gain_loss(TFSA) == pytest.approx(165.0)
    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == 0.0


def test_foreign_commission_converts_at_the_trade_rate():
    ledger = _funded_ledger()
    ledger.extend(
        [
            convert(),
            buy(
                quantity=10,
                price=50.0,
                commission=1.0,
                day=2,
                currency=Currency.USD,
                fx_rate=1.35,
            ),
        ]
    )

    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == pytest.approx(676.35)


def test_cost_basis_pools_are_per_account():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(5000.0, account=TFSA),
            contribution(5000.0, account=NON_REG),
            buy(quantity=10, price=100.0, account=TFSA),
            buy(quantity=10, price=200.0, account=NON_REG),
            sell(quantity=10, price=100.0, account=TFSA, day=3),
        ]
    )

    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == 0.0
    assert ledger.book_cost(NON_REG_ACCOUNT, "VFV.TO") == pytest.approx(2000.0)
    assert ledger.realized_gain_loss(TFSA) == 0.0
    assert ledger.realized_gain_loss(NON_REG) == 0.0


def test_realized_gains_are_reported_per_ticker_and_per_account():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribution(20_000.0, account=NON_REG),
            buy(ticker="VFV.TO", quantity=10, price=100.0, account=NON_REG),
            buy(ticker="VDY.TO", quantity=10, price=50.0, account=NON_REG),
            sell(ticker="VFV.TO", quantity=10, price=110.0, account=NON_REG, day=3),
            sell(ticker="VDY.TO", quantity=10, price=40.0, account=NON_REG, day=4),
        ]
    )

    assert ledger.realized_gain_loss(NON_REG_ACCOUNT, "VFV.TO") == pytest.approx(100.0)
    assert ledger.realized_gain_loss(NON_REG, "VDY.TO") == pytest.approx(-100.0)
    assert ledger.realized_gain_loss(NON_REG) == pytest.approx(0.0)


def test_closed_position_reports_zero_quantity_and_keeps_its_realized_result():
    ledger = _funded_ledger()
    ledger.extend(
        [buy(quantity=10, price=100.0, day=2), sell(quantity=10, price=120.0, day=3)]
    )

    assert ledger.positions == {}
    assert ledger.book_cost(TFSA_ACCOUNT, "VFV.TO") == 0.0
    assert ledger.realized_gain_loss(TFSA) == pytest.approx(200.0)


def test_unseen_account_or_ticker_reports_zero():
    ledger = _funded_ledger()
    ledger.record(buy())

    assert ledger.book_cost(AccountType.RRSP, "VFV.TO") == 0.0
    assert ledger.book_cost_per_share(AccountType.RRSP, "VFV.TO") == 0.0
    assert ledger.realized_gain_loss(AccountType.RRSP) == 0.0
    assert ledger.book_cost(TFSA, "VUN.TO") == 0.0
    assert ledger.book_cost_per_share(TFSA, "VUN.TO") == 0.0
    assert ledger.realized_gain_loss(TFSA, "VUN.TO") == 0.0


def test_querying_an_unseen_ticker_does_not_create_a_report_row():
    ledger = _funded_ledger()
    ledger.record(buy())

    ledger.book_cost(TFSA, "VUN.TO")
    ledger.book_cost_per_share(TFSA, "VEE.TO")

    assert ledger.cost_basis_frame()["ticker"].tolist() == ["VFV.TO"]


def test_rejected_batch_leaves_cost_basis_unchanged():
    ledger = _funded_ledger()
    ledger.record(buy(quantity=10, price=100.0))
    before = (
        ledger.book_cost(TFSA_ACCOUNT, "VFV.TO"),
        ledger.realized_gain_loss(TFSA),
        ledger.positions,
    )

    with pytest.raises(DataValidationError):
        ledger.extend([sell(quantity=10, price=120.0, day=3), sell(quantity=1, day=4)])

    assert (
        ledger.book_cost(TFSA_ACCOUNT, "VFV.TO"),
        ledger.realized_gain_loss(TFSA),
        ledger.positions,
    ) == before


def test_cost_basis_frame_lists_every_touched_security():
    ledger = _funded_ledger(20_000.0)
    ledger.extend(
        [
            buy(ticker="VFV.TO", quantity=10, price=100.0, day=2),
            sell(ticker="VFV.TO", quantity=10, price=120.0, day=3),
            buy(ticker="VDY.TO", quantity=10, price=50.0, day=4),
        ]
    )

    frame = ledger.cost_basis_frame()

    assert list(frame.columns) == [
        "account",
        "registration",
        "ticker",
        "quantity",
        "book_cost",
        "book_cost_per_share",
        "realized_gain_loss",
    ]
    assert frame["account"].tolist() == ["TFSA", "TFSA"]
    assert frame["registration"].tolist() == [TFSA, TFSA]
    assert frame["ticker"].tolist() == ["VDY.TO", "VFV.TO"]
    assert frame["quantity"].tolist() == [10.0, 0.0]
    assert frame["book_cost"].tolist() == pytest.approx([500.0, 0.0])
    assert frame["realized_gain_loss"].tolist() == pytest.approx([0.0, 200.0])


def test_empty_cost_basis_frame_keeps_its_columns():
    frame = PortfolioLedger().cost_basis_frame()
    assert frame.empty
    assert "realized_gain_loss" in frame.columns


# ---------------------------------------------------------------------------
# Reporting surface and package wiring
# ---------------------------------------------------------------------------


def test_transactions_frame_reports_base_currency_columns():
    ledger = PortfolioLedger()
    ledger.extend([contribution(5000.0), buy(quantity=10, price=100.0)])

    frame = ledger.transactions_frame()

    assert list(frame.columns) == [
        "date",
        "account",
        "registration",
        "kind",
        "ticker",
        "quantity",
        "price_per_share",
        "currency",
        "fx_rate",
        "amount",
        "destination_currency",
        "destination_amount",
        "effective_rate",
        "commission",
        "gross_amount_base",
        "cash_flow_base",
    ]
    assert len(frame) == 2
    assert frame["cash_flow_base"].tolist() == [5000.0, -1000.0]
    assert frame["fx_rate"].tolist() == [1.0, 1.0]
    assert frame["destination_currency"].isna().all()
    assert frame["effective_rate"].tolist() == [0.0, 0.0]


def test_empty_ledger_frame_keeps_its_columns():
    frame = PortfolioLedger().transactions_frame()
    assert frame.empty
    assert "cash_flow_base" in frame.columns


def test_bookkeeping_symbols_are_lazily_exported():
    names = {
        "BASE_CURRENCY",
        "Currency",
        "PortfolioLedger",
        "Position",
        "Transaction",
        "TransactionKind",
    }
    assert names.issubset(pysharpe.__all__)
    assert all(getattr(pysharpe, name) is not None for name in names)
