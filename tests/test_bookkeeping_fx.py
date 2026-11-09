"""Tests for multi-currency cash accounting in the bookkeeping ledger.

A Canadian account can hold several currencies at once.  The rules under test:

* cash is held per ``(account, currency)`` in native units and is never netted
  across currencies;
* a foreign-currency trade settles from that currency's own balance, so nothing
  converts automatically;
* moving money between currencies is an explicit ``FX_CONVERSION`` carrying the
  source amount, the destination amount and the rate it achieved;
* only ``CONTRIBUTION``/``WITHDRAWAL`` are external flows, so an internal
  conversion never changes contribution totals;
* each transaction keeps its own ``fx_rate`` as book-cost evidence.

Every expected figure is hand-computed from the documented conventions.
"""

from __future__ import annotations

from datetime import date

import pytest

from pysharpe.bookkeeping import (
    BASE_CURRENCY,
    Account,
    Currency,
    PortfolioLedger,
    Transaction,
    TransactionKind,
)
from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import AccountType

TFSA = AccountType.TFSA
NON_REG = AccountType.NON_REG
TFSA_ACCOUNT = Account.from_registration(TFSA)
NON_REG_ACCOUNT = Account.from_registration(NON_REG)

#: The CAD/USD rate used throughout: CAD units per one USD unit.
RATE = 1.35


# ---------------------------------------------------------------------------
# Journal helpers
# ---------------------------------------------------------------------------


def contribute(amount: float, day: int = 1, account: AccountType = TFSA):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.CONTRIBUTION,
        amount=amount,
    )


def withdraw(amount: float, day: int = 5, account: AccountType = TFSA):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.WITHDRAWAL,
        amount=amount,
    )


def conversion(
    cad: float = 6750.0,
    usd: float = 5000.0,
    day: int = 2,
    account: AccountType = TFSA,
    commission: float = 0.0,
):
    """CAD -> USD conversion of *cad* into *usd*."""
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.FX_CONVERSION,
        amount=cad,
        destination_currency=Currency.USD,
        destination_amount=usd,
        commission=commission,
    )


def buy(
    quantity: float = 10.0,
    price: float = 50.0,
    day: int = 3,
    account: AccountType = TFSA,
    ticker: str = "VOO",
):
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.BUY,
        ticker=ticker,
        quantity=quantity,
        price_per_share=price,
        currency=Currency.USD,
        fx_rate=RATE,
    )


def _scenario_ledger() -> PortfolioLedger:
    """Contribute CAD 10,000, convert 6,750 CAD to 5,000 USD, buy 10 shares."""
    ledger = PortfolioLedger()
    ledger.extend([contribute(10_000.0), conversion(), buy()])
    return ledger


# ---------------------------------------------------------------------------
# The milestone scenario
# ---------------------------------------------------------------------------


def test_scenario_balances_are_per_currency_and_never_netted():
    ledger = _scenario_ledger()

    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(3250.0)
    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == pytest.approx(4500.0)
    assert ledger.cash(TFSA_ACCOUNT) == pytest.approx(3250.0)  # CAD by default
    assert ledger.positions == {(TFSA_ACCOUNT, "VOO"): 10.0}


def test_scenario_books_the_security_at_its_own_acquisition_rate():
    ledger = _scenario_ledger()

    # 10 shares x USD 50 x 1.35 CAD/USD.
    assert ledger.book_cost(TFSA_ACCOUNT, "VOO") == pytest.approx(675.0)
    assert ledger.book_cost_per_share(TFSA_ACCOUNT, "VOO") == pytest.approx(67.5)


def test_scenario_counts_only_the_real_external_flow():
    ledger = _scenario_ledger()

    assert ledger.contributed(TFSA_ACCOUNT) == pytest.approx(10_000.0)
    assert ledger.withdrawn(TFSA_ACCOUNT) == 0.0
    assert ledger.total_contributions == pytest.approx(10_000.0)
    assert ledger.total_withdrawals == 0.0
    # The conversion is recorded as its own kind, not as a contribution.
    assert ledger.totals_by_kind[TransactionKind.FX_CONVERSION] == pytest.approx(6750.0)


def test_conversion_records_both_amounts_and_its_effective_rate():
    txn = conversion()

    assert txn.cash_balance_delta == {Currency.CAD: -6750.0, Currency.USD: 5000.0}
    assert txn.effective_rate == pytest.approx(0.7407407407407407)  # USD per CAD
    assert txn.destination_rate_to_base == pytest.approx(RATE)  # CAD per USD
    assert txn.position_delta == 0.0
    # An internal transfer whose rates agree nets to zero in base currency.
    assert txn.cash_flow_base == pytest.approx(0.0)


def test_usd_to_cad_conversion_uses_the_same_evidence():
    txn = Transaction(
        date=date(2024, 1, 4),
        account=TFSA,
        kind=TransactionKind.FX_CONVERSION,
        amount=5000.0,
        currency=Currency.USD,
        fx_rate=RATE,
        destination_currency=Currency.CAD,
        destination_amount=6750.0,
    )

    assert txn.cash_balance_delta == {Currency.USD: -5000.0, Currency.CAD: 6750.0}
    assert txn.destination_rate_to_base == pytest.approx(1.0)
    assert txn.effective_rate == pytest.approx(1.35)  # CAD per USD
    assert txn.cash_flow_base == pytest.approx(0.0)


def test_conversion_never_changes_holdings_or_book_cost():
    ledger = PortfolioLedger()
    ledger.extend([contribute(10_000.0), conversion()])

    assert ledger.positions == {}
    assert ledger.total_commissions == 0.0
    assert ledger.realized_gain_loss(TFSA) == 0.0


# ---------------------------------------------------------------------------
# Each currency is its own balance
# ---------------------------------------------------------------------------


def test_insufficient_usd_cash_rejects_the_purchase_and_leaves_state_untouched():
    ledger = PortfolioLedger()
    ledger.extend([contribute(10_000.0), conversion()])
    before = ledger.transactions, ledger.positions, ledger.cash_balances

    # USD 5,000 held; this order needs 6,000.
    with pytest.raises(DataValidationError, match="TFSA USD cash would fall to"):
        ledger.record(buy(quantity=120.0))

    assert (ledger.transactions, ledger.positions, ledger.cash_balances) == before
    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == pytest.approx(5000.0)
    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(3250.0)


def test_a_cad_balance_never_covers_a_usd_deficit():
    ledger = PortfolioLedger()
    ledger.record(contribute(100_000.0))

    with pytest.raises(DataValidationError, match="USD cash would fall to"):
        ledger.record(buy(quantity=1.0))

    assert ledger.transactions == (contribute(100_000.0),)
    assert ledger.cash(TFSA_ACCOUNT) == pytest.approx(100_000.0)


def test_usd_dividends_credit_usd_cash_only():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribute(10_000.0),
            conversion(),
            Transaction(
                date=date(2024, 1, 4),
                account=TFSA,
                kind=TransactionKind.DIVIDEND,
                ticker="VOO",
                amount=12.34,
                currency=Currency.USD,
                fx_rate=1.36,
            ),
        ]
    )

    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == pytest.approx(5012.34)
    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(3250.0)
    assert ledger.totals_by_kind[TransactionKind.DIVIDEND] == pytest.approx(
        12.34 * 1.36
    )
    assert ledger.contributed(TFSA_ACCOUNT) == pytest.approx(10_000.0)


def test_usd_withdrawal_is_an_external_flow_from_usd_cash():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribute(10_000.0),
            conversion(),
            Transaction(
                date=date(2024, 1, 5),
                account=TFSA,
                kind=TransactionKind.WITHDRAWAL,
                amount=500.0,
                currency=Currency.USD,
                fx_rate=RATE,
            ),
        ]
    )

    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == pytest.approx(4500.0)
    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(3250.0)
    assert ledger.withdrawn(TFSA_ACCOUNT) == pytest.approx(500.0 * RATE)
    assert ledger.total_withdrawals == pytest.approx(500.0 * RATE)


def test_overspending_the_source_currency_is_rejected():
    ledger = PortfolioLedger()
    ledger.record(contribute(10_000.0))

    with pytest.raises(
        DataValidationError,
        match="CAD cash would fall to -1",
    ):
        ledger.record(conversion(cad=20_000.0))

    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(10_000.0)
    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == 0.0


# ---------------------------------------------------------------------------
# Explicit conversion fees
# ---------------------------------------------------------------------------


def test_an_explicit_fee_is_charged_once_on_the_source_side():
    ledger = PortfolioLedger()
    ledger.extend([contribute(10_000.0), conversion(commission=5.0)])

    # 6,750 converted + 5.00 fee; the USD receipt is untouched by the fee.
    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(3245.0)
    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == pytest.approx(5000.0)
    assert ledger.total_commissions == pytest.approx(5.0)
    # Base-currency view of the conversion: exactly the fee, not the principal
    # plus the fee.
    assert ledger.totals_by_kind[TransactionKind.FX_CONVERSION] == pytest.approx(6750.0)
    assert conversion(commission=5.0).cash_flow_base == pytest.approx(-5.0)


def test_a_conversion_fee_is_not_loaded_into_a_security_cost():
    ledger = PortfolioLedger()
    ledger.extend([contribute(10_000.0), conversion(commission=5.0), buy()])

    assert ledger.book_cost(TFSA_ACCOUNT, "VOO") == pytest.approx(675.0)
    assert ledger.total_commissions == pytest.approx(5.0)


def test_reversing_a_conversion_does_not_create_an_external_flow():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribute(10_000.0),
            conversion(),
            Transaction(
                date=date(2024, 1, 4),
                account=TFSA,
                kind=TransactionKind.FX_CONVERSION,
                amount=5000.0,
                currency=Currency.USD,
                fx_rate=RATE,
                destination_currency=Currency.CAD,
                destination_amount=6750.0,
            ),
        ]
    )

    assert ledger.cash(TFSA_ACCOUNT, Currency.CAD) == pytest.approx(10_000.0)
    assert ledger.cash(TFSA_ACCOUNT, Currency.USD) == pytest.approx(0.0)
    assert ledger.contributed(TFSA_ACCOUNT) == pytest.approx(10_000.0)
    assert ledger.withdrawn(TFSA_ACCOUNT) == 0.0
    assert ledger.total_contributions == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# FX evidence is per transaction
# ---------------------------------------------------------------------------


def test_a_later_conversion_does_not_rewrite_an_acquisition_rate():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribute(10_000.0),
            conversion(),
            Transaction(
                date=date(2024, 1, 3),
                account=TFSA,
                kind=TransactionKind.BUY,
                ticker="VOO",
                quantity=1.0,
                price_per_share=100.0,
                currency=Currency.USD,
                fx_rate=1.30,
            ),
            # A second conversion at a different rate arrives later.
            conversion(cad=675.0, usd=500.0, day=4),
        ]
    )

    assert ledger.book_cost(TFSA_ACCOUNT, "VOO") == pytest.approx(130.0)


def test_transactions_frame_exposes_conversion_evidence():
    frame = _scenario_ledger().transactions_frame()
    converted = frame.loc[frame["kind"] == TransactionKind.FX_CONVERSION].iloc[0]

    assert converted["currency"] == Currency.CAD
    assert converted["fx_rate"] == pytest.approx(1.0)
    assert converted["amount"] == pytest.approx(6750.0)
    assert converted["destination_currency"] == Currency.USD
    assert converted["destination_amount"] == pytest.approx(5000.0)
    assert converted["effective_rate"] == pytest.approx(5000.0 / 6750.0)
    purchase = frame.loc[frame["kind"] == TransactionKind.BUY].iloc[0]
    assert purchase["fx_rate"] == pytest.approx(RATE)
    assert purchase["effective_rate"] == 0.0


# ---------------------------------------------------------------------------
# Accounts stay isolated
# ---------------------------------------------------------------------------


def test_accounts_keep_independent_balances_in_each_currency():
    first = Account("TFSA-A", TFSA)
    second = Account("TFSA-B", TFSA)
    ledger = PortfolioLedger()
    ledger.extend(
        [
            Transaction(
                date=date(2024, 1, 1),
                account=first,
                kind=TransactionKind.CONTRIBUTION,
                amount=10_000.0,
            ),
            Transaction(
                date=date(2024, 1, 1),
                account=second,
                kind=TransactionKind.CONTRIBUTION,
                amount=10_000.0,
            ),
            Transaction(
                date=date(2024, 1, 2),
                account=first,
                kind=TransactionKind.FX_CONVERSION,
                amount=6750.0,
                destination_currency=Currency.USD,
                destination_amount=5000.0,
            ),
            Transaction(
                date=date(2024, 1, 2),
                account=second,
                kind=TransactionKind.FX_CONVERSION,
                amount=2700.0,
                destination_currency=Currency.USD,
                destination_amount=2000.0,
            ),
            Transaction(
                date=date(2024, 1, 3),
                account=first,
                kind=TransactionKind.BUY,
                ticker="VOO",
                quantity=10.0,
                price_per_share=50.0,
                currency=Currency.USD,
                fx_rate=RATE,
            ),
            Transaction(
                date=date(2024, 1, 3),
                account=second,
                kind=TransactionKind.BUY,
                ticker="VOO",
                quantity=4.0,
                price_per_share=100.0,
                currency=Currency.USD,
                fx_rate=RATE,
            ),
        ]
    )

    assert ledger.cash(first, Currency.CAD) == pytest.approx(3250.0)
    assert ledger.cash(first, Currency.USD) == pytest.approx(4500.0)
    assert ledger.cash(second, Currency.CAD) == pytest.approx(7300.0)
    assert ledger.cash(second, Currency.USD) == pytest.approx(1600.0)
    assert ledger.contributed(first) == pytest.approx(10_000.0)
    assert ledger.contributed(second) == pytest.approx(10_000.0)
    # First account's USD balance cannot fund an order in the second.
    with pytest.raises(DataValidationError, match="TFSA-B USD cash would fall to"):
        ledger.record(
            Transaction(
                date=date(2024, 1, 4),
                account=second,
                kind=TransactionKind.BUY,
                ticker="VOO",
                quantity=100.0,
                price_per_share=50.0,
                currency=Currency.USD,
                fx_rate=RATE,
            )
        )


def test_a_cad_only_journal_is_unchanged_and_needs_no_rate():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contribute(5000.0),
            Transaction(
                date=date(2024, 1, 2),
                account=TFSA,
                kind=TransactionKind.BUY,
                ticker="VFV.TO",
                quantity=10.0,
                price_per_share=100.0,
            ),
        ]
    )

    assert ledger.cash_balances == {(TFSA_ACCOUNT, BASE_CURRENCY): 4000.0}
    # No conversion kinds, and valuation needs no FX rate.
    assert ledger.totals_by_kind[TransactionKind.FX_CONVERSION] == 0.0
    assert ledger.valuation({"VFV.TO": 110.0}, {}).total_cash == 4000.0


# ---------------------------------------------------------------------------
# Valuation of foreign cash needs a current rate
# ---------------------------------------------------------------------------


def test_foreign_cash_needs_a_supplied_current_rate():
    ledger = _scenario_ledger()

    with pytest.raises(DataValidationError, match="no current USD->CAD rate"):
        ledger.valuation({})

    valuation = ledger.valuation({}, {Currency.USD: RATE})
    assert valuation.total_cash == pytest.approx(3250.0 + 4500.0 * RATE)


def test_valuation_does_not_accept_a_rate_for_the_base_currency():
    ledger = _scenario_ledger()

    with pytest.raises(DataValidationError, match="must not re-rate CAD"):
        ledger.valuation({}, {Currency.CAD: 1.35})


@pytest.mark.parametrize(
    ("rates", "message"),
    [
        ({"USD": 1.35}, "keys must be Currency members"),
        ({Currency.USD: float("nan")}, "must be finite and positive"),
        ({Currency.USD: float("inf")}, "must be finite and positive"),
        ({Currency.USD: 0.0}, "must be finite and positive"),
        ({Currency.USD: "1.35"}, "must be a number"),
    ],
)
def test_unusable_current_rates_are_rejected(rates: object, message: str):
    with pytest.raises(DataValidationError, match=message):
        _scenario_ledger().valuation({}, rates)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Malformed conversions are rejected at the record
# ---------------------------------------------------------------------------


def _conversion(**overrides):
    """Build a conversion, overriding any field, to test one rule at a time."""
    fields = {
        "date": date(2024, 1, 2),
        "account": TFSA,
        "kind": TransactionKind.FX_CONVERSION,
        "amount": 6750.0,
        "destination_currency": Currency.USD,
        "destination_amount": 5000.0,
        "commission": 0.0,
    }
    fields.update(overrides)
    return Transaction(**fields)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"amount": float("nan")}, "amount must be finite"),
        ({"amount": float("inf")}, "amount must be finite"),
        ({"amount": 0.0}, "requires a positive source amount"),
        ({"amount": -1.0}, "amount must be non-negative"),
        ({"destination_amount": float("nan")}, "destination_amount must be finite"),
        ({"destination_amount": float("inf")}, "destination_amount must be finite"),
        ({"destination_amount": 0.0}, "requires a positive destination_amount"),
        ({"destination_amount": -1.0}, "destination_amount must be non-negative"),
        ({"destination_currency": None}, "destination_currency must be one of"),
        ({"destination_currency": "EUR"}, "destination_currency must be one of"),
        ({"destination_currency": Currency.CAD}, "both sides are CAD"),
        ({"fx_rate": float("nan")}, "fx_rate must be finite"),
        ({"fx_rate": float("inf")}, "fx_rate must be finite"),
        ({"fx_rate": 0.0}, "fx_rate must be strictly positive"),
        ({"fx_rate": -1.35}, "fx_rate must be non-negative"),
        ({"fx_rate": 1.35}, "implicit rate of 1.0"),
        ({"commission": float("nan")}, "commission must be finite"),
        ({"commission": -1.0}, "commission must be non-negative"),
        ({"ticker": "VOO"}, "cannot name a ticker"),
        ({"quantity": 10.0}, "moves no shares"),
        ({"price_per_share": 50.0}, "moves no shares"),
    ],
)
def test_malformed_conversions_are_rejected(overrides: dict, message: str):
    with pytest.raises(DataValidationError, match=message):
        _conversion(**overrides)


def test_a_usd_sourced_conversion_still_needs_an_explicit_rate():
    with pytest.raises(DataValidationError, match="fx_rate is required for USD"):
        Transaction(
            date=date(2024, 1, 2),
            account=TFSA,
            kind=TransactionKind.FX_CONVERSION,
            amount=5000.0,
            currency=Currency.USD,
            destination_currency=Currency.CAD,
            destination_amount=6750.0,
        )


def test_destination_fields_are_rejected_on_other_kinds():
    with pytest.raises(DataValidationError, match="belong to FX_CONVERSION only"):
        Transaction(
            date=date(2024, 1, 2),
            account=TFSA,
            kind=TransactionKind.CONTRIBUTION,
            amount=1000.0,
            destination_currency=Currency.USD,
            destination_amount=740.0,
        )


def test_cash_kinds_may_not_carry_a_conversion_fee():
    with pytest.raises(DataValidationError, match="record trading cost as a FEE"):
        Transaction(
            date=date(2024, 1, 2),
            account=TFSA,
            kind=TransactionKind.DIVIDEND,
            amount=10.0,
            commission=1.0,
        )


# ---------------------------------------------------------------------------
# Replay determinism and atomic failure
# ---------------------------------------------------------------------------


def test_replay_is_deterministic_regardless_of_insertion_order():
    events = [contribute(10_000.0), conversion(), buy()]

    forward = PortfolioLedger()
    forward.extend(events)
    backward = PortfolioLedger()
    backward.extend(reversed(events))

    assert forward.cash_balances == backward.cash_balances
    assert forward.positions == backward.positions
    assert forward.contributed(TFSA_ACCOUNT) == backward.contributed(TFSA_ACCOUNT)


def test_a_conversion_dated_after_the_purchase_cannot_retroactively_fund_it():
    ledger = PortfolioLedger()
    ledger.record(contribute(10_000.0))

    # Replay sorts by date, so the day-3 purchase settles before the day-4
    # conversion that would have funded it.
    with pytest.raises(DataValidationError, match="USD cash would fall to"):
        ledger.extend([buy(day=3), conversion(day=4)])

    assert ledger.transactions == (contribute(10_000.0),)
    assert ledger.cash_balances == {(TFSA_ACCOUNT, Currency.CAD): 10_000.0}


def test_a_rejected_batch_rolls_back_everything():
    ledger = PortfolioLedger()
    ledger.extend([contribute(10_000.0), conversion()])
    before = ledger.transactions, ledger.positions, ledger.cash_balances

    with pytest.raises(DataValidationError):
        ledger.extend([buy(quantity=10.0), buy(quantity=1000.0, day=4)])

    assert (ledger.transactions, ledger.positions, ledger.cash_balances) == before
