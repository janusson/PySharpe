"""Tests for account identity and taxpayer-level taxable ACB pooling.

Two facts must stay separate:

* **identity** — which account a trade happened in, so two TFSAs never share
  shares, cash, or book cost;
* **registration** — the tax wrapper, which decides whether a holding joins the
  pooled taxable ACB (ITA s. 47(1) averages identical properties per *taxpayer*,
  not per account).

Expected figures are hand-calculated from the CRA weighted-average rule using
commission-free trades.
"""

from __future__ import annotations

from datetime import date

import pytest

from pysharpe.bookkeeping import (
    Account,
    Currency,
    PortfolioLedger,
    Transaction,
    TransactionKind,
)
from pysharpe.bookkeeping.accounts import account_of
from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import AccountType

TFSA = AccountType.TFSA
NON_REG = AccountType.NON_REG

TFSA_1 = Account("TFSA-1", TFSA)
TFSA_2 = Account("TFSA-2", TFSA)
NR_1 = Account("NonReg-1", NON_REG)
NR_2 = Account("NonReg-2", NON_REG)


def contrib(account: Account, amount: float, day: int = 1) -> Transaction:
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.CONTRIBUTION,
        amount=amount,
    )


def buy(account: Account, quantity: float, price: float, day: int = 2) -> Transaction:
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.BUY,
        ticker="VFV.TO",
        quantity=quantity,
        price_per_share=price,
    )


def sell(account: Account, quantity: float, price: float, day: int = 3) -> Transaction:
    return Transaction(
        date=date(2024, 1, day),
        account=account,
        kind=TransactionKind.SELL,
        ticker="VFV.TO",
        quantity=quantity,
        price_per_share=price,
    )


# ---------------------------------------------------------------------------
# The Account value type
# ---------------------------------------------------------------------------


def test_account_keeps_identity_and_registration_separate():
    account = Account("TFSA-2024", TFSA)

    assert account.id == "TFSA-2024"
    assert account.registration is TFSA
    assert not account.is_taxable


def test_only_non_registered_accounts_are_taxable():
    assert Account("m", NON_REG).is_taxable
    for registration in (TFSA, AccountType.FHSA, AccountType.RRSP, AccountType.LIRA):
        assert not Account("r", registration).is_taxable


def test_bare_registration_resolves_to_its_default_account():
    assert Account.from_registration(NON_REG) == Account("NON_REG", NON_REG)


def test_account_rejects_blank_identity_and_unknown_registration():
    with pytest.raises(DataValidationError, match="account id must be a non-empty"):
        Account("  ", TFSA)
    with pytest.raises(DataValidationError, match="account owner must be a non-empty"):
        Account("TFSA-1", TFSA, owner=" ")
    with pytest.raises(DataValidationError, match="account must be an AccountType"):
        Account("x", "Non-Reg")  # type: ignore[arg-type]


def test_accounts_are_value_objects():
    assert Account("a", TFSA) == Account("a", TFSA)
    assert Account("a", TFSA) != Account("a", NON_REG)
    assert Account("a", TFSA) != Account("a", TFSA, owner="spouse")
    assert len({Account("a", TFSA), Account("a", TFSA)}) == 1


def test_transaction_normalizes_a_bare_registration():
    transaction = Transaction(
        date=date(2024, 1, 1),
        account=TFSA,
        kind=TransactionKind.CONTRIBUTION,
        amount=100.0,
    )

    assert transaction.account == Account.from_registration(TFSA)
    assert transaction.registration is TFSA


def test_account_of_returns_an_account_unchanged():
    assert account_of(TFSA_1) is TFSA_1
    assert account_of(NON_REG) == Account.from_registration(NON_REG)
    assert account_of("fHSA") == Account("FHSA", AccountType.FHSA)


# ---------------------------------------------------------------------------
# Two accounts of one registration type stay independent
# ---------------------------------------------------------------------------


def test_two_tfsas_keep_separate_positions_cash_and_book_cost():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(TFSA_1, 5000.0),
            buy(TFSA_1, 10, 100.0),
            contrib(TFSA_2, 5000.0, day=3),
            buy(TFSA_2, 10, 200.0, day=4),
        ]
    )

    assert ledger.accounts == (TFSA_1, TFSA_2)
    assert ledger.positions == {(TFSA_1, "VFV.TO"): 10.0, (TFSA_2, "VFV.TO"): 10.0}
    assert ledger.cash_balances == {
        (TFSA_1, Currency.CAD): 4000.0,
        (TFSA_2, Currency.CAD): 3000.0,
    }
    assert ledger.book_cost(TFSA_1, "VFV.TO") == 1000.0
    assert ledger.book_cost(TFSA_2, "VFV.TO") == 2000.0
    assert ledger.book_cost_per_share(TFSA_1, "VFV.TO") == 100.0
    assert ledger.book_cost_per_share(TFSA_2, "VFV.TO") == 200.0


def test_sale_from_the_wrong_account_is_rejected_without_partial_mutation():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(TFSA_1, 1000.0),
            buy(TFSA_1, 10, 100.0),
            contrib(TFSA_2, 5000.0, day=3),
        ]
    )
    before = (
        ledger.transactions,
        ledger.positions,
        ledger.cash_balances,
        ledger.book_cost(TFSA_1, "VFV.TO"),
    )

    # TFSA-2 holds cash but no shares; TFSA-1's shares must not fund this sale.
    with pytest.raises(DataValidationError, match="in TFSA-2 \(TFSA\)"):
        ledger.record(sell(TFSA_2, 10, 120.0, day=4))

    assert (
        ledger.transactions,
        ledger.positions,
        ledger.cash_balances,
        ledger.book_cost(TFSA_1, "VFV.TO"),
    ) == before


def test_a_bare_registration_is_ambiguous_once_two_accounts_exist():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(TFSA_1, 1000.0),
            contrib(TFSA_2, 1000.0, day=2),
        ]
    )

    with pytest.raises(DataValidationError, match="2 accounts are registered as TFSA"):
        ledger.book_cost(TFSA, "VFV.TO")
    with pytest.raises(DataValidationError, match="pass the Account itself"):
        ledger.realized_gain_loss(TFSA)
    # The dict is keyed by (Account, currency), so each account stays addressable.
    assert set(ledger.cash_balances) == {(TFSA_1, Currency.CAD), (TFSA_2, Currency.CAD)}


def test_a_bare_registration_still_resolves_when_only_one_account_exists():
    ledger = PortfolioLedger()
    ledger.extend([contrib(TFSA_1, 1000.0), buy(TFSA_1, 10, 100.0)])

    assert ledger.book_cost(TFSA, "VFV.TO") == 1000.0
    assert ledger.book_cost_per_share(TFSA, "VFV.TO") == 100.0


# ---------------------------------------------------------------------------
# Taxable ACB is pooled across non-registered accounts
# ---------------------------------------------------------------------------


def test_two_non_registered_accounts_keep_positions_but_share_one_tax_pool():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(NR_1, 10_000.0),
            buy(NR_1, 100, 100.0),
            contrib(NR_2, 20_000.0, day=3),
            buy(NR_2, 100, 200.0, day=4),
        ]
    )

    # Holdings and book cost stay per account.
    assert ledger.positions == {(NR_1, "VFV.TO"): 100.0, (NR_2, "VFV.TO"): 100.0}
    assert ledger.book_cost(NR_1, "VFV.TO") == 10_000.0
    assert ledger.book_cost(NR_2, "VFV.TO") == 20_000.0

    # The CRA cost basis is one pool: 30,000 / 200 shares.
    assert ledger.tax_acb("VFV.TO") == 30_000.0
    assert ledger.tax_acb_per_share("VFV.TO") == 150.0


def test_taxable_sale_uses_the_pooled_acb():
    """Accounts A and B buy 100 shares at 100 and 200; A sells 100 at 150.

    CRA: proceeds 15,000 - pooled ACB of 100 shares (100 x 150) = 0.00 gain.
    A per-account calculation would wrongly report 15,000 - 10,000 = 5,000.
    """
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(NR_1, 10_000.0),
            buy(NR_1, 100, 100.0),
            contrib(NR_2, 20_000.0, day=3),
            buy(NR_2, 100, 200.0, day=4),
            sell(NR_1, 100, 150.0, day=5),
        ]
    )

    # Account A: sold out, holding the proceeds (10,000 in - 10,000 purchase
    # + 15,000 sale).
    assert ledger.positions.get((NR_1, "VFV.TO"), 0.0) == 0.0
    assert ledger.cash(NR_1) == 15_000.0
    # Account B: untouched by A's sale, book cost still its own 20,000.
    assert ledger.positions == {(NR_2, "VFV.TO"): 100.0}
    assert ledger.book_cost(NR_2, "VFV.TO") == 20_000.0
    assert ledger.cash(NR_2) == 0.0
    # Shared taxpayer-level pool: 100 shares left at the blended 150 average.
    assert ledger.tax_acb("VFV.TO") == 15_000.0
    assert ledger.tax_acb_per_share("VFV.TO") == 150.0
    assert ledger.realized_tax_gain() == 0.0
    assert ledger.realized_gain_loss(NR_1) == 0.0


def test_taxable_sale_beyond_the_pool_is_still_rejected_per_account():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(NR_1, 10_000.0),
            buy(NR_1, 100, 100.0),
            contrib(NR_2, 20_000.0, day=3),
        ]
    )

    # NonReg-2 has cash but no shares; the pooled 100 shares belong to NonReg-1.
    with pytest.raises(DataValidationError, match="only 0 held in that account"):
        ledger.record(sell(NR_2, 100, 150.0, day=4))


def test_roc_reduces_the_pooled_acb_for_taxable_accounts():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(NR_1, 10_000.0),
            buy(NR_1, 100, 100.0),
            contrib(NR_2, 20_000.0, day=3),
            buy(NR_2, 100, 200.0, day=4),
            Transaction(
                date=date(2024, 2, 1),
                account=NR_1,
                kind=TransactionKind.RETURN_OF_CAPITAL,
                ticker="VFV.TO",
                amount=1000.0,
            ),
        ]
    )

    assert ledger.tax_acb("VFV.TO") == 29_000.0
    assert ledger.tax_acb_per_share("VFV.TO") == 145.0
    # Book cost is reduced only in the account that received the distribution.
    assert ledger.book_cost(NR_1, "VFV.TO") == 9000.0
    assert ledger.book_cost(NR_2, "VFV.TO") == 20_000.0


# ---------------------------------------------------------------------------
# Registered holdings never join the taxable pool
# ---------------------------------------------------------------------------


def test_registered_holdings_are_excluded_from_the_taxable_pool():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(TFSA_1, 10_000.0),
            buy(TFSA_1, 100, 100.0),
            contrib(NR_1, 20_000.0, day=3),
            buy(NR_1, 100, 200.0, day=4),
        ]
    )

    assert ledger.tax_acb("VFV.TO") == 20_000.0
    assert ledger.tax_acb_per_share("VFV.TO") == 200.0
    assert ledger.book_cost(TFSA_1, "VFV.TO") == 10_000.0


def test_registered_sale_realizes_on_book_basis_and_is_not_a_tax_gain():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(TFSA_1, 10_000.0),
            buy(TFSA_1, 100, 100.0),
            sell(TFSA_1, 100, 150.0, day=3),
        ]
    )

    # A reporting metric only: no capital gain arises inside a TFSA.
    assert ledger.realized_gain_loss(TFSA_1) == 5000.0
    assert ledger.realized_tax_gain() == 0.0


def test_tax_gain_totals_only_non_registered_accounts():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            contrib(TFSA_1, 10_000.0),
            buy(TFSA_1, 10, 100.0),
            sell(TFSA_1, 10, 200.0, day=3),
            contrib(NR_1, 10_000.0, day=4),
            buy(NR_1, 10, 100.0, day=5),
            sell(NR_1, 10, 200.0, day=6),
        ]
    )

    assert ledger.realized_gain_loss(TFSA_1) == 1000.0
    assert ledger.realized_gain_loss(NR_1) == 1000.0
    assert ledger.realized_tax_gain() == 1000.0


# ---------------------------------------------------------------------------
# One ledger, one taxpayer
# ---------------------------------------------------------------------------


def test_accounts_of_another_owner_are_rejected_not_pooled():
    ledger = PortfolioLedger()
    spouse = Account("NonReg-spouse", NON_REG, owner="spouse")

    with pytest.raises(DataValidationError, match="belongs to owner 'spouse'"):
        ledger.record(contrib(spouse, 10_000.0))

    assert ledger.accounts == ()
    assert ledger.transactions == ()


def test_owner_is_part_of_the_ledger_scope():
    ledger = PortfolioLedger(owner="self")
    ledger.extend([contrib(NR_1, 1000.0)])

    assert ledger.accounts == (NR_1,)
    assert ledger.tax_acb("VFV.TO") == 0.0


def test_reusing_an_account_id_with_another_registration_is_rejected():
    ledger = PortfolioLedger()
    ledger.record(contrib(NR_1, 1000.0))

    with pytest.raises(DataValidationError, match="reused with different attributes"):
        ledger.record(contrib(Account("NonReg-1", TFSA), 1000.0, day=2))


# ---------------------------------------------------------------------------
# Legacy, registration-only journals are unchanged
# ---------------------------------------------------------------------------


def test_legacy_registration_only_journal_is_unchanged():
    ledger = PortfolioLedger()
    ledger.extend(
        [
            Transaction(
                date=date(2024, 1, 1),
                account=TFSA,
                kind=TransactionKind.CONTRIBUTION,
                amount=5000.0,
            ),
            Transaction(
                date=date(2024, 1, 2),
                account=TFSA,
                kind=TransactionKind.BUY,
                ticker="VFV.TO",
                quantity=10.0,
                price_per_share=100.0,
                commission=10.0,
            ),
            Transaction(
                date=date(2024, 1, 3),
                account=TFSA,
                kind=TransactionKind.SELL,
                ticker="VFV.TO",
                quantity=10.0,
                price_per_share=120.0,
            ),
        ]
    )

    default = Account.from_registration(TFSA)
    assert ledger.accounts == (default,)
    assert ledger.cash_balances == {(default, Currency.CAD): 5000.0 - 1010.0 + 1200.0}
    assert ledger.positions == {}
    assert ledger.realized_gain_loss(TFSA) == 190.0  # 1200 - 1010
    assert ledger.realized_gain_loss(default) == 190.0
    assert ledger.cost_basis_frame().iloc[0]["account"] == "TFSA"
