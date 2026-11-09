"""Replay a transaction journal into per-account holdings, cash and cost basis.

:class:`PortfolioLedger` is the bookkeeping substrate every downstream consumer
(rebalancing, cash-flow routing, reporting, imports) reads from: it answers
"what do I hold, how much cash is in each account, and what has it cost me"
from the recorded events alone, without touching market data or the network.

Scope of this module
--------------------
It reconstructs **quantities, cash, book cost, and realized gains** from the
journal.  Cost basis arithmetic — weighted-average cost (ITA s. 47(1)),
commission loading, and return-of-capital reduction — is not reimplemented
here: every account owns one
:class:`pysharpe.guardrails.tax_compliance.ACBTracker` pool and the ledger
feeds it converted amounts.  That tracker is the commission-aware compliance
ledger, and it shares the return-of-capital rule with
:mod:`pysharpe.execution.tax_tracker` through
:func:`~pysharpe.execution.tax_tracker.apply_return_of_capital`.

Two layers of cost are tracked, because they answer different questions:

* **book cost**, per :class:`~pysharpe.bookkeeping.Account` — what the statement
  says the account paid.  Registered accounts have nothing else, because no
  capital gain arises inside them.
* **taxable ACB**, pooled across *all* non-registered accounts of the taxpayer
  per ticker.  ITA s. 47(1) averages identical properties owned by the
  *taxpayer*, and a brokerage account is not a taxpayer, so a sale in one
  non-registered account must consume the blended cost of every taxable account
  holding that security.

The ledger is scoped to a single taxpayer.  Accounts belonging to another owner
are rejected rather than pooled.

Everything the ledger reports is in base currency (CAD), converted at each
transaction's own recorded rate.  That matches CRA's requirement to translate
an acquisition at its acquisition-date rate and a disposition at its
disposition-date rate, so the currency effect lands in the realized gain rather
than in the book cost.

The one exception is cash, which is a quantity of a *currency*: it is held per
``(account, currency)`` in native units, because "9,500 CAD" and "9,500 USD"
are different amounts of money and netting them would invent a rate.  Foreign
securities settle from the cash of the currency they trade in, and moving money
between currencies is an explicit ``FX_CONVERSION`` event carrying both amounts
and the effective rate it achieved.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from pysharpe.bookkeeping.accounts import (
    SELF_OWNER,
    Account,
    account_of,
    coerce_registration,
)
from pysharpe.bookkeeping.transactions import (
    BASE_CURRENCY,
    Currency,
    Transaction,
    TransactionKind,
)
from pysharpe.bookkeeping.valuation import Valuation, value_portfolio
from pysharpe.exceptions import DataValidationError
from pysharpe.guardrails.tax_compliance import ACBPosition, ACBTracker
from pysharpe.optimization.tax_location import AccountType

# Absolute tolerance for floating-point drift in share counts and base-currency
# balances.  It absorbs summation noise only; it is not a rounding rule and does
# not make a genuinely unbalanced journal pass.
_TOLERANCE = 1e-9


def _describe(transaction: Transaction) -> str:
    """Return a compact, log-friendly description of *transaction*."""

    subject = transaction.ticker or "cash"
    return (
        f"{transaction.kind.value} {subject} in {transaction.account.id} "
        f"({transaction.registration.value}) on {transaction.date.isoformat()}"
    )


@dataclass(frozen=True)
class Position:
    """Quantity of one ticker held in one account.

    Attributes:
        account (Account): Individual account holding the shares.
        ticker (str): Security symbol.
        quantity (float): Shares held; always strictly positive.
    """

    account: Account
    ticker: str
    quantity: float


class PortfolioLedger:
    """Replay a journal of :class:`Transaction` records into current state.

    Transactions may be appended in any order.  Replay sorts them by
    ``(date, insertion order)``, so same-day events settle in the order they
    were recorded, and a late-arriving statement entry dated earlier changes the
    resulting balances.  Replay is deterministic: the same journal always yields
    the same holdings and cash.

    Every mutation replays and validates immediately, so invariant violations
    raise at the call that introduced them and a rejected batch leaves the
    ledger in its previous, valid state.

    Invariants:
        * A ``SELL`` may not exceed the shares held **in that account**.
        * A cash balance may not go negative unless ``allow_negative_cash`` is
          set.  The rule applies per **(account, currency)**: a positive CAD
          balance never covers a USD deficit, because nothing converts
          automatically.
        * Every account belongs to one owner; accounts of another owner are
          rejected instead of being pooled with this ledger's accounts.
        * An account id always carries the same registration and owner.

    Cash is tracked per account **and per currency**, in that currency's own
    units: a CAD purchase debits CAD cash, a USD purchase debits USD cash, and a
    USD dividend credits USD cash.  Moving money between currencies requires an
    explicit ``FX_CONVERSION``, so a ledger never invents a rate.  Legacy
    CAD-only journals behave exactly as before; their only balance is
    ``(account, Currency.CAD)``.

    Contributions and withdrawals are the only external flows: they are the only
    kinds that move money into or out of an account.  An ``FX_CONVERSION`` is a
    transfer between two balances of the same account and therefore never
    changes :meth:`contributed` or :meth:`withdrawn`.

    Cost basis is tracked in two layers.  Each account keeps its own
    weighted-average **book cost** per ticker, fed with CAD amounts derived from
    each transaction's own rate; commission is loaded into the purchase cost and
    deducted from sale proceeds, and a ``RETURN_OF_CAPITAL`` reduces the pool.
    Non-registered accounts additionally share one **taxable ACB** pool per
    ticker, because ITA s. 47(1) averages identical properties per taxpayer, not
    per account.  Registered accounts (TFSA, FHSA, RRSP, …) never join that
    pool: their cost base is book value for reporting only.

    Args:
        allow_negative_cash (bool): Permit accounts to carry a negative cash
            balance (for example margin accounts, or journals recorded before
            their contributions).  Defaults to ``False``.
        owner (str): Taxpayer this ledger describes.  Defaults to
            :data:`~pysharpe.bookkeeping.accounts.SELF_OWNER`.

    Example:
        >>> from datetime import date
        >>> from pysharpe.bookkeeping import (
        ...     Account,
        ...     PortfolioLedger,
        ...     Transaction,
        ...     TransactionKind,
        ... )
        >>> from pysharpe.optimization.tax_location import AccountType
        >>> ledger = PortfolioLedger()
        >>> tfsa = Account("TFSA-2024", AccountType.TFSA)
        >>> _ = ledger.record(Transaction(
        ...     date=date(2024, 1, 2),
        ...     account=tfsa,
        ...     kind=TransactionKind.CONTRIBUTION,
        ...     amount=5000.0,
        ... ))
        >>> _ = ledger.record(Transaction(
        ...     date=date(2024, 1, 3),
        ...     account=tfsa,
        ...     kind=TransactionKind.BUY,
        ...     ticker="VFV.TO",
        ...     quantity=10,
        ...     price_per_share=100.0,
        ... ))
        >>> ledger.positions[(tfsa, "VFV.TO")]
        10.0
        >>> round(ledger.cash(tfsa), 2)
        4000.0
        >>> ledger.book_cost_per_share(tfsa, "VFV.TO")
        100.0
        >>> round(ledger.valuation({"VFV.TO": 110.0}).total_unrealized_gain_loss, 2)
        100.0
        >>> ledger.contributed(tfsa)
        5000.0

    A foreign-currency purchase settles from that currency's own balance, which
    an ``FX_CONVERSION`` must fund first:

        >>> from pysharpe.bookkeeping import Currency
        >>> _ = ledger.record(Transaction(
        ...     date=date(2024, 1, 4),
        ...     account=tfsa,
        ...     kind=TransactionKind.FX_CONVERSION,
        ...     amount=1350.0,
        ...     destination_currency=Currency.USD,
        ...     destination_amount=1000.0,
        ... ))
        >>> _ = ledger.record(Transaction(
        ...     date=date(2024, 1, 5),
        ...     account=tfsa,
        ...     kind=TransactionKind.BUY,
        ...     ticker="VOO",
        ...     quantity=2,
        ...     price_per_share=200.0,
        ...     currency=Currency.USD,
        ...     fx_rate=1.35,
        ... ))
        >>> ledger.cash(tfsa, Currency.CAD)
        2650.0
        >>> ledger.cash(tfsa, Currency.USD)
        600.0
        >>> ledger.contributed(tfsa)
        5000.0
    """

    def __init__(
        self, *, allow_negative_cash: bool = False, owner: str = SELF_OWNER
    ) -> None:
        self._allow_negative_cash = bool(allow_negative_cash)
        self._owner = owner
        self._transactions: list[Transaction] = []
        self._accounts: dict[str, Account] = {}
        self._book_pools: dict[str, ACBTracker] = {}
        self._tax_pool = ACBTracker()
        self._positions: dict[tuple[Account, str], float] = {}
        self._cash: dict[tuple[Account, Currency], float] = {}
        self._contributed: dict[str, float] = {}
        self._withdrawn: dict[str, float] = {}
        self._realized: dict[tuple[str, str], float] = {}
        self._totals: dict[TransactionKind, float] = {}
        self._total_commissions: float = 0.0
        self._replay()

    # -- Mutation ------------------------------------------------------------

    def record(self, transaction: Transaction) -> None:
        """Append one transaction, then replay and validate the journal.

        Args:
            transaction (Transaction): Validated event to append.

        Raises:
            DataValidationError: If appending it would break a ledger invariant;
                the ledger keeps its previous state.
        """

        self.extend([transaction])

    def extend(self, transactions: Iterable[Transaction]) -> None:
        """Append a batch of transactions with a single replay.

        Prefer this over repeated :meth:`record` calls when importing a
        statement.

        Args:
            transactions (Iterable[Transaction]): Events to append.

        Raises:
            DataValidationError: If the combined journal breaks an invariant;
                the batch is discarded and the ledger keeps its previous state.
        """

        batch = list(transactions)
        for transaction in batch:
            if not isinstance(transaction, Transaction):
                raise DataValidationError(
                    "PortfolioLedger accepts Transaction records; "
                    f"got {type(transaction).__name__}"
                )
        if not batch:
            return

        start = len(self._transactions)
        self._transactions.extend(batch)
        try:
            self._replay()
        except DataValidationError:
            del self._transactions[start:]
            raise

    # -- Read-only accessors -------------------------------------------------

    @property
    def transactions(self) -> tuple[Transaction, ...]:
        """Journal in insertion order."""

        return tuple(self._transactions)

    @property
    def ordered_transactions(self) -> tuple[Transaction, ...]:
        """Journal in replay order: ``(date, insertion order)``."""

        return tuple(self._ordered())

    @property
    def accounts(self) -> tuple[Account, ...]:
        """Accounts appearing in the journal, sorted by registration then id."""

        return tuple(
            sorted(
                self._accounts.values(),
                key=lambda account: (account.registration.value, account.id),
            )
        )

    @property
    def positions(self) -> dict[tuple[Account, str], float]:
        """Current share counts keyed by ``(account, ticker)``, zero holdings excluded."""

        return dict(self._positions)

    @property
    def cash_balances(self) -> dict[tuple[Account, Currency], float]:
        """Current cash keyed by ``(account, currency)``, in native units.

        Balances are never netted or converted across currencies.  Every
        ``(account, currency)`` pair the journal touched is present, including a
        zero balance, so a fully invested account still reports its currencies.
        For a CAD-only journal this is exactly the old per-account map, keyed
        with :data:`BASE_CURRENCY`.
        """

        return dict(self._cash)

    def cash(
        self,
        account: Account | AccountType | str,
        currency: Currency = BASE_CURRENCY,
    ) -> float:
        """Return one account's cash in one currency, in that currency's units.

        Args:
            account (Account | AccountType | str): Account to report on.  A bare
                registration is accepted when exactly one account of that
                registration exists.
            currency (Currency): Currency of the balance.  Defaults to CAD.

        Returns:
            float: Balance in *currency*, or ``0.0`` when the account has never
                held that currency.
        """

        return self._cash.get((self._resolve(account), currency), 0.0)

    @property
    def total_contributions(self) -> float:
        """External contributions across the ledger, in base currency (CAD).

        Only ``CONTRIBUTION`` events count.  Internal transfers — including an
        ``FX_CONVERSION`` — never add to this figure, so it stays a record of
        money that actually entered the accounts.
        """

        return self._totals[TransactionKind.CONTRIBUTION]

    @property
    def total_withdrawals(self) -> float:
        """External withdrawals across the ledger, in base currency (CAD)."""

        return self._totals[TransactionKind.WITHDRAWAL]

    def contributed(self, account: Account | AccountType | str) -> float:
        """Return an account's external contributions, in base currency (CAD).

        A contribution is money entering from outside the account.  Moving money
        between currencies inside the same account is not a contribution, so an
        ``FX_CONVERSION`` leaves this figure untouched.

        Args:
            account (Account | AccountType | str): Account to report on.

        Returns:
            float: Contributions in CAD at each contribution's own recorded
                rate; ``0.0`` when the account has never been funded.
        """

        return self._contributed.get(self._resolve(account).id, 0.0)

    def withdrawn(self, account: Account | AccountType | str) -> float:
        """Return an account's external withdrawals, in base currency (CAD).

        Args:
            account (Account | AccountType | str): Account to report on.

        Returns:
            float: Withdrawals in CAD at each withdrawal's own recorded rate;
                ``0.0`` when nothing has been withdrawn.
        """

        return self._withdrawn.get(self._resolve(account).id, 0.0)

    @property
    def totals_by_kind(self) -> dict[TransactionKind, float]:
        """Gross base-currency total recorded for each kind, before commissions."""

        return dict(self._totals)

    @property
    def total_commissions(self) -> float:
        """Total transaction costs paid across the journal, in base currency.

        Includes trading commissions and the fees charged on explicit currency
        conversions.  Each cost is recorded once, on the transaction that
        incurred it.
        """

        return self._total_commissions

    def book_cost(self, account: Account | AccountType | str, ticker: str) -> float:
        """Return an account's **book cost** for a holding, in CAD.

        Book cost is what the account paid: acquisition cost plus commissions,
        reduced by return of capital.  It is a statement figure.  For a
        non-registered account it is *not* the CRA cost basis once the same
        security is also held in another taxable account — use
        :meth:`tax_acb` for that.

        Args:
            account (Account | AccountType | str): Account to report on.  A bare
                registration is accepted when exactly one account of that
                registration exists.
            ticker (str): Security symbol.

        Returns:
            float: Book cost in CAD, or ``0.0`` when the account holds no shares
                of *ticker*.
        """

        position = self._book_position(account, ticker)
        return 0.0 if position is None else position.total_cost

    def book_cost_per_share(
        self, account: Account | AccountType | str, ticker: str
    ) -> float:
        """Return an account's book cost per share, in CAD.

        Returns:
            float: ``book_cost / shares`` in CAD, or ``0.0`` when no shares are
                held.
        """

        position = self._book_position(account, ticker)
        return 0.0 if position is None else position.acb_per_share

    def tax_acb(self, ticker: str) -> float:
        """Return the pooled taxable ACB for *ticker*, in CAD.

        The pool spans every non-registered account of this ledger's taxpayer,
        because ITA s. 47(1) averages identical properties owned by the taxpayer
        rather than by account.  Registered holdings never enter it.

        Returns:
            float: Pooled taxable ACB in CAD, or ``0.0`` when no non-registered
                account holds *ticker*.
        """

        position = self._tax_pool.positions.get(ticker)
        return 0.0 if position is None else position.total_cost

    def tax_acb_per_share(self, ticker: str) -> float:
        """Return the pooled taxable ACB per share for *ticker*, in CAD."""

        position = self._tax_pool.positions.get(ticker)
        return 0.0 if position is None else position.acb_per_share

    def _book_position(
        self, account: Account | AccountType | str, ticker: str
    ) -> ACBPosition | None:
        """Return the book-cost position for *account*/*ticker*, if any.

        Read-only: an unknown security is not added to the pool as a zero
        position, so queries cannot create phantom report rows.
        """

        pool = self._book_pools.get(self._resolve(account).id)
        if pool is None:
            return None
        return pool.positions.get(ticker)

    def _resolve(self, account: Account | AccountType | str) -> Account:
        """Resolve an account reference without silently merging accounts.

        An explicit :class:`Account` is used as given.  A bare registration is
        accepted only when exactly one account of that registration exists, so
        legacy single-account journals keep working while a two-TFSA journal
        fails loudly instead of pooling the wrong account.
        """

        if isinstance(account, Account):
            return account
        registration = coerce_registration(account)
        matches = [
            known
            for known in self._accounts.values()
            if known.registration is registration
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return account_of(registration)
        detail = ", ".join(sorted(known.id for known in matches))
        raise DataValidationError(
            f"{len(matches)} accounts are registered as {registration.value} "
            f"({detail}); pass the Account itself instead of the registration"
        )

    def realized_gain_loss(
        self, account: Account | AccountType | str, ticker: str | None = None
    ) -> float:
        """Return realized capital gains attributed to an account, in CAD.

        The figure is computed on the basis the account's registration calls
        for, so it is never a misleading tax number:

        * non-registered accounts realize against the **pooled taxable ACB**;
        * registered accounts realize against their **own book average**, a
          reporting metric only (no capital gain arises inside a TFSA/RRSP).

        Includes ``SELL`` dispositions and ``RETURN_OF_CAPITAL`` amounts that
        exceeded the remaining cost base.

        Args:
            account (Account | AccountType | str): Account to report on.  A bare
                registration is accepted when exactly one account of that
                registration exists.
            ticker (str | None): Restrict the total to one security.  Defaults
                to the whole account.

        Returns:
            float: Realized gain (positive) or loss (negative) in CAD; ``0.0``
                when nothing has been realized.
        """

        account_id = self._resolve(account).id
        return sum(
            gain
            for (holder, symbol), gain in self._realized.items()
            if holder == account_id and (ticker is None or symbol == ticker)
        )

    def realized_tax_gain(self, ticker: str | None = None) -> float:
        """Return realized capital gains across all taxable accounts, in CAD.

        This is the reporting-level total for a Canadian capital gains return:
        every non-registered disposition of this ledger's taxpayer, measured
        against the pooled ACB.  Registered accounts are excluded because they
        generate no reportable gain.

        Args:
            ticker (str | None): Restrict the total to one security.
        """

        taxable = {
            account.id for account in self._accounts.values() if account.is_taxable
        }
        return sum(
            gain
            for (holder, symbol), gain in self._realized.items()
            if holder in taxable and (ticker is None or symbol == ticker)
        )

    def cost_basis_frame(self) -> pd.DataFrame:
        """Return book cost and realized results per ``(account, ticker)``.

        Columns are ``account`` (the account id), ``registration``, ``ticker``,
        ``quantity``, ``book_cost``, ``book_cost_per_share`` and
        ``realized_gain_loss``.  Every security an account has touched appears
        once, so fully disposed holdings are included with a zero quantity and
        their realized result.  The pooled taxable ACB is not a column here — it
        spans accounts — so use :meth:`tax_acb` for it.
        """

        columns = [
            "account",
            "registration",
            "ticker",
            "quantity",
            "book_cost",
            "book_cost_per_share",
            "realized_gain_loss",
        ]
        rows = []
        for account in self.accounts:
            # Every registered account received a pool during replay.
            pool = self._book_pools[account.id]
            for ticker, position in sorted(pool.positions.items()):
                rows.append(
                    {
                        "account": account.id,
                        "registration": account.registration,
                        "ticker": ticker,
                        "quantity": position.total_shares,
                        "book_cost": position.total_cost,
                        "book_cost_per_share": position.acb_per_share,
                        "realized_gain_loss": self.realized_gain_loss(account, ticker),
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def holdings(self) -> list[Position]:
        """Return current positions sorted by registration, account id, then ticker."""

        return [
            Position(account=account, ticker=ticker, quantity=quantity)
            for (account, ticker), quantity in sorted(
                self._positions.items(),
                key=lambda item: (
                    item[0][0].registration.value,
                    item[0][0].id,
                    item[0][1],
                ),
            )
        ]

    def valuation(
        self,
        prices: Mapping[str, float],
        fx_rates: Mapping[Currency, float] | None = None,
    ) -> Valuation:
        """Mark the current holdings and cash to market, in base currency (CAD).

        Prices are per share in CAD and are never derived from the ledger's own
        historical ``fx_rate``s, which describe book cost rather than today's
        market.  Translate prices upstream (the data pipeline already produces
        CAD series) before valuing foreign securities.

        Foreign-currency cash is held in its own units, so marking it to CAD
        needs a **current** rate that the ledger does not know and must not
        invent.  Supply one per non-base currency held; a non-zero balance in a
        currency with no rate raises instead of being dropped or counted at
        par.

        Args:
            prices (Mapping[str, float]): Price per share in base currency
                (CAD).  A ``NaN`` entry means the price is unknown.
            fx_rates (Mapping[Currency, float] | None): Current CAD units per
                one unit of each non-base currency (``{Currency.USD: 1.36}``).
                Required only for currencies with a non-zero balance.

        Returns:
            Valuation: Priced positions, cash, and any held security left
                unpriced.  A held security without a price is excluded from the
                totals and flags the result as incomplete — it is never valued
                at zero.

        Raises:
            DataValidationError: If a non-zero foreign balance has no supplied
                rate, or a supplied rate is not a finite, positive number.
        """

        return value_portfolio(
            self._positions,
            self._cash_in_base(fx_rates),
            prices,
            self._cost_basis_map(),
        )

    def _checked_fx_rates(
        self, fx_rates: Mapping[Currency, float] | None
    ) -> dict[Currency, float]:
        """Validate caller-supplied current rates for valuation.

        A rate is CAD units per one unit of the quoted currency.  CAD is the base
        currency, so it needs no rate and cannot be re-rated.
        """

        rates: dict[Currency, float] = {}
        for currency, raw in (fx_rates or {}).items():
            if not isinstance(currency, Currency):
                raise DataValidationError(
                    "fx_rates keys must be Currency members; "
                    f"got {currency!r}. Pass {{{Currency.USD!r}: 1.36}} "
                    "rather than string keys."
                )
            if currency is BASE_CURRENCY:
                raise DataValidationError(
                    "fx_rates must not re-rate CAD: it is the base currency, "
                    "so its rate is 1.0"
                )
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise DataValidationError(
                    f"fx_rates[{currency.value}] must be a number; got {raw!r}"
                )
            rate = float(raw)
            if not math.isfinite(rate) or rate <= 0:
                raise DataValidationError(
                    f"fx_rates[{currency.value}] must be finite and positive; "
                    f"got {raw!r}"
                )
            rates[currency] = rate
        return rates

    def _cash_in_base(
        self, fx_rates: Mapping[Currency, float] | None
    ) -> dict[Account, float]:
        """Translate per-currency cash into one CAD figure per account.

        Nothing is netted across currencies: each balance is multiplied by the
        caller's current rate.  A zero balance in an unrated currency contributes
        nothing, so a CAD-only journal never needs a rate.
        """

        rates = self._checked_fx_rates(fx_rates)
        totals: dict[Account, float] = {}
        for (account, currency), amount in self._cash.items():
            if amount == 0.0:
                continue
            if currency is BASE_CURRENCY:
                rate = 1.0
            elif currency in rates:
                rate = rates[currency]
            else:
                raise DataValidationError(
                    f"{account.id} holds {amount:.2f} {currency.value} but no "
                    f"current {currency.value}->CAD rate was supplied; pass "
                    f"fx_rates={{{currency!r}: <rate>}} to value the portfolio"
                )
            totals[account] = totals.get(account, 0.0) + amount * rate
        return totals

    def _cost_basis_map(self) -> dict[tuple[Account, str], float]:
        """Return book costs for the open holdings, keyed like :attr:`positions`."""

        return {
            (self._accounts[account_id], ticker): position.total_cost
            for account_id, pool in self._book_pools.items()
            for ticker, position in pool.positions.items()
            if position.total_shares > _TOLERANCE
        }

    def transactions_frame(self) -> pd.DataFrame:
        """Return the journal as a DataFrame with base-currency columns.

        Columns are ``date``, ``account`` (the id), ``registration``, ``kind``,
        ``ticker``, ``quantity``, ``price_per_share``, ``currency``,
        ``fx_rate``, ``amount``, ``destination_currency``,
        ``destination_amount``, ``effective_rate``, ``commission``,
        ``gross_amount_base`` and ``cash_flow_base``, in replay order.

        The destination columns and ``effective_rate`` are evidence for explicit
        conversions: they are empty/``0.0`` on every other kind.  ``fx_rate``
        stays each transaction's own recorded rate, so acquisition evidence is
        never overwritten by a later conversion.
        """

        columns = [
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
        rows = [
            {
                "date": txn.date,
                "account": txn.account.id,
                "registration": txn.registration,
                "kind": txn.kind,
                "ticker": txn.ticker,
                "quantity": txn.quantity,
                "price_per_share": txn.price_per_share,
                "currency": txn.currency,
                "fx_rate": txn.rate_to_base,
                "amount": txn.amount,
                "destination_currency": txn.destination_currency,
                "destination_amount": txn.destination_amount,
                "effective_rate": txn.effective_rate,
                "commission": txn.commission,
                "gross_amount_base": txn.gross_amount_base,
                "cash_flow_base": txn.cash_flow_base,
            }
            for txn in self._ordered()
        ]
        return pd.DataFrame(rows, columns=columns)

    # -- Replay --------------------------------------------------------------

    def _ordered(self) -> list[Transaction]:
        """Return the journal sorted by ``(date, insertion order)``."""

        indexed = enumerate(self._transactions)
        return [
            transaction
            for _, transaction in sorted(
                indexed, key=lambda item: (item[1].date, item[0])
            )
        ]

    def _replay(self) -> None:
        """Rebuild accounts, positions, cash and cost basis from the journal.

        State is built into locals and assigned only on success, so a failed
        replay leaves the previous state untouched.  Holdings are derived from
        the per-account book-cost pools rather than tracked in parallel, so share
        counts and book cost can never disagree.
        """

        accounts: dict[str, Account] = {}
        book_pools: dict[str, ACBTracker] = defaultdict(ACBTracker)
        tax_pool = ACBTracker()
        cash: dict[tuple[str, Currency], float] = defaultdict(float)
        contributed: dict[str, float] = defaultdict(float)
        withdrawn: dict[str, float] = defaultdict(float)
        realized: dict[tuple[str, str], float] = {}
        totals: dict[TransactionKind, float] = dict.fromkeys(TransactionKind, 0.0)
        commissions = 0.0

        for transaction in self._ordered():
            account = transaction.account
            ticker = transaction.ticker
            self._check_account(account, accounts)
            pool = book_pools[account.id]

            if transaction.kind is TransactionKind.SELL and ticker:
                held = pool.get_shares(ticker)
                if transaction.quantity > held + _TOLERANCE:
                    raise DataValidationError(
                        f"Cannot sell {transaction.quantity:g} {ticker} on "
                        f"{transaction.date.isoformat()} in {account.id} "
                        f"({account.registration.value}): only {held:g} held "
                        "in that account."
                    )

            # Cash is applied per currency, from the transaction's own deltas, so
            # a foreign trade can only draw on that currency's balance.
            for currency, delta in transaction.cash_balance_delta.items():
                balance = cash[(account.id, currency)] + delta
                cash[(account.id, currency)] = balance
                if not self._allow_negative_cash and balance < -_TOLERANCE:
                    raise DataValidationError(
                        f"{account.id} {currency.value} cash would fall to "
                        f"{balance:.2f} after {_describe(transaction)}. Record "
                        f"the missing contribution or {currency.value} "
                        "conversion first, or construct the ledger with "
                        "allow_negative_cash=True."
                    )

            # Only these two kinds are money entering or leaving the account.
            # An FX_CONVERSION is internal, so it never lands in these totals.
            if transaction.kind is TransactionKind.CONTRIBUTION:
                contributed[account.id] += transaction.gross_amount_base
            elif transaction.kind is TransactionKind.WITHDRAWAL:
                withdrawn[account.id] += transaction.gross_amount_base

            gain = self._apply_to_pools(pool, tax_pool, transaction)
            if gain and ticker is not None:
                key = (account.id, ticker)
                realized[key] = realized.get(key, 0.0) + gain

            totals[transaction.kind] += transaction.gross_amount_base
            commissions += transaction.commission_base

        positions = {
            (accounts[account_id], ticker): position.total_shares
            for account_id, pool in book_pools.items()
            for ticker, position in pool.positions.items()
            if position.total_shares > _TOLERANCE
        }

        self._accounts = accounts
        self._book_pools = dict(book_pools)
        self._tax_pool = tax_pool
        self._positions = positions
        self._cash = {
            (accounts[account_id], currency): balance
            for (account_id, currency), balance in cash.items()
        }
        self._contributed = dict(contributed)
        self._withdrawn = dict(withdrawn)
        self._realized = realized
        self._totals = totals
        self._total_commissions = commissions

    def _check_account(self, account: Account, seen: dict[str, Account]) -> None:
        """Reject foreign owners and reuse of an account id, then register it."""

        if account.owner != self._owner:
            raise DataValidationError(
                f"account {account.id!r} belongs to owner {account.owner!r}, but "
                f"this ledger is scoped to {self._owner!r}. One ledger holds one "
                "taxpayer's accounts; different owners are never pooled."
            )
        known = seen.get(account.id)
        if known is None:
            seen[account.id] = account
        elif known != account:
            raise DataValidationError(
                f"account id {account.id!r} is reused with different attributes: "
                f"{known.registration.value}/{known.owner} then "
                f"{account.registration.value}/{account.owner}"
            )

    def _apply_to_pools(
        self, pool: ACBTracker, tax_pool: ACBTracker, transaction: Transaction
    ) -> float:
        """Book a transaction into the account pool and, if taxable, the tax pool.

        Prices and commissions are converted at the transaction's own rate, so
        both pools are denominated in base currency.  Returns the realized gain
        to attribute to the account: the pooled taxable figure for non-registered
        accounts, the account's own book figure otherwise.
        """

        ticker = transaction.ticker
        if ticker is None:
            return 0.0

        taxable = transaction.account.is_taxable
        if transaction.kind is TransactionKind.BUY:
            arguments = (
                ticker,
                transaction.quantity,
                transaction.price_per_share_base,
                transaction.date,
            )
            keyword = {
                "commission": transaction.commission_base,
                "account_type": transaction.registration.value,
            }
            pool.record_buy(*arguments, **keyword)
            if taxable:
                tax_pool.record_buy(*arguments, **keyword)
            return 0.0

        if transaction.kind is TransactionKind.SELL:
            arguments = (
                ticker,
                transaction.quantity,
                transaction.price_per_share_base,
                transaction.date,
            )
            keyword = {
                "commission": transaction.commission_base,
                "account_type": transaction.registration.value,
            }
            book_gain = pool.record_sell(*arguments, **keyword)
            if taxable:
                return tax_pool.record_sell(*arguments, **keyword)
            return book_gain

        if transaction.kind is TransactionKind.RETURN_OF_CAPITAL:
            _, book_gain = pool.record_roc(
                ticker,
                transaction.gross_amount_base,
                transaction.date,
                account_type=transaction.registration.value,
            )
            if taxable:
                _, tax_gain = tax_pool.record_roc(
                    ticker,
                    transaction.gross_amount_base,
                    transaction.date,
                    account_type=transaction.registration.value,
                )
                return tax_gain
            return book_gain

        return 0.0
