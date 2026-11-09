"""Immutable transaction records for multi-account portfolio bookkeeping.

This module defines the vocabulary — currency, event kind, and a single
validated :class:`Transaction` record — that every bookkeeping consumer
(journal replay, cost basis, reporting, imports) shares.

Conventions
-----------
* **CAD is the base currency.**  :data:`BASE_CURRENCY` is fixed and every
  ``*_base`` property reports Canadian dollars.
* **Native-currency fields.**  ``quantity``, ``price_per_share``, ``amount``
  and ``commission`` are expressed in the transaction's own ``currency``.
* **Explicit FX.**  ``fx_rate`` is the number of base-currency units per one
  unit of ``currency`` (for example ``1.35`` for USD→CAD).  A foreign-currency
  transaction must supply it; the rate is never fetched, inferred, or
  defaulted.  CAD transactions use the default ``1.0``.
* **One cash balance per currency.**  Cash is never netted across currencies:
  a trade debits or credits its own ``currency``, and moving money between
  currencies takes an explicit ``FX_CONVERSION``.  Each transaction's
  :attr:`Transaction.cash_balance_delta` states its native-currency cash
  effect, so the ledger applies one rule instead of reinterpreting signs.
* **Internal FX.**  An ``FX_CONVERSION`` sells ``amount`` of ``currency`` for
  ``destination_amount`` of ``destination_currency`` inside one account.  It
  is a transfer, not an external flow: it never counts as a contribution or a
  withdrawal.  Its ``commission`` is the conversion fee, charged once in the
  source currency.
* **Signs.**  All monetary fields are non-negative.  Direction is carried by
  :class:`TransactionKind`; :attr:`Transaction.cash_balance_delta` holds the
  only signed native quantities and :attr:`Transaction.cash_flow_base` the only
  signed base-currency one.  The ``*_base`` properties are reporting figures in
  CAD and never replace a native-currency balance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

from pysharpe.bookkeeping.accounts import Account, account_of
from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import AccountType


class Currency(StrEnum):
    """Currency in which a transaction's monetary fields are expressed."""

    CAD = "CAD"
    USD = "USD"


BASE_CURRENCY: Final[Currency] = Currency.CAD


class TransactionKind(StrEnum):
    """Type of portfolio event.

    ``BUY`` and ``SELL`` move shares and settle through cash.  ``DIVIDEND``,
    ``INTEREST``, ``RETURN_OF_CAPITAL``, ``CONTRIBUTION``, ``WITHDRAWAL`` and
    ``FEE`` move cash only.

    ``RETURN_OF_CAPITAL`` is reported as its own kind rather than being folded
    into ``DIVIDEND``: it is a distribution of capital, not investment income.

    ``CONTRIBUTION`` and ``WITHDRAWAL`` are the only external cash flows into
    or out of an account.  ``FX_CONVERSION`` moves cash between two currencies
    already inside one account and is therefore internal: it never changes
    contributions or withdrawals.
    """

    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    INTEREST = "INTEREST"
    RETURN_OF_CAPITAL = "RETURN_OF_CAPITAL"
    CONTRIBUTION = "CONTRIBUTION"
    WITHDRAWAL = "WITHDRAWAL"
    FEE = "FEE"
    FX_CONVERSION = "FX_CONVERSION"


# Share-moving kinds: they require a ticker, a positive quantity and a price.
_TRADED_KINDS: Final[frozenset[TransactionKind]] = frozenset(
    {TransactionKind.BUY, TransactionKind.SELL}
)

# Cash-only kinds that are identified by a security.
_TICKER_REQUIRED_KINDS: Final[frozenset[TransactionKind]] = frozenset(
    {TransactionKind.BUY, TransactionKind.SELL, TransactionKind.RETURN_OF_CAPITAL}
)

# Cash-only kinds that are account-level and cannot name a security.
_TICKER_FORBIDDEN_KINDS: Final[frozenset[TransactionKind]] = frozenset(
    {
        TransactionKind.INTEREST,
        TransactionKind.CONTRIBUTION,
        TransactionKind.WITHDRAWAL,
        TransactionKind.FEE,
        TransactionKind.FX_CONVERSION,
    }
)

# Cash-only kinds that increase the account's cash balance.
_CASH_INFLOW_KINDS: Final[frozenset[TransactionKind]] = frozenset(
    {
        TransactionKind.DIVIDEND,
        TransactionKind.INTEREST,
        TransactionKind.RETURN_OF_CAPITAL,
        TransactionKind.CONTRIBUTION,
    }
)


def _coerce_member[E: StrEnum](value: object, enum_cls: type[E], field: str) -> E:
    """Resolve *value* to an ``enum_cls`` member, ignoring case."""

    label = str(getattr(value, "value", value)).strip().upper()
    for member in enum_cls:
        if member.value.upper() == label:
            return member
    choices = ", ".join(member.value for member in enum_cls)
    raise DataValidationError(f"{field} must be one of {choices}; got {value!r}")


def _coerce_account(value: object) -> Account:
    """Resolve *value* to an :class:`~pysharpe.bookkeeping.Account`."""

    if isinstance(value, Account):
        return value
    return account_of(value)


def _non_negative(field: str, value: object) -> float:
    """Return *value* as a finite, non-negative ``float``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{field} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(f"{field} must be finite; got {value!r}")
    if number < 0:
        raise DataValidationError(f"{field} must be non-negative; got {value!r}")
    return number


def _resolve_fx_rate(currency: Currency, fx_rate: object) -> float:
    """Validate the recorded conversion rate for *currency*.

    ``fx_rate`` is base-currency units per one unit of *currency*.  There is no
    default for a foreign currency, and a CAD transaction may not carry a rate
    other than ``1.0``.
    """

    if fx_rate is None:
        if currency is not BASE_CURRENCY:
            raise DataValidationError(
                f"fx_rate is required for {currency.value} transactions: "
                "base-currency units per unit of transaction currency "
                "(for example 1.35 for USD->CAD). No rate is inferred."
            )
        return 1.0

    rate = _non_negative("fx_rate", fx_rate)
    if rate <= 0:
        raise DataValidationError(f"fx_rate must be strictly positive; got {rate!r}")
    if currency is BASE_CURRENCY and rate != 1.0:
        raise DataValidationError(
            "CAD transactions use an implicit rate of 1.0; "
            f"got fx_rate={rate!r}. Record a cross-currency purchase in the "
            "currency actually traded."
        )
    return rate


def _reject_destination(
    kind: TransactionKind, destination_currency: object, destination_amount: object
) -> None:
    """Reject conversion fields on a transaction that is not a conversion.

    A non-conversion event settles entirely in its own ``currency``; accepting
    a destination here would create a second, unattributable cash movement.
    """

    if destination_currency is not None or destination_amount != 0.0:
        raise DataValidationError(
            f"{kind.value} is not a currency conversion; destination_currency "
            "and destination_amount belong to FX_CONVERSION only."
        )


@dataclass(frozen=True)
class Transaction:
    """One validated portfolio event.

    Monetary fields are expressed in :attr:`currency`; the ``*_base``
    properties convert them to CAD at the recorded :attr:`fx_rate`.

    Attributes:
        date (date): Trade, settlement, or distribution date.
        account (Account): Account the event belongs to.  A bare
            :class:`AccountType` is accepted and resolves to that registration's
            default account, so single-account journals stay unchanged.
        kind (TransactionKind): Event type; also carries direction.
        ticker (str | None): Security symbol, when the event concerns one.
        quantity (float): Shares transacted (0 for cash-only kinds).
        price_per_share (float): Price in ``currency`` (0 for cash-only kinds).
        amount (float): Cash amount in ``currency`` (0 for traded kinds).  For
            ``FX_CONVERSION`` this is the **source** amount sold.
        currency (Currency): Currency of the native monetary fields.  For
            ``FX_CONVERSION`` this is the currency being sold.
        fx_rate (float | None): Base-currency units per unit of ``currency``.
            Required for foreign currency; ``1.0`` for CAD.
        destination_currency (Currency | None): Currency received by an
            ``FX_CONVERSION``; must be unset for every other kind.
        destination_amount (float): Amount received by an ``FX_CONVERSION``, in
            ``destination_currency``; ``0.0`` for every other kind.
        commission (float): Commission in ``currency`` (0 for cash-only kinds).
            For ``FX_CONVERSION`` it is the conversion fee, charged once in the
            source currency and debited on top of ``amount``.
        note (str): Free-form annotation; never interpreted.

    Raises:
        DataValidationError: If any field violates the conventions above.

    Example:
        >>> from datetime import date
        >>> from pysharpe.bookkeeping import (
        ...     Account,
        ...     Currency,
        ...     Transaction,
        ...     TransactionKind,
        ... )
        >>> from pysharpe.optimization.tax_location import AccountType
        >>> txn = Transaction(
        ...     date=date(2024, 1, 2),
        ...     account=AccountType.TFSA,
        ...     kind=TransactionKind.BUY,
        ...     ticker="VFV.TO",
        ...     quantity=10,
        ...     price_per_share=100.0,
        ... )
        >>> txn.account.id, txn.account.registration.value
        ('TFSA', 'TFSA')
        >>> named = Transaction(
        ...     date=date(2024, 1, 2),
        ...     account=Account("TFSA-2024", AccountType.TFSA),
        ...     kind=TransactionKind.BUY,
        ...     ticker="VFV.TO",
        ...     quantity=10,
        ...     price_per_share=100.0,
        ... )
        >>> round(named.gross_amount_base, 2)
        1000.0
        >>> conversion = Transaction(
        ...     date=date(2024, 1, 3),
        ...     account=AccountType.TFSA,
        ...     kind=TransactionKind.FX_CONVERSION,
        ...     amount=6750.0,
        ...     destination_currency=Currency.USD,
        ...     destination_amount=5000.0,
        ... )
        >>> conversion.cash_balance_delta[Currency.CAD]
        -6750.0
        >>> conversion.cash_balance_delta[Currency.USD]
        5000.0
        >>> round(conversion.effective_rate, 4)
        0.7407
    """

    date: date
    account: Account
    kind: TransactionKind
    ticker: str | None = None
    quantity: float = 0.0
    price_per_share: float = 0.0
    amount: float = 0.0
    currency: Currency = BASE_CURRENCY
    fx_rate: float | None = None
    destination_currency: Currency | None = None
    destination_amount: float = 0.0
    commission: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        kind = _coerce_member(self.kind, TransactionKind, "kind")
        account = _coerce_account(self.account)
        currency = _coerce_member(self.currency, Currency, "currency")

        ticker = self.ticker
        if ticker is not None:
            ticker = str(ticker).strip() or None

        quantity = _non_negative("quantity", self.quantity)
        price = _non_negative("price_per_share", self.price_per_share)
        amount = _non_negative("amount", self.amount)
        commission = _non_negative("commission", self.commission)
        rate = _resolve_fx_rate(currency, self.fx_rate)
        destination: Currency | None = None
        destination_amount = 0.0

        # Account-level kinds are the ones that move cash without naming a
        # security; a conversion is one of them.
        if kind in _TICKER_FORBIDDEN_KINDS and ticker is not None:
            raise DataValidationError(
                f"{kind.value} is account-level and cannot name a ticker; "
                f"got {ticker!r}"
            )

        if kind is TransactionKind.FX_CONVERSION:
            if quantity != 0 or price != 0:
                raise DataValidationError(
                    "FX_CONVERSION moves no shares; got "
                    f"quantity={quantity!r}, price_per_share={price!r}"
                )
            if amount <= 0:
                raise DataValidationError(
                    f"FX_CONVERSION requires a positive source amount; got {amount!r}"
                )
            destination = _coerce_member(
                self.destination_currency, Currency, "destination_currency"
            )
            if destination is currency:
                raise DataValidationError(
                    "FX_CONVERSION converts between two different currencies; "
                    f"both sides are {currency.value}. A same-currency "
                    "transfer is not a conversion."
                )
            destination_amount = _non_negative(
                "destination_amount", self.destination_amount
            )
            if destination_amount <= 0:
                raise DataValidationError(
                    "FX_CONVERSION requires a positive destination_amount; "
                    f"got {destination_amount!r}"
                )
        elif kind in _TRADED_KINDS:
            _reject_destination(
                kind, self.destination_currency, self.destination_amount
            )
            if ticker is None:
                raise DataValidationError(f"{kind.value} requires a ticker")
            if quantity <= 0:
                raise DataValidationError(
                    f"{kind.value} requires a positive quantity; got {quantity!r}"
                )
            if price <= 0:
                raise DataValidationError(
                    f"{kind.value} requires a positive price_per_share; got {price!r}"
                )
            if amount != 0:
                raise DataValidationError(
                    f"{kind.value} derives its cash amount from quantity and price; "
                    f"amount stayed {amount!r}"
                )
        else:
            _reject_destination(
                kind, self.destination_currency, self.destination_amount
            )
            if quantity != 0 or price != 0:
                raise DataValidationError(
                    f"{kind.value} moves no shares; got quantity={quantity!r}, "
                    f"price_per_share={price!r}"
                )
            if amount <= 0:
                raise DataValidationError(
                    f"{kind.value} requires a positive amount; got {amount!r}"
                )
            if commission != 0:
                raise DataValidationError(
                    f"{kind.value} is a cash event; record trading cost as a "
                    f"FEE instead of commission={commission!r}"
                )
            if kind in _TICKER_REQUIRED_KINDS and ticker is None:
                raise DataValidationError(f"{kind.value} requires a ticker")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "account", account)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price_per_share", price)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "destination_currency", destination)
        object.__setattr__(self, "destination_amount", destination_amount)
        object.__setattr__(self, "commission", commission)
        object.__setattr__(self, "fx_rate", rate)

    @property
    def registration(self) -> AccountType:
        """Registration type of :attr:`account`."""

        return self.account.registration

    @property
    def rate_to_base(self) -> float:
        """Base-currency units per unit of :attr:`currency`.

        ``__post_init__`` always stores a concrete rate, so the fallback only
        satisfies static analysis.
        """

        return self.fx_rate if self.fx_rate is not None else 1.0

    @property
    def price_per_share_base(self) -> float:
        """Price per share in base currency (CAD), at the recorded rate."""

        return self.price_per_share * self.rate_to_base

    @property
    def gross_amount_base(self) -> float:
        """Base-currency value of the trade or cash amount, before commission.

        For an ``FX_CONVERSION`` this is the source side translated at
        :attr:`rate_to_base` — the CAD value of the money sold, not of the money
        bought.
        """

        if self.kind in _TRADED_KINDS:
            return self.quantity * self.price_per_share_base
        return self.amount * self.rate_to_base

    @property
    def commission_base(self) -> float:
        """Commission in base currency."""

        return self.commission * self.rate_to_base

    @property
    def effective_rate(self) -> float:
        """Conversion rate actually achieved: destination units per source unit.

        Derived from the two recorded amounts, so it always agrees with them.
        ``0.0`` for kinds other than ``FX_CONVERSION``.

        Example: selling 6,750 CAD for 5,000 USD gives
        ``5_000 / 6_750 = 0.7407`` USD per CAD.
        """

        if self.kind is not TransactionKind.FX_CONVERSION:
            return 0.0
        return self.destination_amount / self.amount

    @property
    def destination_rate_to_base(self) -> float:
        """Base-currency units per unit of ``destination_currency``.

        Derived from the recorded amounts and the source's :attr:`fx_rate`, so
        a conversion records its own rate evidence instead of borrowing a later
        transaction's.  ``0.0`` for kinds other than ``FX_CONVERSION``.
        """

        if self.kind is not TransactionKind.FX_CONVERSION:
            return 0.0
        return (self.amount * self.rate_to_base) / self.destination_amount

    @property
    def position_delta(self) -> float:
        """Signed change to the share count (+buy, -sell, 0 otherwise)."""

        if self.kind is TransactionKind.BUY:
            return self.quantity
        if self.kind is TransactionKind.SELL:
            return -self.quantity
        return 0.0

    @property
    def cash_balance_delta(self) -> dict[Currency, float]:
        """Signed native-currency change to the account's cash, per currency.

        This is the module's single cash rule: the ledger applies these deltas
        directly, so no consumer reinterprets signs or guesses which balance a
        transaction touches.  A conversion debits its source currency (plus any
        fee) and credits its destination currency; the fee is charged once, on
        the source side.

        Empty currencies are omitted, so an ``FX_CONVERSION`` is the only kind
        that returns two entries.
        """

        if self.kind is TransactionKind.FX_CONVERSION:
            assert self.destination_currency is not None  # set in __post_init__
            return {
                self.currency: -(self.amount + self.commission),
                self.destination_currency: self.destination_amount,
            }

        native_gross = (
            self.quantity * self.price_per_share
            if self.kind in _TRADED_KINDS
            else self.amount
        )
        if self.kind is TransactionKind.BUY:
            return {self.currency: -(native_gross + self.commission)}
        if self.kind is TransactionKind.SELL:
            return {self.currency: native_gross - self.commission}
        if self.kind in _CASH_INFLOW_KINDS:
            return {self.currency: native_gross}
        return {self.currency: -native_gross}

    @property
    def cash_flow_base(self) -> float:
        """Signed base-currency value of this transaction's cash movement.

        A reporting figure in CAD; the balances it changes are per currency (see
        :attr:`cash_balance_delta`).  An internal conversion whose recorded
        rates agree nets to zero here, so only its fee shows up as a cost.
        """

        if self.kind is TransactionKind.FX_CONVERSION:
            assert self.destination_currency is not None  # set in __post_init__
            converted = self.destination_amount * self.destination_rate_to_base
            paid = self.amount * self.rate_to_base + self.commission_base
            return converted - paid

        gross = self.gross_amount_base
        if self.kind is TransactionKind.BUY:
            return -(gross + self.commission_base)
        if self.kind is TransactionKind.SELL:
            return gross - self.commission_base
        if self.kind in _CASH_INFLOW_KINDS:
            return gross
        return -gross
