"""Multi-currency Canadian portfolio ledger and accounting reconciliation.

Provides a deterministic, append-and-replay transaction ledger for Canadian
multi-account portfolios holding CAD and USD cash balances alongside CAD-listed
and USD-listed securities.

Key design guarantees
---------------------
- **Exact decimal cash & position bookkeeping:** Monetary amounts and FX
  conversions use :class:`decimal.Decimal` with an explicit configurable
  rounding mode (default ``ROUND_HALF_EVEN``) rather than silent binary float
  rounding.
- **Chronological deterministic replay:** Transactions may be added out of order
  and are replayed in ``(timestamp, insertion_order)`` sequence. Any
  invariant violation (insufficient cash when ``allow_negative_cash=False``,
  overselling shares, conflicting account metadata, or mixed naive/aware
  timestamps) rolls back atomically and raises :class:`DataValidationError`.
- **Account & currency isolation:** CAD and USD cash balances, positions, and
  book costs are isolated per ``(account_id, currency)`` and
  ``(account_id, ticker)``. No implicit FX sweeping occurs.
- **CRA Non-Reg ACB separation:** Only ``AccountType.NON_REG`` buys and sells
  feed the commission-aware :class:`pysharpe.guardrails.tax_compliance.ACBTracker`
  using explicit CAD-translated prices and fees. Registered wrappers (TFSA,
  RRSP, FHSA, LIRA, RRIF) never pollute Non-Registered tax ACB.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
)
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import pandas as pd

from pysharpe.exceptions import DataValidationError
from pysharpe.execution.tax_tracker import ACBPosition
from pysharpe.optimization.tax_location import AccountType

if TYPE_CHECKING:
    from pysharpe.guardrails.tax_compliance import ACBTracker as NonRegACBTracker


def _new_non_reg_acb_tracker() -> NonRegACBTracker:
    # Deferred import avoids a package-init cycle when guardrails.tax_compliance
    # (which imports pysharpe.execution.tax_tracker) is imported before execution.
    from pysharpe.guardrails.tax_compliance import ACBTracker

    return ACBTracker()


_CENT = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_PENNY_TOLERANCE = Decimal("0.01")

_SUPPORTED_ROUNDING_MODES: frozenset[str] = frozenset(
    {
        ROUND_HALF_EVEN,
        ROUND_HALF_UP,
        ROUND_HALF_DOWN,
        ROUND_DOWN,
        ROUND_CEILING,
        ROUND_FLOOR,
    }
)


class Currency(StrEnum):
    """Supported settlement currencies in the portfolio ledger."""

    CAD = "CAD"
    USD = "USD"


class TransactionKind(StrEnum):
    """Supported account transaction types."""

    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    BUY = "BUY"
    SELL = "SELL"
    FEE = "FEE"
    FX_CONVERSION = "FX_CONVERSION"


NumericInput = Decimal | int | float | str


def _to_decimal(value: NumericInput, *, field_name: str) -> Decimal:
    """Convert numeric input to a finite :class:`Decimal` without float drift."""
    if isinstance(value, bool):
        raise DataValidationError(f"{field_name} cannot be a boolean.")
    try:
        dec = (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value) if isinstance(value, (int, float)) else value)
        )
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise DataValidationError(
            f"{field_name} must be a valid finite number; got {value!r}."
        ) from exc
    if not dec.is_finite():
        raise DataValidationError(f"{field_name} must be finite; got {value!r}.")
    return dec


def _quantize_money(value: Decimal, rounding: str) -> Decimal:
    """Round a monetary Decimal value to cents using *rounding*."""
    return value.quantize(_CENT, rounding=rounding)


def _require_positive_decimal(value: Decimal, *, field_name: str) -> Decimal:
    if value <= _ZERO:
        raise DataValidationError(
            f"{field_name} must be strictly positive; got {value}."
        )
    return value


def _require_non_negative_decimal(value: Decimal, *, field_name: str) -> Decimal:
    if value < _ZERO:
        raise DataValidationError(f"{field_name} must be non-negative; got {value}.")
    return value


def _validate_rounding_mode(rounding: str) -> None:
    if rounding not in _SUPPORTED_ROUNDING_MODES:
        raise DataValidationError(
            f"Unsupported Decimal rounding mode {rounding!r}; expected one of "
            f"{sorted(_SUPPORTED_ROUNDING_MODES)}."
        )


@dataclass(frozen=True)
class Account:
    """Typed Canadian investment account wrapper.

    Parameters
    ----------
    account_id
        Unique non-empty identifier (e.g., ``"TFSA-Main"``, ``"NonReg-Margin"``).
    account_type
        Explicit :class:`AccountType` enum value (no silent alias coercion).
    """

    account_id: str
    account_type: AccountType

    def __post_init__(self) -> None:
        if not isinstance(self.account_id, str) or not self.account_id.strip():
            raise DataValidationError("account_id must be a non-empty string.")
        object.__setattr__(self, "account_id", self.account_id.strip())
        if not isinstance(self.account_type, AccountType):
            raise DataValidationError(
                f"account_type must be an AccountType enum; got {self.account_type!r}."
            )


@dataclass(frozen=True)
class Transaction:
    """Immutable cash, FX conversion, fee, or security trade transaction.

    Parameters
    ----------
    timestamp
        Explicit transaction :class:`datetime`. Do not mix timezone-naive and
        timezone-aware timestamps within a single ledger.
    account
        Target :class:`Account`.
    kind
        :class:`TransactionKind` event discriminator.
    currency
        Settlement :class:`Currency` (source currency for ``FX_CONVERSION``).
    fx_rate
        Explicit ``CAD per 1 USD`` exchange rate. Required whenever
        ``currency == Currency.USD`` or ``kind == TransactionKind.FX_CONVERSION``.
    ticker
        Security identifier (required for ``BUY`` / ``SELL``).
    quantity
        Unit quantity transacted (required ``> 0`` for ``BUY`` / ``SELL``).
    price
        Execution price per unit in ``currency`` (required ``> 0`` for
        ``BUY`` / ``SELL``).
    amount
        Gross cash amount in ``currency``. Required ``> 0`` for
        ``DEPOSIT``, ``WITHDRAWAL``, ``FEE``, and ``FX_CONVERSION``. Optional on
        ``BUY`` / ``SELL`` (verified within ``$0.01`` of ``quantity * price`` if
        provided).
    fee
        Explicit commission/fee in ``currency`` (default ``0``). Not permitted
        as a separate non-zero field on ``FEE`` transactions.
    to_currency
        Target currency for ``FX_CONVERSION`` (must differ from ``currency``).
    to_amount
        Optional explicit converted cash credited to ``to_currency`` on
        ``FX_CONVERSION``. If provided, it must agree with ``amount`` and
        ``fx_rate`` within ``$0.01``; when ``None``, it is computed using the
        ledger rounding mode.
    note
        Optional audit note.
    """

    timestamp: datetime
    account: Account
    kind: TransactionKind
    currency: Currency
    fx_rate: NumericInput | None = None
    ticker: str | None = None
    quantity: NumericInput | None = None
    price: NumericInput | None = None
    amount: NumericInput | None = None
    fee: NumericInput = _ZERO
    to_currency: Currency | None = None
    to_amount: NumericInput | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise DataValidationError(
                "Transaction timestamp must be a datetime instance."
            )
        if not isinstance(self.account, Account):
            raise DataValidationError("Transaction account must be an Account.")
        if not isinstance(self.kind, TransactionKind):
            raise DataValidationError(
                f"Transaction kind must be a TransactionKind; got {self.kind!r}."
            )
        if not isinstance(self.currency, Currency):
            raise DataValidationError(
                f"Transaction currency must be Currency.CAD or Currency.USD; "
                f"got {self.currency!r}."
            )

        fee_dec = _require_non_negative_decimal(
            _to_decimal(self.fee, field_name="fee"), field_name="fee"
        )
        object.__setattr__(self, "fee", fee_dec)

        fx_dec: Decimal | None = None
        if self.fx_rate is not None:
            fx_dec = _require_positive_decimal(
                _to_decimal(self.fx_rate, field_name="fx_rate"),
                field_name="fx_rate",
            )
        object.__setattr__(self, "fx_rate", fx_dec)

        if self.currency == Currency.USD and fx_dec is None:
            raise DataValidationError(
                "fx_rate (CAD per 1 USD) is required for USD transactions."
            )

        self._validate_kind_fields(fee_dec, fx_dec)

    def _validate_kind_fields(self, fee_dec: Decimal, fx_dec: Decimal | None) -> None:
        if self.kind in {TransactionKind.BUY, TransactionKind.SELL}:
            self._validate_trade_fields()
            return
        if self.kind == TransactionKind.FX_CONVERSION:
            self._validate_fx_fields(fx_dec)
            return
        self._validate_cash_fields(fee_dec)

    def _validate_trade_fields(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise DataValidationError(
                f"ticker must be a non-empty string for {self.kind.value}."
            )
        object.__setattr__(self, "ticker", self.ticker.strip())
        if self.quantity is None or self.price is None:
            raise DataValidationError(
                f"quantity and price are required for {self.kind.value}."
            )
        if self.to_currency is not None or self.to_amount is not None:
            raise DataValidationError(
                f"to_currency/to_amount are invalid for {self.kind.value}."
            )
        qty_dec = _require_positive_decimal(
            _to_decimal(self.quantity, field_name="quantity"),
            field_name="quantity",
        )
        price_dec = _require_positive_decimal(
            _to_decimal(self.price, field_name="price"),
            field_name="price",
        )
        object.__setattr__(self, "quantity", qty_dec)
        object.__setattr__(self, "price", price_dec)

        if self.amount is not None:
            amt_dec = _require_positive_decimal(
                _to_decimal(self.amount, field_name="amount"),
                field_name="amount",
            )
            gross = qty_dec * price_dec
            if abs(amt_dec - gross) > _PENNY_TOLERANCE:
                raise DataValidationError(
                    f"Provided amount {amt_dec} disagrees with quantity * price "
                    f"({gross}) by more than ${_PENNY_TOLERANCE}."
                )
            object.__setattr__(self, "amount", amt_dec)

    def _validate_fx_fields(self, fx_dec: Decimal | None) -> None:
        if (
            self.ticker is not None
            or self.quantity is not None
            or self.price is not None
        ):
            raise DataValidationError(
                "ticker, quantity, and price must be None for FX_CONVERSION."
            )
        if fx_dec is None:
            raise DataValidationError(
                "fx_rate (CAD per 1 USD) is required for FX_CONVERSION."
            )
        if not isinstance(self.to_currency, Currency):
            raise DataValidationError(
                "to_currency must be Currency.CAD or Currency.USD for FX_CONVERSION."
            )
        if self.to_currency == self.currency:
            raise DataValidationError(
                "FX_CONVERSION requires to_currency to differ from currency."
            )
        if self.amount is None:
            raise DataValidationError("amount is required for FX_CONVERSION.")
        amt_dec = _require_positive_decimal(
            _to_decimal(self.amount, field_name="amount"),
            field_name="amount",
        )
        object.__setattr__(self, "amount", amt_dec)

        expected = (
            amt_dec / fx_dec if self.currency == Currency.CAD else amt_dec * fx_dec
        )
        if self.to_amount is not None:
            to_amt_dec = _require_positive_decimal(
                _to_decimal(self.to_amount, field_name="to_amount"),
                field_name="to_amount",
            )
            if abs(to_amt_dec - expected) > _PENNY_TOLERANCE:
                raise DataValidationError(
                    f"Provided to_amount {to_amt_dec} {self.to_currency.value} "
                    f"disagrees with expected {expected:.4f} at fx_rate={fx_dec}."
                )
            object.__setattr__(self, "to_amount", to_amt_dec)

    def _validate_cash_fields(self, fee_dec: Decimal) -> None:
        if (
            self.ticker is not None
            or self.quantity is not None
            or self.price is not None
            or self.to_currency is not None
            or self.to_amount is not None
        ):
            raise DataValidationError(
                f"Trade and FX fields must be None for {self.kind.value}."
            )
        if self.amount is None:
            raise DataValidationError(f"amount is required for {self.kind.value}.")
        amt_dec = _require_positive_decimal(
            _to_decimal(self.amount, field_name="amount"),
            field_name="amount",
        )
        object.__setattr__(self, "amount", amt_dec)
        if self.kind == TransactionKind.FEE and fee_dec != _ZERO:
            raise DataValidationError(
                "fee field must be 0 for TransactionKind.FEE; encode cost in amount."
            )

    @property
    def cad_per_usd_decimal(self) -> Decimal:
        """Return CAD-per-USD multiplier (``1`` for CAD cash/trades without FX)."""
        if self.currency == Currency.CAD and self.kind != TransactionKind.FX_CONVERSION:
            return _ONE
        assert isinstance(self.fx_rate, Decimal)
        return self.fx_rate


@dataclass(frozen=True)
class MarketSnapshot:
    """Explicit point-in-time market valuation snapshot with no network calls.

    Parameters
    ----------
    as_of
        Snapshot timestamp matching the timezone awareness of ledger entries.
    cad_per_usd
        Explicit spot exchange rate (CAD per 1 USD, strictly ``> 0``).
    cad_prices
        Mapping of CAD-quoted ticker -> unit price in CAD.
    usd_prices
        Mapping of USD-quoted ticker -> unit price in USD.
    """

    as_of: datetime
    cad_per_usd: NumericInput
    cad_prices: Mapping[str, NumericInput] = field(default_factory=dict)
    usd_prices: Mapping[str, NumericInput] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.as_of, datetime):
            raise DataValidationError("MarketSnapshot.as_of must be a datetime.")
        fx_dec = _require_positive_decimal(
            _to_decimal(self.cad_per_usd, field_name="cad_per_usd"),
            field_name="cad_per_usd",
        )
        object.__setattr__(self, "cad_per_usd", fx_dec)

        norm_cad: dict[str, Decimal] = {}
        for ticker, raw_price in self.cad_prices.items():
            clean_ticker = ticker.strip()
            if not clean_ticker:
                raise DataValidationError(
                    "Ticker keys in cad_prices must be non-empty."
                )
            norm_cad[clean_ticker] = _require_positive_decimal(
                _to_decimal(raw_price, field_name=f"cad_prices[{clean_ticker!r}]"),
                field_name=f"cad_prices[{clean_ticker!r}]",
            )

        norm_usd: dict[str, Decimal] = {}
        for ticker, raw_price in self.usd_prices.items():
            clean_ticker = ticker.strip()
            if not clean_ticker:
                raise DataValidationError(
                    "Ticker keys in usd_prices must be non-empty."
                )
            norm_usd[clean_ticker] = _require_positive_decimal(
                _to_decimal(raw_price, field_name=f"usd_prices[{clean_ticker!r}]"),
                field_name=f"usd_prices[{clean_ticker!r}]",
            )

        overlap = set(norm_cad) & set(norm_usd)
        if overlap:
            raise DataValidationError(
                f"Tickers cannot appear in both cad_prices and usd_prices: "
                f"{sorted(overlap)}."
            )

        object.__setattr__(self, "cad_prices", norm_cad)
        object.__setattr__(self, "usd_prices", norm_usd)

    def quote_for(self, ticker: str, currency: Currency) -> Decimal:
        """Return native price for *ticker* in *currency*, or raise on mismatch."""
        price_map = self.cad_prices if currency == Currency.CAD else self.usd_prices
        if ticker not in price_map:
            raise DataValidationError(
                f"Missing {currency.value} quote for held ticker {ticker!r} "
                f"at {self.as_of.isoformat()}."
            )
        price = price_map[ticker]
        assert isinstance(price, Decimal)
        return price


@dataclass(frozen=True)
class CashBalanceSummary:
    """Audited cash balance and conservation check for one ``(account, currency)``."""

    account_id: str
    account_type: AccountType
    currency: Currency
    balance: Decimal
    deposits: Decimal
    withdrawals: Decimal
    buy_principal: Decimal
    sell_proceeds: Decimal
    fees: Decimal
    fx_out: Decimal
    fx_in: Decimal
    expected_balance: Decimal
    conservation_residual: Decimal


@dataclass(frozen=True)
class PositionState:
    """Per-account security position and historical book cost."""

    account_id: str
    account_type: AccountType
    ticker: str
    currency: Currency
    shares: Decimal
    book_cost_native: Decimal
    book_cost_cad: Decimal
    realized_gain_native: Decimal
    realized_gain_cad: Decimal

    @property
    def avg_cost_native(self) -> Decimal:
        """Weighted-average native cost per unit (``0`` when flat)."""
        if self.shares <= _ZERO:
            return _ZERO
        return self.book_cost_native / self.shares

    @property
    def avg_cost_cad(self) -> Decimal:
        """Weighted-average historical-FX CAD cost per unit (``0`` when flat)."""
        if self.shares <= _ZERO:
            return _ZERO
        return self.book_cost_cad / self.shares


@dataclass(frozen=True)
class HoldingValuation:
    """Reconciled holding valuation at snapshot prices and spot FX."""

    account_id: str
    account_type: AccountType
    ticker: str
    currency: Currency
    shares: Decimal
    market_price_native: Decimal
    market_value_native: Decimal
    market_value_cad: Decimal
    book_cost_native: Decimal
    book_cost_cad: Decimal
    unrealized_gain_native: Decimal
    unrealized_gain_cad: Decimal
    realized_gain_native: Decimal
    realized_gain_cad: Decimal


@dataclass(frozen=True)
class AccountValuationSummary:
    """Reconciled account-level valuation combining cash and holdings."""

    account_id: str
    account_type: AccountType
    cash_cad: Decimal
    cash_usd: Decimal
    cash_total_cad: Decimal
    holdings_market_value_cad: Decimal
    total_value_cad: Decimal
    holdings: tuple[HoldingValuation, ...]
    cash_balances: tuple[CashBalanceSummary, ...]


@dataclass(frozen=True)
class ReconciliationReport:
    """Full portfolio reconciliation across accounts, cash currencies, and ACB."""

    as_of: datetime
    cad_per_usd: Decimal
    accounts: dict[str, AccountValuationSummary]
    total_cash_cad: Decimal
    total_cash_usd: Decimal
    total_cash_equivalent_cad: Decimal
    total_holdings_cad: Decimal
    total_portfolio_cad: Decimal
    currency_exposure_cad: dict[Currency, Decimal]
    cash_conservation_residuals: dict[tuple[str, Currency], Decimal]
    all_cash_conserved: bool
    non_reg_tax_acb: dict[str, ACBPosition]
    non_reg_realized_gains_cad: Decimal

    def to_holdings_frame(
        self, *, by: Literal["account_type", "account_id"] = "account_type"
    ) -> pd.DataFrame:
        """Export open equity positions as a CAD DataFrame for rebalancing.

        Parameters
        ----------
        by
            ``"account_type"`` (default) emits :class:`AccountType` wrapper
            labels (e.g., ``"TFSA"``, ``"NON_REG"``) matching
            :func:`allocate_contribution_cash_flow` and
            :func:`build_rebalance_plan`, aggregating across same-type
            accounts. ``"account_id"`` preserves individual account IDs.

        Returns
        -------
        pd.DataFrame
            Columns ``ticker``, ``account``, ``current_value``,
            ``current_shares`` in CAD market value sorted by ``(account, ticker)``.
        """
        if by not in {"account_type", "account_id"}:
            raise DataValidationError(
                f"by must be 'account_type' or 'account_id'; got {by!r}."
            )

        grouped: dict[tuple[str, str], tuple[Decimal, Decimal]] = {}
        for account_id in sorted(self.accounts):
            summary = self.accounts[account_id]
            for holding in summary.holdings:
                if holding.shares <= _ZERO:
                    continue
                acct_label = (
                    holding.account_type.value if by == "account_type" else account_id
                )
                key = (acct_label, holding.ticker)
                prev_val, prev_sh = grouped.get(key, (_ZERO, _ZERO))
                grouped[key] = (
                    prev_val + holding.market_value_cad,
                    prev_sh + holding.shares,
                )

        if not grouped:
            return pd.DataFrame(
                columns=["ticker", "account", "current_value", "current_shares"]
            )
        rows = [
            {
                "ticker": ticker,
                "account": acct_label,
                "current_value": float(val),
                "current_shares": float(sh),
            }
            for (acct_label, ticker), (val, sh) in sorted(grouped.items())
        ]
        return pd.DataFrame(rows)


@dataclass
class _MutableCashBucket:
    account_id: str
    account_type: AccountType
    currency: Currency
    balance: Decimal = _ZERO
    deposits: Decimal = _ZERO
    withdrawals: Decimal = _ZERO
    buy_principal: Decimal = _ZERO
    sell_proceeds: Decimal = _ZERO
    fees: Decimal = _ZERO
    fx_out: Decimal = _ZERO
    fx_in: Decimal = _ZERO

    def expected_balance(self) -> Decimal:
        return (
            self.deposits
            - self.withdrawals
            - self.buy_principal
            + self.sell_proceeds
            - self.fees
            - self.fx_out
            + self.fx_in
        )

    def to_summary(self) -> CashBalanceSummary:
        expected = self.expected_balance()
        return CashBalanceSummary(
            account_id=self.account_id,
            account_type=self.account_type,
            currency=self.currency,
            balance=self.balance,
            deposits=self.deposits,
            withdrawals=self.withdrawals,
            buy_principal=self.buy_principal,
            sell_proceeds=self.sell_proceeds,
            fees=self.fees,
            fx_out=self.fx_out,
            fx_in=self.fx_in,
            expected_balance=expected,
            conservation_residual=self.balance - expected,
        )


@dataclass
class _MutablePosition:
    account_id: str
    account_type: AccountType
    ticker: str
    currency: Currency
    shares: Decimal = _ZERO
    book_cost_native: Decimal = _ZERO
    book_cost_cad: Decimal = _ZERO
    realized_gain_native: Decimal = _ZERO
    realized_gain_cad: Decimal = _ZERO

    def to_state(self) -> PositionState:
        return PositionState(
            account_id=self.account_id,
            account_type=self.account_type,
            ticker=self.ticker,
            currency=self.currency,
            shares=self.shares,
            book_cost_native=self.book_cost_native,
            book_cost_cad=self.book_cost_cad,
            realized_gain_native=self.realized_gain_native,
            realized_gain_cad=self.realized_gain_cad,
        )


class PortfolioLedger:
    """Deterministic multi-account CAD/USD portfolio transaction ledger.

    Parameters
    ----------
    allow_negative_cash
        When ``False`` (default), any transaction causing a negative cash balance
        in an ``(account_id, currency)`` bucket fails and rolls back cleanly.
    rounding
        Explicit :mod:`decimal` rounding mode applied to cent quantization and
        derived FX conversion proceeds (default ``ROUND_HALF_EVEN``).
    """

    def __init__(
        self,
        *,
        allow_negative_cash: bool = False,
        rounding: str = ROUND_HALF_EVEN,
    ) -> None:
        _validate_rounding_mode(rounding)
        self._allow_negative_cash = allow_negative_cash
        self._rounding = rounding
        self._raw_transactions: list[tuple[int, Transaction]] = []
        self._ordered_transactions: list[Transaction] = []
        self._accounts: dict[str, Account] = {}
        self._cash: dict[tuple[str, Currency], _MutableCashBucket] = {}
        self._positions: dict[tuple[str, str], _MutablePosition] = {}
        self._tax_acb: NonRegACBTracker = _new_non_reg_acb_tracker()

    @property
    def rounding(self) -> str:
        """Return configured Decimal rounding mode."""
        return self._rounding

    @property
    def allow_negative_cash(self) -> bool:
        """Whether negative cash balances are permitted."""
        return self._allow_negative_cash

    @property
    def accounts(self) -> dict[str, Account]:
        """Registered accounts keyed by ``account_id``."""
        return dict(self._accounts)

    @property
    def transactions(self) -> list[Transaction]:
        """Chronologically ordered copy of all recorded transactions."""
        return list(self._ordered_transactions)

    @property
    def non_reg_tax_acb(self) -> dict[str, ACBPosition]:
        """Taxpayer-level Non-Registered CRA weighted-average ACB positions (CAD)."""
        return self._tax_acb.positions

    def record(self, transaction: Transaction) -> None:
        """Record one transaction and atomically validate & replay the ledger."""
        self.record_many([transaction])

    def record_many(self, transactions: Iterable[Transaction]) -> None:
        """Record multiple transactions and atomically replay the ledger."""
        batch = list(transactions)
        if not batch:
            return
        for tx in batch:
            if not isinstance(tx, Transaction):
                raise DataValidationError(
                    f"Expected Transaction instance; got {type(tx).__name__}."
                )

        candidate_raw = list(self._raw_transactions)
        next_idx = len(candidate_raw)
        for tx in batch:
            candidate_raw.append((next_idx, tx))
            next_idx += 1

        replayed = self._replay(candidate_raw)
        self._raw_transactions = candidate_raw
        (
            self._ordered_transactions,
            self._accounts,
            self._cash,
            self._positions,
            self._tax_acb,
        ) = replayed

    def cash_balance(self, account_id: str, currency: Currency) -> Decimal:
        """Return current cash balance for ``(account_id, currency)``."""
        bucket = self._cash.get((account_id, currency))
        return bucket.balance if bucket is not None else _ZERO

    def cash_balances(
        self, account_id: str | None = None
    ) -> dict[tuple[str, Currency], CashBalanceSummary]:
        """Return audited cash summaries, optionally filtered by *account_id*."""
        return {
            key: bucket.to_summary()
            for key, bucket in sorted(
                self._cash.items(), key=lambda kv: (kv[0][0], kv[0][1].value)
            )
            if account_id is None or key[0] == account_id
        }

    def positions(self, account_id: str | None = None) -> list[PositionState]:
        """Return position states sorted by ``(account_id, ticker)``."""
        return [
            pos.to_state()
            for (_, _), pos in sorted(self._positions.items())
            if account_id is None or pos.account_id == account_id
        ]

    def reconcile(self, snapshot: MarketSnapshot) -> ReconciliationReport:
        """Reconcile cash ledgers, positions, and tax ACB against *snapshot*."""
        if not isinstance(snapshot, MarketSnapshot):
            raise DataValidationError("snapshot must be a MarketSnapshot instance.")

        if self._ordered_transactions:
            snap_aware = snapshot.as_of.tzinfo is not None
            tx_aware = self._ordered_transactions[0].timestamp.tzinfo is not None
            if snap_aware != tx_aware:
                raise DataValidationError(
                    "MarketSnapshot.as_of timezone awareness must match ledger "
                    "transaction timestamps."
                )
            latest_ts = self._ordered_transactions[-1].timestamp
            if snapshot.as_of < latest_ts:
                raise DataValidationError(
                    f"MarketSnapshot.as_of ({snapshot.as_of.isoformat()}) cannot "
                    f"precede latest recorded transaction ({latest_ts.isoformat()})."
                )

        assert isinstance(snapshot.cad_per_usd, Decimal)
        spot_fx = snapshot.cad_per_usd

        cash_summaries = self.cash_balances()
        cash_residuals: dict[tuple[str, Currency], Decimal] = {
            key: summary.conservation_residual
            for key, summary in cash_summaries.items()
        }
        all_cash_conserved = all(res == _ZERO for res in cash_residuals.values())

        accounts_out: dict[str, AccountValuationSummary] = {}
        total_cash_cad = _ZERO
        total_cash_usd = _ZERO
        total_cash_equiv_cad = _ZERO
        total_holdings_cad = _ZERO
        cad_exposure_cad = _ZERO
        usd_exposure_cad = _ZERO

        for account_id in sorted(self._accounts):
            account = self._accounts[account_id]
            acct_cash_cad = self.cash_balance(account_id, Currency.CAD)
            acct_cash_usd = self.cash_balance(account_id, Currency.USD)
            acct_usd_cash_cad = _quantize_money(acct_cash_usd * spot_fx, self._rounding)
            acct_cash_total_cad = acct_cash_cad + acct_usd_cash_cad

            total_cash_cad += acct_cash_cad
            total_cash_usd += acct_cash_usd
            total_cash_equiv_cad += acct_cash_total_cad
            cad_exposure_cad += acct_cash_cad
            usd_exposure_cad += acct_usd_cash_cad

            acct_holdings: list[HoldingValuation] = []
            acct_holdings_cad = _ZERO

            for pos in self.positions(account_id):
                if pos.shares <= _ZERO:
                    continue
                quote_native = snapshot.quote_for(pos.ticker, pos.currency)
                mv_native = _quantize_money(pos.shares * quote_native, self._rounding)
                if pos.currency == Currency.CAD:
                    mv_cad = mv_native
                    cad_exposure_cad += mv_cad
                else:
                    mv_cad = _quantize_money(mv_native * spot_fx, self._rounding)
                    usd_exposure_cad += mv_cad

                acct_holdings_cad += mv_cad
                acct_holdings.append(
                    HoldingValuation(
                        account_id=account_id,
                        account_type=account.account_type,
                        ticker=pos.ticker,
                        currency=pos.currency,
                        shares=pos.shares,
                        market_price_native=quote_native,
                        market_value_native=mv_native,
                        market_value_cad=mv_cad,
                        book_cost_native=pos.book_cost_native,
                        book_cost_cad=pos.book_cost_cad,
                        unrealized_gain_native=mv_native - pos.book_cost_native,
                        unrealized_gain_cad=mv_cad - pos.book_cost_cad,
                        realized_gain_native=pos.realized_gain_native,
                        realized_gain_cad=pos.realized_gain_cad,
                    )
                )

            acct_cash_list = tuple(
                summary
                for (cid, _), summary in cash_summaries.items()
                if cid == account_id
            )
            total_holdings_cad += acct_holdings_cad
            accounts_out[account_id] = AccountValuationSummary(
                account_id=account_id,
                account_type=account.account_type,
                cash_cad=acct_cash_cad,
                cash_usd=acct_cash_usd,
                cash_total_cad=acct_cash_total_cad,
                holdings_market_value_cad=acct_holdings_cad,
                total_value_cad=acct_cash_total_cad + acct_holdings_cad,
                holdings=tuple(acct_holdings),
                cash_balances=acct_cash_list,
            )

        # Sum per-account CAD-translated cash equivalents so portfolio totals tie
        # out identically to the sum of account valuations and currency exposures.
        non_reg_realized = Decimal(
            str(
                sum(
                    trade.realized_gain_loss
                    for trade in self._tax_acb.trades
                    if trade.is_sell
                )
            )
        )
        non_reg_realized = _quantize_money(non_reg_realized, self._rounding)

        return ReconciliationReport(
            as_of=snapshot.as_of,
            cad_per_usd=spot_fx,
            accounts=accounts_out,
            total_cash_cad=total_cash_cad,
            total_cash_usd=total_cash_usd,
            total_cash_equivalent_cad=total_cash_equiv_cad,
            total_holdings_cad=total_holdings_cad,
            total_portfolio_cad=total_cash_equiv_cad + total_holdings_cad,
            currency_exposure_cad={
                Currency.CAD: cad_exposure_cad,
                Currency.USD: usd_exposure_cad,
            },
            cash_conservation_residuals=cash_residuals,
            all_cash_conserved=all_cash_conserved,
            non_reg_tax_acb=self._tax_acb.positions,
            non_reg_realized_gains_cad=non_reg_realized,
        )

    def _replay(
        self, raw: list[tuple[int, Transaction]]
    ) -> tuple[
        list[Transaction],
        dict[str, Account],
        dict[tuple[str, Currency], _MutableCashBucket],
        dict[tuple[str, str], _MutablePosition],
        NonRegACBTracker,
    ]:
        # Validate timezone uniformity and account consistency before replaying.
        aware_flags = {tx.timestamp.tzinfo is not None for _, tx in raw}
        if len(aware_flags) > 1:
            raise DataValidationError(
                "Cannot mix timezone-naive and timezone-aware timestamps in ledger."
            )

        accounts: dict[str, Account] = {}
        for _, tx in raw:
            existing = accounts.get(tx.account.account_id)
            if (
                existing is not None
                and existing.account_type != tx.account.account_type
            ):
                raise DataValidationError(
                    f"Account {tx.account.account_id!r} cannot switch AccountType "
                    f"from {existing.account_type.value} to "
                    f"{tx.account.account_type.value}."
                )
            accounts[tx.account.account_id] = tx.account

        ordered = [
            tx for _, tx in sorted(raw, key=lambda item: (item[1].timestamp, item[0]))
        ]
        cash: dict[tuple[str, Currency], _MutableCashBucket] = {}
        positions: dict[tuple[str, str], _MutablePosition] = {}
        tax_acb = _new_non_reg_acb_tracker()

        for tx in ordered:
            self._apply_tx(tx, cash=cash, positions=positions, tax_acb=tax_acb)

        return ordered, accounts, cash, positions, tax_acb

    def _get_or_create_bucket(
        self,
        cash: dict[tuple[str, Currency], _MutableCashBucket],
        account: Account,
        currency: Currency,
    ) -> _MutableCashBucket:
        key = (account.account_id, currency)
        if key not in cash:
            cash[key] = _MutableCashBucket(
                account_id=account.account_id,
                account_type=account.account_type,
                currency=currency,
            )
        return cash[key]

    def _check_non_negative_cash(
        self, bucket: _MutableCashBucket, tx: Transaction
    ) -> None:
        if not self._allow_negative_cash and bucket.balance < _ZERO:
            raise DataValidationError(
                f"Insufficient {bucket.currency.value} cash in account "
                f"{bucket.account_id!r} at {tx.timestamp.isoformat()} "
                f"({tx.kind.value}): balance would become {bucket.balance}."
            )

    def _apply_tx(
        self,
        tx: Transaction,
        *,
        cash: dict[tuple[str, Currency], _MutableCashBucket],
        positions: dict[tuple[str, str], _MutablePosition],
        tax_acb: NonRegACBTracker,
    ) -> None:
        fee = tx.fee
        assert isinstance(fee, Decimal)
        fee_q = _quantize_money(fee, self._rounding)
        bucket = self._get_or_create_bucket(cash, tx.account, tx.currency)

        if tx.kind == TransactionKind.DEPOSIT:
            assert isinstance(tx.amount, Decimal)
            amt = _quantize_money(tx.amount, self._rounding)
            bucket.deposits += amt
            bucket.fees += fee_q
            bucket.balance += amt - fee_q
            self._check_non_negative_cash(bucket, tx)
            return

        if tx.kind == TransactionKind.WITHDRAWAL:
            assert isinstance(tx.amount, Decimal)
            amt = _quantize_money(tx.amount, self._rounding)
            bucket.withdrawals += amt
            bucket.fees += fee_q
            bucket.balance -= amt + fee_q
            self._check_non_negative_cash(bucket, tx)
            return

        if tx.kind == TransactionKind.FEE:
            assert isinstance(tx.amount, Decimal)
            amt = _quantize_money(tx.amount, self._rounding)
            bucket.fees += amt
            bucket.balance -= amt
            self._check_non_negative_cash(bucket, tx)
            return

        if tx.kind == TransactionKind.FX_CONVERSION:
            assert isinstance(tx.amount, Decimal)
            assert isinstance(tx.to_currency, Currency)
            assert isinstance(tx.fx_rate, Decimal)
            src_amt = _quantize_money(tx.amount, self._rounding)
            if tx.to_amount is not None:
                assert isinstance(tx.to_amount, Decimal)
                dst_amt = _quantize_money(tx.to_amount, self._rounding)
            else:
                raw_dst = (
                    src_amt / tx.fx_rate
                    if tx.currency == Currency.CAD
                    else src_amt * tx.fx_rate
                )
                dst_amt = _quantize_money(raw_dst, self._rounding)

            bucket.fx_out += src_amt
            bucket.fees += fee_q
            bucket.balance -= src_amt + fee_q
            self._check_non_negative_cash(bucket, tx)

            dst_bucket = self._get_or_create_bucket(cash, tx.account, tx.to_currency)
            dst_bucket.fx_in += dst_amt
            dst_bucket.balance += dst_amt
            return

        # Trade execution (BUY or SELL)
        assert tx.ticker is not None
        assert isinstance(tx.quantity, Decimal)
        assert isinstance(tx.price, Decimal)
        principal = _quantize_money(tx.quantity * tx.price, self._rounding)
        fx_cad_per_usd = tx.cad_per_usd_decimal
        pos_key = (tx.account.account_id, tx.ticker)

        if pos_key in positions and positions[pos_key].currency != tx.currency:
            raise DataValidationError(
                f"Position ({tx.account.account_id!r}, {tx.ticker!r}) cannot "
                f"mix currencies {positions[pos_key].currency.value} and "
                f"{tx.currency.value}."
            )

        if pos_key not in positions:
            positions[pos_key] = _MutablePosition(
                account_id=tx.account.account_id,
                account_type=tx.account.account_type,
                ticker=tx.ticker,
                currency=tx.currency,
            )
        pos = positions[pos_key]

        if tx.kind == TransactionKind.BUY:
            total_outflow = principal + fee_q
            bucket.buy_principal += principal
            bucket.fees += fee_q
            bucket.balance -= total_outflow
            self._check_non_negative_cash(bucket, tx)

            cad_principal = _quantize_money(principal * fx_cad_per_usd, self._rounding)
            cad_fee = _quantize_money(fee_q * fx_cad_per_usd, self._rounding)
            cad_cost = cad_principal + cad_fee
            pos.shares += tx.quantity
            pos.book_cost_native += total_outflow
            pos.book_cost_cad += cad_cost

            if tx.account.account_type == AccountType.NON_REG:
                cad_price = float(cad_principal / tx.quantity)
                tax_acb.record_buy(
                    ticker=tx.ticker,
                    shares=float(tx.quantity),
                    price_per_share=cad_price,
                    trade_date=tx.timestamp.date(),
                    commission=float(cad_fee),
                    account_type=AccountType.NON_REG.value,
                )
            return

        # SELL
        if tx.quantity > pos.shares:
            raise DataValidationError(
                f"Cannot sell {tx.quantity} shares of {tx.ticker!r} in account "
                f"{tx.account.account_id!r} at {tx.timestamp.isoformat()}: only "
                f"{pos.shares} shares held."
            )

        net_proceeds_native = principal - fee_q
        bucket.sell_proceeds += principal
        bucket.fees += fee_q
        bucket.balance += net_proceeds_native
        self._check_non_negative_cash(bucket, tx)

        if tx.quantity == pos.shares:
            removed_native = pos.book_cost_native
            removed_cad = pos.book_cost_cad
            pos.shares = _ZERO
            pos.book_cost_native = _ZERO
            pos.book_cost_cad = _ZERO
        else:
            removed_native = min(
                pos.book_cost_native,
                _quantize_money(
                    (pos.book_cost_native * tx.quantity) / pos.shares,
                    self._rounding,
                ),
            )
            removed_cad = min(
                pos.book_cost_cad,
                _quantize_money(
                    (pos.book_cost_cad * tx.quantity) / pos.shares,
                    self._rounding,
                ),
            )
            pos.shares -= tx.quantity
            pos.book_cost_native -= removed_native
            pos.book_cost_cad -= removed_cad

        cad_principal = _quantize_money(principal * fx_cad_per_usd, self._rounding)
        cad_fee = _quantize_money(fee_q * fx_cad_per_usd, self._rounding)
        proceeds_cad = cad_principal - cad_fee
        pos.realized_gain_native += net_proceeds_native - removed_native
        pos.realized_gain_cad += proceeds_cad - removed_cad

        if tx.account.account_type == AccountType.NON_REG:
            cad_price = float(cad_principal / tx.quantity)
            tax_acb.record_sell(
                ticker=tx.ticker,
                shares=float(tx.quantity),
                price_per_share=cad_price,
                trade_date=tx.timestamp.date(),
                commission=float(cad_fee),
                account_type=AccountType.NON_REG.value,
            )


__all__ = [
    "Account",
    "AccountValuationSummary",
    "CashBalanceSummary",
    "Currency",
    "HoldingValuation",
    "MarketSnapshot",
    "PortfolioLedger",
    "PositionState",
    "ReconciliationReport",
    "Transaction",
    "TransactionKind",
]
