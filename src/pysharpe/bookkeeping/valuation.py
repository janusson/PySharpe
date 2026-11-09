"""Mark-to-market valuation of ledger holdings, in base currency (CAD).

Turns recorded holdings, their cost base, and cash into market value,
unrealized gain/loss, and allocation weights.  Everything here is a pure
function of its inputs: no price is fetched, no rate is inferred, and no
missing observation is treated as zero.

Conventions
-----------
* **Market prices are per share in base currency (CAD).**  A 1.40 USD->CAD
  rate on the trade that *bought* a security is a historical book-cost fact; it
  says nothing about the security's price today.  Valuation therefore never
  reuses :attr:`Transaction.fx_rate`.  Translate prices upstream (the data
  pipeline already produces CAD series) and pass them here.
* **Portfolio value includes cash.**  ``value_portfolio`` takes cash already
  expressed in base currency.  ``PortfolioLedger.valuation`` translates
  per-currency balances at the current ``fx_rates`` the caller supplies before
  calling it, so nothing is converted here.
* **Missing prices stay unknown.**  A held security without a usable price is
  reported in :attr:`Valuation.unpriced_tickers`; it is excluded from totals
  rather than counted as zero, and :attr:`Valuation.is_complete` turns false.
* **Cost basis is required.**  A holding without a cost-basis entry raises
  instead of reporting its entire market value as a gain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from pysharpe.bookkeeping.accounts import Account
from pysharpe.exceptions import DataValidationError


def _weights[K](values: Mapping[K, float], total: float) -> dict[K, float]:
    """Return *values* as fractions of *total*, or empty when *total* is zero."""

    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}


def _market_price(ticker: str, prices: Mapping[str, float]) -> float | None:
    """Return the usable CAD price for *ticker*, or ``None`` when unknown.

    A ``NaN`` observation means "no data for this security", not "worth zero",
    so it is reported as unknown.  A negative or infinite price is a caller
    error and raises.
    """

    if ticker not in prices:
        return None
    raw = prices[ticker]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DataValidationError(f"price for {ticker} must be a number; got {raw!r}")
    price = float(raw)
    if math.isnan(price):
        return None
    if not math.isfinite(price) or price < 0:
        raise DataValidationError(
            f"price for {ticker} must be finite and non-negative; got {raw!r}"
        )
    return price


@dataclass(frozen=True)
class PositionValuation:
    """One priced holding, marked to market.

    Attributes:
        account (Account): Individual account holding the security.
        ticker (str): Security symbol.
        quantity (float): Shares held.
        market_price (float): Price per share in base currency (CAD).
        book_cost (float): Book cost of the holding in base currency (CAD).
            For registered accounts this is book value, never a tax figure, and
            it stays account-level even for non-registered accounts (the pooled
            taxable ACB is reported by ``PortfolioLedger.tax_acb``).
    """

    account: Account
    ticker: str
    quantity: float
    market_price: float
    book_cost: float

    @property
    def market_value(self) -> float:
        """Mark-to-market value of the holding in base currency (CAD)."""

        return self.quantity * self.market_price

    @property
    def unrealized_gain_loss(self) -> float:
        """Market value less cost base, in base currency (CAD)."""

        return self.market_value - self.book_cost

    @property
    def unrealized_gain_loss_pct(self) -> float:
        """Unrealized gain/loss as a fraction of cost base.

        Returns ``0.0`` when the cost base is zero, matching
        :meth:`pysharpe.execution.tax_tracker.ACBTracker.get_unrealized_gain_loss_pct`.
        """

        if self.book_cost <= 0:
            return 0.0
        return self.unrealized_gain_loss / self.book_cost


@dataclass(frozen=True)
class Valuation:
    """Result of marking a portfolio to market at one point in time.

    Attributes:
        positions (tuple[PositionValuation, ...]): Priced holdings, sorted by
            registration, account id, then ticker.
        cash (dict[Account, float]): Cash per account, in base currency.
        unpriced_tickers (tuple[str, ...]): Held securities with no usable
            price. Their value is unknown, not zero, and they are excluded
            from every total.
    """

    positions: tuple[PositionValuation, ...]
    cash: dict[Account, float]
    unpriced_tickers: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """True when every held security has a price."""

        return not self.unpriced_tickers

    @property
    def total_cash(self) -> float:
        """Cash across all accounts, in base currency (CAD)."""

        return sum(self.cash.values())

    @property
    def total_market_value(self) -> float:
        """Known portfolio value: priced holdings plus cash, in CAD.

        Understates the portfolio when :attr:`is_complete` is false; missing
        prices are never treated as zero.
        """

        positions_value = sum(position.market_value for position in self.positions)
        return positions_value + self.total_cash

    @property
    def total_book_cost(self) -> float:
        """Book cost of the priced holdings, in base currency (CAD)."""

        return sum(position.book_cost for position in self.positions)

    @property
    def total_unrealized_gain_loss(self) -> float:
        """Unrealized gain or loss across the priced holdings, in CAD."""

        return sum(position.unrealized_gain_loss for position in self.positions)

    def account_values(self) -> dict[Account, float]:
        """Return each account's value in CAD, including its cash.

        Accounts holding only cash appear with the cash amount; accounts with
        nothing at all are omitted.
        """

        values: dict[Account, float] = dict(self.cash)
        for position in self.positions:
            values[position.account] = (
                values.get(position.account, 0.0) + position.market_value
            )
        return values

    def allocation_by_account(self) -> dict[Account, float]:
        """Return each account's share of :attr:`total_market_value`.

        Cash is included.  Fractions sum to ``1.0`` for a non-empty portfolio
        and are empty when the portfolio is worth nothing.
        """

        return _weights(self.account_values(), self.total_market_value)

    def allocation_by_ticker(self) -> dict[str, float]:
        """Return each security's share of :attr:`total_market_value`.

        The same ticker held in several accounts is combined.  Cash has no
        ticker, so these fractions sum to less than ``1.0`` whenever the
        portfolio holds cash.
        """

        values: dict[str, float] = {}
        for position in self.positions:
            values[position.ticker] = (
                values.get(position.ticker, 0.0) + position.market_value
            )
        return _weights(values, self.total_market_value)

    def to_frame(self) -> pd.DataFrame:
        """Return the priced holdings as a DataFrame, one row per position.

        Columns are ``account`` (the id), ``registration``, ``ticker``,
        ``quantity``, ``market_price``, ``market_value``, ``book_cost`` and
        ``unrealized_gain_loss``.  Cash is not a row; see :attr:`total_cash` or
        :meth:`account_values`.
        """

        columns = [
            "account",
            "registration",
            "ticker",
            "quantity",
            "market_price",
            "market_value",
            "book_cost",
            "unrealized_gain_loss",
        ]
        rows = [
            {
                "account": position.account.id,
                "registration": position.account.registration,
                "ticker": position.ticker,
                "quantity": position.quantity,
                "market_price": position.market_price,
                "market_value": position.market_value,
                "book_cost": position.book_cost,
                "unrealized_gain_loss": position.unrealized_gain_loss,
            }
            for position in self.positions
        ]
        return pd.DataFrame(rows, columns=columns)


def value_portfolio(
    holdings: Mapping[tuple[Account, str], float],
    cash: Mapping[Account, float],
    prices: Mapping[str, float],
    cost_basis: Mapping[tuple[Account, str], float],
) -> Valuation:
    """Mark *holdings* and *cash* to market using base-currency *prices*.

    Args:
        holdings (Mapping[tuple[Account, str], float]): Share counts keyed by
            ``(account, ticker)``, as produced by
            :attr:`pysharpe.bookkeeping.PortfolioLedger.positions`.
        cash (Mapping[Account, float]): Cash per account in base currency.
        prices (Mapping[str, float]): Price per share in base currency (CAD).
            Tickers that are not held are ignored; a ``NaN`` entry means the
            price is unknown.
        cost_basis (Mapping[tuple[Account, str], float]): Book cost per
            ``(account, ticker)``, in base currency.  Every held pair must be
            present.

    Returns:
        Valuation: Priced positions, cash, and the securities left unpriced.

    Raises:
        DataValidationError: If a price is negative or non-finite, if cash is
            not finite, or if a held holding has no cost-basis entry.
    """

    for account, amount in cash.items():
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise DataValidationError(
                f"cash for {account.id} must be a number; got {amount!r}"
            )
        if not math.isfinite(float(amount)):
            raise DataValidationError(
                f"cash for {account.id} must be finite; got {amount!r}"
            )

    priced: list[PositionValuation] = []
    unpriced: list[str] = []
    for (account, ticker), quantity in sorted(
        holdings.items(),
        key=lambda item: (
            item[0][0].registration.value,
            item[0][0].id,
            item[0][1],
        ),
    ):
        price = _market_price(ticker, prices)
        if price is None:
            unpriced.append(ticker)
            continue
        if (account, ticker) not in cost_basis:
            raise DataValidationError(
                f"no cost basis for {ticker} in {account.id}; "
                "unrealized gain/loss would be unknown"
            )
        priced.append(
            PositionValuation(
                account=account,
                ticker=ticker,
                quantity=quantity,
                market_price=price,
                book_cost=cost_basis[(account, ticker)],
            )
        )

    return Valuation(
        positions=tuple(priced),
        cash=dict(cash),
        unpriced_tickers=tuple(unpriced),
    )
