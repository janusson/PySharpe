"""Reproducible Canadian multi-account portfolio reference example.

One synthetic Canadian taxpayer, three accounts, two securities, a fixed FX
rate, and no fees.  The example shows the accepted tax-ACB ledger
(:class:`pysharpe.guardrails.tax_compliance.ACBTracker`) working with an
explicit per-account CAD/USD cash and book-cost ledger.

The package has no account-level cash ledger, so one is implemented here
rather than in ``src/``: the example is meant to be read end to end, and its
only borrowed implementation detail is the CRA pooled ACB ledger.

Scenario
--------
1. ``TFSA-A`` — contribute CAD 10,000; convert CAD 6,750 into USD 5,000;
   buy 10 units of the USD security at USD 50.
2. ``TAXABLE-A`` — contribute CAD 10,000; buy 100 shares of the CAD security
   at CAD 100; later sell those 100 shares at CAD 150.
3. ``TAXABLE-B`` — contribute CAD 20,000; buy 100 shares of the identical CAD
   security at CAD 200.

Both taxable purchases settle before the taxable sale.  Valuation uses
CAD 150/share for the CAD security, USD 50/share for the USD security, and
CAD 1.35 per USD, giving a total portfolio value of CAD 40,000.

Two ledgers, deliberately separate
----------------------------------
**Account book cost** is what *this account* paid for the units it holds, in
CAD at the acquisition FX rate.  It is a brokerage-statement figure and is
used for performance reporting.

**Tax ACB** is a CRA concept (ITA s. 47(1)): identical property held by one
taxpayer is a *single* pool across all non-registered accounts.  Acquisitions
add cost, dispositions remove cost pro rata, and the gain is measured against
the pooled average.  TFSA and RRSP holdings never enter the pool.

Here the pool holds 200 shares costing CAD 30,000 (ACB CAD 150/share), so
selling 100 shares at CAD 150 realizes **zero** taxable gain even though
``TAXABLE-A`` bought them at CAD 100 (account-level gain CAD 5,000) and
``TAXABLE-B`` still reports CAD 20,000 of book cost on 100 shares whose
pooled tax ACB is CAD 15,000.

Running it::

    uv run python examples/canadian_portfolio_reference.py

All data is synthetic: no network access, no brokerage integration, no real
securities, and no personal information.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from pysharpe.execution.tax_tracker import ACBPosition
from pysharpe.guardrails.tax_compliance import ACBTracker

Currency = Literal["CAD", "USD"]

# Synthetic instruments and settlement dates.  None of these correspond to a
# real security, price, or account.
CAD_SECURITY = "SYN-CAD.TO"
USD_SECURITY = "SYN-USD"
PURCHASE_DATE = date(2025, 1, 15)
SALE_DATE = date(2025, 2, 18)

#: Transaction and valuation FX rate: CAD per USD.
CAD_PER_USD = 1.35

#: Account names in reporting order.
ACCOUNT_ORDER = ("TFSA-A", "TAXABLE-A", "TAXABLE-B")

_CENTS = 2
_REPORT_WIDTH = 13


def _money(amount: float) -> float:
    """Round a dollar amount to the posting unit (cents)."""
    return round(amount, _CENTS)


@dataclass(frozen=True)
class MarketSnapshot:
    """Explicit prices used for valuation and FX conversion.

    Attributes
    ----------
    cad_prices : Mapping[str, float]
        CAD price per share, keyed by ticker.
    usd_prices : Mapping[str, float]
        USD price per share, keyed by ticker.
    cad_per_usd : float
        Valuation FX rate (CAD per USD).
    """

    cad_prices: Mapping[str, float]
    usd_prices: Mapping[str, float]
    cad_per_usd: float

    def price(self, ticker: str, currency: Currency) -> float:
        """Return the price per share of *ticker* in *currency*."""
        prices = self.cad_prices if currency == "CAD" else self.usd_prices
        if ticker not in prices:
            raise ValueError(f"No {currency} price for {ticker!r}.")
        return prices[ticker]

    def to_cad(self, amount: float, currency: Currency) -> float:
        """Convert *amount* of *currency* into CAD at the valuation rate."""
        if currency == "USD":
            return _money(amount * self.cad_per_usd)
        return _money(amount)

    def security_value_cad(
        self, ticker: str, quantity: float, currency: Currency
    ) -> float:
        """Return the CAD market value of *quantity* shares of *ticker*."""
        return self.to_cad(quantity * self.price(ticker, currency), currency)


@dataclass(frozen=True)
class LedgerEntry:
    """One posted movement in an account ledger.

    Attributes
    ----------
    date : date
        Settlement date of the movement.
    account : str
        Account name that posted the entry.
    kind : str
        ``"CONTRIBUTION"``, ``"FX_CONVERSION"``, ``"BUY"``, or ``"SELL"``.
    currency : Currency
        Currency of ``amount``.
    amount : float
        Signed amount in ``currency`` (negative for cash outflows).
    cad_value : float
        Signed CAD equivalent of ``amount`` at the rate used when posting.
    ticker : str | None
        Security traded, for ``BUY`` and ``SELL`` entries.
    quantity : float
        Units traded (always positive), for ``BUY`` and ``SELL`` entries.
    """

    date: date
    account: str
    kind: str
    currency: Currency
    amount: float
    cad_value: float
    ticker: str | None = None
    quantity: float = 0.0


@dataclass
class InvestmentAccount:
    """A single account holding CAD and USD cash plus long positions.

    Book cost is tracked per account, in CAD, at the acquisition FX rate.  It
    is **not** the taxpayer's tax ACB: see the module docstring.

    Attributes
    ----------
    name : str
        Unique account label.
    account_type : str
        ``"TFSA"`` or ``"TAXABLE"``.
    cash : dict[Currency, float]
        Settled cash per currency.
    quantities : dict[str, float]
        Units held per ticker.
    ticker_currency : dict[str, Currency]
        Trading currency of each held ticker.
    book_cost_cad : dict[str, float]
        CAD acquisition book cost per ticker.
    contributions_cad : float
        Cumulative external CAD contributions.
    realized_gain_cad : float
        Cumulative account-level realized gain/loss in CAD, measured against
        this account's book cost.
    journal : list[LedgerEntry]
        Every posted movement, in posting order.
    """

    name: str
    account_type: str = "TAXABLE"
    cash: dict[Currency, float] = field(
        default_factory=lambda: {"CAD": 0.0, "USD": 0.0}
    )
    quantities: dict[str, float] = field(default_factory=dict)
    ticker_currency: dict[str, Currency] = field(default_factory=dict)
    book_cost_cad: dict[str, float] = field(default_factory=dict)
    contributions_cad: float = 0.0
    realized_gain_cad: float = 0.0
    journal: list[LedgerEntry] = field(default_factory=list)

    @property
    def is_taxable(self) -> bool:
        """``True`` when dispositions in this account are taxable."""
        return self.account_type.upper() != "TFSA"

    # -- Cash movements ------------------------------------------------------

    def contribute(self, amount_cad: float, on: date) -> None:
        """Post an external CAD contribution."""
        if amount_cad <= 0:
            raise ValueError(f"Contribution must be positive; got {amount_cad}.")
        self.cash["CAD"] = _money(self.cash["CAD"] + amount_cad)
        self.contributions_cad = _money(self.contributions_cad + amount_cad)
        self._post(
            LedgerEntry(
                date=on,
                account=self.name,
                kind="CONTRIBUTION",
                currency="CAD",
                amount=_money(amount_cad),
                cad_value=_money(amount_cad),
            )
        )

    def convert(
        self,
        amount: float,
        source: Currency,
        target: Currency,
        cad_per_usd: float,
        on: date,
    ) -> float:
        """Convert settled cash between currencies at *cad_per_usd*.

        Returns
        -------
        float
            Amount received in the target currency.
        """
        if source == target:
            raise ValueError("Source and target currencies must differ.")
        if amount <= 0:
            raise ValueError(f"Conversion amount must be positive; got {amount}.")
        if cad_per_usd <= 0:
            raise ValueError(f"cad_per_usd must be positive; got {cad_per_usd}.")
        if amount > self.cash[source]:
            raise ValueError(
                f"{self.name}: cannot convert {amount:.2f} {source}; "
                f"only {self.cash[source]:.2f} available."
            )

        cad_value = _money(amount * cad_per_usd) if source == "USD" else _money(amount)
        received = _money(cad_value / cad_per_usd) if target == "USD" else cad_value

        self.cash[source] = _money(self.cash[source] - amount)
        self.cash[target] = _money(self.cash[target] + received)
        self._post(
            LedgerEntry(
                date=on,
                account=self.name,
                kind="FX_CONVERSION",
                currency=source,
                amount=-_money(amount),
                cad_value=-cad_value,
            )
        )
        self._post(
            LedgerEntry(
                date=on,
                account=self.name,
                kind="FX_CONVERSION",
                currency=target,
                amount=received,
                cad_value=cad_value,
            )
        )
        return received

    # -- Trades --------------------------------------------------------------

    def buy(
        self,
        ticker: str,
        quantity: float,
        price: float,
        currency: Currency,
        cad_per_usd: float,
        on: date,
    ) -> None:
        """Buy *quantity* of *ticker* at *price* per share in *currency*.

        The CAD acquisition book cost uses *cad_per_usd* for a USD purchase.
        """
        if quantity <= 0:
            raise ValueError(f"Buy quantity must be positive; got {quantity}.")
        if price <= 0:
            raise ValueError(f"Buy price must be positive; got {price}.")

        cost = _money(quantity * price)
        if cost > self.cash[currency]:
            raise ValueError(
                f"{self.name}: {currency} cash {self.cash[currency]:.2f} cannot "
                f"fund a {cost:.2f} purchase of {ticker}."
            )
        cost_cad = _money(cost * cad_per_usd) if currency == "USD" else cost

        self.cash[currency] = _money(self.cash[currency] - cost)
        self.quantities[ticker] = self.quantities.get(ticker, 0.0) + quantity
        self.ticker_currency[ticker] = currency
        self.book_cost_cad[ticker] = _money(
            self.book_cost_cad.get(ticker, 0.0) + cost_cad
        )
        self._post(
            LedgerEntry(
                date=on,
                account=self.name,
                kind="BUY",
                currency=currency,
                amount=-cost,
                cad_value=-cost_cad,
                ticker=ticker,
                quantity=quantity,
            )
        )

    def sell(
        self,
        ticker: str,
        quantity: float,
        price: float,
        currency: Currency,
        cad_per_usd: float,
        on: date,
    ) -> float:
        """Sell *quantity* of *ticker* at *price* per share in *currency*.

        Book cost is released pro rata, so book cost per share is unchanged by
        the sale.

        Returns
        -------
        float
            Account-level realized gain in CAD, measured against this
            account's book cost.  The taxable gain comes from the pooled tax
            ACB instead and can differ.
        """
        held = self.quantities.get(ticker, 0.0)
        if quantity <= 0:
            raise ValueError(f"Sell quantity must be positive; got {quantity}.")
        if price < 0:
            raise ValueError(f"Sell price must be non-negative; got {price}.")
        if quantity > held:
            raise ValueError(
                f"{self.name}: cannot sell {quantity} {ticker}; only {held} held."
            )

        proceeds = _money(quantity * price)
        proceeds_cad = _money(proceeds * cad_per_usd) if currency == "USD" else proceeds
        held_cost = self.book_cost_cad.get(ticker, 0.0)
        released_cost = _money(held_cost * quantity / held)
        gain_cad = _money(proceeds_cad - released_cost)

        self.cash[currency] = _money(self.cash[currency] + proceeds)
        remaining = held - quantity
        if remaining <= 1e-12:
            self.quantities.pop(ticker)
            self.ticker_currency.pop(ticker, None)
            self.book_cost_cad.pop(ticker, None)
        else:
            self.quantities[ticker] = remaining
            self.book_cost_cad[ticker] = _money(held_cost - released_cost)
        self.realized_gain_cad = _money(self.realized_gain_cad + gain_cad)

        self._post(
            LedgerEntry(
                date=on,
                account=self.name,
                kind="SELL",
                currency=currency,
                amount=proceeds,
                cad_value=proceeds_cad,
                ticker=ticker,
                quantity=quantity,
            )
        )
        return gain_cad

    # -- Valuation -----------------------------------------------------------

    def security_value_cad(self, market: MarketSnapshot) -> float:
        """CAD market value of this account's securities."""
        return _money(
            sum(
                market.security_value_cad(
                    ticker, quantity, self.ticker_currency[ticker]
                )
                for ticker, quantity in self.quantities.items()
            )
        )

    def cash_value_cad(self, market: MarketSnapshot) -> float:
        """CAD value of this account's settled cash."""
        return _money(self.cash["CAD"] + market.to_cad(self.cash["USD"], "USD"))

    def book_cost_total_cad(self) -> float:
        """CAD acquisition book cost of this account's holdings."""
        return _money(sum(self.book_cost_cad.values()))

    def total_value_cad(self, market: MarketSnapshot) -> float:
        """CAD value of cash plus securities (contributions are not re-added)."""
        return _money(self.cash_value_cad(market) + self.security_value_cad(market))

    def journal_cash_cad(self) -> float:
        """Sum of the CAD equivalent of every posted movement."""
        return _money(sum(entry.cad_value for entry in self.journal))

    def _post(self, entry: LedgerEntry) -> None:
        self.journal.append(entry)


@dataclass
class ReferenceScenario:
    """Materialized final state of the reference scenario.

    Attributes
    ----------
    market : MarketSnapshot
        Prices and FX used for valuation.
    accounts : dict[str, InvestmentAccount]
        Accounts keyed by name.
    tax_acb : ACBTracker
        Taxpayer-level pooled ACB for the non-registered accounts.
    pooled_before_sale : ACBPosition
        Pooled CAD position immediately before the taxable sale.
    tax_realized_gain_cad : float
        Taxable capital gain reported by the pooled ACB for the sale.
    """

    market: MarketSnapshot
    accounts: dict[str, InvestmentAccount]
    tax_acb: ACBTracker
    pooled_before_sale: ACBPosition
    tax_realized_gain_cad: float

    # -- Totals --------------------------------------------------------------

    def cash_total_cad(self) -> float:
        """CAD value of all settled cash across accounts."""
        return _money(
            sum(account.cash_value_cad(self.market) for account in self._ordered())
        )

    def security_total_cad(self) -> float:
        """CAD market value of all securities across accounts."""
        return _money(
            sum(account.security_value_cad(self.market) for account in self._ordered())
        )

    def total_value_cad(self) -> float:
        """Total portfolio value in CAD."""
        return _money(
            sum(account.total_value_cad(self.market) for account in self._ordered())
        )

    def tax_acb_remaining_cad(self) -> float:
        """Remaining pooled tax ACB after the sale."""
        return _money(self.tax_acb.get_total_acb(CAD_SECURITY))

    # -- Reconciliation ------------------------------------------------------

    def account_table(self) -> pd.DataFrame:
        """One row per account with cash, holdings, book cost, and totals."""
        rows = []
        for account in self._ordered():
            rows.append(
                {
                    "account": account.name,
                    "type": account.account_type,
                    "cash_cad": account.cash["CAD"],
                    "cash_usd": account.cash["USD"],
                    "securities_cad": account.security_value_cad(self.market),
                    "book_cost_cad": account.book_cost_total_cad(),
                    "total_value_cad": account.total_value_cad(self.market),
                    "contributions_cad": account.contributions_cad,
                    "realized_gain_cad": account.realized_gain_cad,
                }
            )
        return pd.DataFrame(rows)

    def currency_table(self) -> pd.DataFrame:
        """Cash and securities per currency, with CAD equivalents."""
        rows = []
        for currency in ("CAD", "USD"):
            cash_native = _money(
                sum(account.cash[currency] for account in self._ordered())
            )
            securities_native = _money(
                sum(
                    quantity * self.market.price(ticker, currency)
                    for account in self._ordered()
                    for ticker, quantity in account.quantities.items()
                    if account.ticker_currency[ticker] == currency
                )
            )
            cash_cad = self.market.to_cad(cash_native, currency)
            securities_cad = self.market.to_cad(securities_native, currency)
            rows.append(
                {
                    "currency": currency,
                    "cash_native": cash_native,
                    "cash_cad": cash_cad,
                    "securities_native": securities_native,
                    "securities_cad": securities_cad,
                    "total_cad": _money(cash_cad + securities_cad),
                }
            )
        return pd.DataFrame(rows)

    def conservation_table(self) -> pd.DataFrame:
        """Per-account cash conservation residuals.

        ``residual_cad = contributions + realized gain - cash - book cost``

        Zero for every account means no cash was created, lost, or
        double-counted.  With a constant FX rate this holds exactly; with a
        moving rate the difference would be an unrealized FX gain on the
        foreign-currency cash.
        """
        rows = []
        for account in self._ordered():
            cash_cad = account.cash_value_cad(self.market)
            book_cost_cad = account.book_cost_total_cad()
            rows.append(
                {
                    "account": account.name,
                    "contributions_cad": account.contributions_cad,
                    "realized_gain_cad": account.realized_gain_cad,
                    "cash_cad": cash_cad,
                    "book_cost_cad": book_cost_cad,
                    "residual_cad": _money(
                        account.contributions_cad
                        + account.realized_gain_cad
                        - cash_cad
                        - book_cost_cad
                    ),
                }
            )
        return pd.DataFrame(rows)

    # -- Reporting -----------------------------------------------------------

    def report(self) -> str:
        """Return a compact reconciliation report by account and currency."""
        accounts = self.account_table()
        currencies = self.currency_table()
        pooled = self.pooled_before_sale

        header = (
            f"{'Account':<12}{'Type':<10}"
            f"{'CAD cash':>{_REPORT_WIDTH}}{'USD cash':>{_REPORT_WIDTH}}"
            f"{'Securities':>{_REPORT_WIDTH}}{'Book cost':>{_REPORT_WIDTH}}"
            f"{'Total value':>{_REPORT_WIDTH}}{'Contrib.':>{_REPORT_WIDTH}}"
            f"{'Realized':>{_REPORT_WIDTH}}"
        )
        lines = [
            "PySharpe — Canadian portfolio reference example (synthetic data)",
            "=" * 78,
            f"prices: {CAD_SECURITY} CAD {self.market.price(CAD_SECURITY, 'CAD'):,.2f}"
            f" | {USD_SECURITY} USD {self.market.price(USD_SECURITY, 'USD'):,.2f}",
            f"valuation FX: {self.market.cad_per_usd:.2f} CAD per USD",
            "",
            "ACCOUNT RECONCILIATION (CAD unless the column says otherwise)",
            header,
        ]
        for row in accounts.itertuples(index=False):
            lines.append(
                f"{row.account:<12}{row.type:<10}"
                f"{row.cash_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.cash_usd:>{_REPORT_WIDTH},.2f}"
                f"{row.securities_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.book_cost_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.total_value_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.contributions_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.realized_gain_cad:>{_REPORT_WIDTH},.2f}"
            )
        lines.append(
            f"{'TOTAL':<22}"
            f"{accounts['cash_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{accounts['cash_usd'].sum():>{_REPORT_WIDTH},.2f}"
            f"{accounts['securities_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{accounts['book_cost_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{accounts['total_value_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{accounts['contributions_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{accounts['realized_gain_cad'].sum():>{_REPORT_WIDTH},.2f}"
        )

        lines.extend(
            [
                "",
                "CURRENCY BREAKDOWN (CAD equivalent at the valuation FX rate)",
                f"{'Ccy':<6}{'Cash':>{_REPORT_WIDTH}}{'Cash (CAD)':>{_REPORT_WIDTH}}"
                f"{'Securities':>{_REPORT_WIDTH}}{'Sec. (CAD)':>{_REPORT_WIDTH}}"
                f"{'Total (CAD)':>{_REPORT_WIDTH}}",
            ]
        )
        for row in currencies.itertuples(index=False):
            lines.append(
                f"{row.currency:<6}"
                f"{row.cash_native:>{_REPORT_WIDTH},.2f}"
                f"{row.cash_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.securities_native:>{_REPORT_WIDTH},.2f}"
                f"{row.securities_cad:>{_REPORT_WIDTH},.2f}"
                f"{row.total_cad:>{_REPORT_WIDTH},.2f}"
            )
        lines.append(
            f"{'TOTAL':<6}{'':>{_REPORT_WIDTH}}"
            f"{currencies['cash_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{'':>{_REPORT_WIDTH}}"
            f"{currencies['securities_cad'].sum():>{_REPORT_WIDTH},.2f}"
            f"{self.total_value_cad():>{_REPORT_WIDTH},.2f}"
        )

        lines.extend(
            [
                "",
                f"TAXPAYER TAX ACB — pooled {CAD_SECURITY}, non-registered only",
                f"  {'original pooled cost':<24}{pooled.total_cost:>{_REPORT_WIDTH},.2f}",
                f"  {'pooled quantity':<24}{pooled.total_shares:>{_REPORT_WIDTH},.0f}",
                f"  {'ACB per share':<24}{pooled.acb_per_share:>{_REPORT_WIDTH},.2f}",
                f"  {'realized taxable gain':<24}"
                f"{self.tax_realized_gain_cad:>{_REPORT_WIDTH},.2f}",
                f"  {'remaining tax ACB':<24}"
                f"{self.tax_acb_remaining_cad():>{_REPORT_WIDTH},.2f}",
                "",
                "  TAXABLE-B book cost is "
                f"{self.accounts['TAXABLE-B'].book_cost_total_cad():,.2f} on "
                f"{self.accounts['TAXABLE-B'].quantities.get(CAD_SECURITY, 0.0):,.0f}"
                f" shares, while its pooled tax ACB is "
                f"{self.tax_acb_remaining_cad():,.2f}.",
                "  Account book cost records what an account paid; tax ACB pools",
                "  identical property across all non-registered accounts.",
                "",
                "CASH CONSERVATION (contributions + realized − cash − book cost)",
            ]
        )
        residuals = self.conservation_table()
        lines.append(
            "  "
            + "   ".join(
                f"{row.account} {row.residual_cad:,.2f}"
                for row in residuals.itertuples(index=False)
            )
        )
        lines.extend(["", f"TOTAL PORTFOLIO VALUE {self.total_value_cad():>,.2f} CAD"])
        return "\n".join(lines)

    # -- Replay --------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Plain, comparable copy of every scenario figure.

        Used by the integration tests to prove deterministic replay.
        """
        return {
            "accounts": {
                name: {
                    "account_type": account.account_type,
                    "cash": dict(account.cash),
                    "quantities": dict(account.quantities),
                    "ticker_currency": dict(account.ticker_currency),
                    "book_cost_cad": dict(account.book_cost_cad),
                    "contributions_cad": account.contributions_cad,
                    "realized_gain_cad": account.realized_gain_cad,
                    "journal": [
                        (
                            entry.date,
                            entry.account,
                            entry.kind,
                            entry.currency,
                            entry.amount,
                            entry.cad_value,
                            entry.ticker,
                            entry.quantity,
                        )
                        for entry in account.journal
                    ],
                }
                for name, account in self.accounts.items()
            },
            "tax_acb": {
                "pooled_before_sale": (
                    self.pooled_before_sale.total_shares,
                    self.pooled_before_sale.total_cost,
                ),
                "shares": self.tax_acb.get_shares(CAD_SECURITY),
                "total_cost": self.tax_acb.get_total_acb(CAD_SECURITY),
                "realized_gain_cad": self.tax_realized_gain_cad,
                "trades": [
                    (
                        trade.ticker,
                        trade.date,
                        trade.action,
                        trade.shares,
                        trade.price_per_share,
                        trade.total_amount,
                        trade.realized_gain_loss,
                    )
                    for trade in self.tax_acb.trades
                ],
            },
            "total_value_cad": self.total_value_cad(),
            "cash_total_cad": self.cash_total_cad(),
            "security_total_cad": self.security_total_cad(),
        }

    def _ordered(self) -> list[InvestmentAccount]:
        return [self.accounts[name] for name in ACCOUNT_ORDER]


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def _record_taxable_buy(
    account: InvestmentAccount,
    tax_acb: ACBTracker,
    ticker: str,
    quantity: float,
    price: float,
    on: date,
) -> None:
    """Buy in the account ledger and add the acquisition to the tax ACB pool."""
    account.buy(ticker, quantity, price, "CAD", CAD_PER_USD, on)
    tax_acb.record_buy(ticker, quantity, price, on, account_type="NON_REG")


def _record_taxable_sell(
    account: InvestmentAccount,
    tax_acb: ACBTracker,
    ticker: str,
    quantity: float,
    price: float,
    on: date,
) -> float:
    """Sell in the account ledger and realize the gain against the tax ACB pool."""
    account.sell(ticker, quantity, price, "CAD", CAD_PER_USD, on)
    return tax_acb.record_sell(ticker, quantity, price, on, account_type="NON_REG")


def run_reference_scenario() -> ReferenceScenario:
    """Execute the reference scenario end to end.

    Deterministic: fixed prices, fixed FX, fixed settlement dates, no I/O.

    Returns
    -------
    ReferenceScenario
        Final account, currency, and pooled tax-ACB state.
    """
    market = MarketSnapshot(
        cad_prices={CAD_SECURITY: 150.0},
        usd_prices={USD_SECURITY: 50.0},
        cad_per_usd=CAD_PER_USD,
    )

    tfsa = InvestmentAccount("TFSA-A", account_type="TFSA")
    taxable_a = InvestmentAccount("TAXABLE-A", account_type="TAXABLE")
    taxable_b = InvestmentAccount("TAXABLE-B", account_type="TAXABLE")
    accounts = {account.name: account for account in (tfsa, taxable_a, taxable_b)}

    # 1. TFSA-A: contribute, convert CAD 6,750 into USD 5,000, buy 10 units.
    tfsa.contribute(10_000.0, PURCHASE_DATE)
    tfsa.convert(6_750.0, "CAD", "USD", CAD_PER_USD, PURCHASE_DATE)
    tfsa.buy(USD_SECURITY, 10.0, 50.0, "USD", CAD_PER_USD, PURCHASE_DATE)

    # 2. Both taxable purchases settle before the sale.  CRA pools identical
    #    property across accounts, so both acquisitions enter one tax pool
    #    even though each account keeps its own book cost.
    tax_acb = ACBTracker()
    taxable_a.contribute(10_000.0, PURCHASE_DATE)
    _record_taxable_buy(taxable_a, tax_acb, CAD_SECURITY, 100.0, 100.0, PURCHASE_DATE)
    taxable_b.contribute(20_000.0, PURCHASE_DATE)
    _record_taxable_buy(taxable_b, tax_acb, CAD_SECURITY, 100.0, 200.0, PURCHASE_DATE)

    pooled_before_sale = ACBPosition(
        ticker=CAD_SECURITY,
        total_shares=tax_acb.get_shares(CAD_SECURITY),
        total_cost=tax_acb.get_total_acb(CAD_SECURITY),
    )

    # 3. TAXABLE-A sells its 100 shares at CAD 150 against the pooled ACB.
    tax_realized_gain = _record_taxable_sell(
        taxable_a, tax_acb, CAD_SECURITY, 100.0, 150.0, SALE_DATE
    )

    # A TFSA book cost (CAD 675) is informational only: TFSA gains are never
    # taxable, so the account is excluded from the tax ACB pool.

    return ReferenceScenario(
        market=market,
        accounts=accounts,
        tax_acb=tax_acb,
        pooled_before_sale=pooled_before_sale,
        tax_realized_gain_cad=tax_realized_gain,
    )


def main() -> None:
    """Run the reference scenario and print the reconciliation report."""
    print(run_reference_scenario().report())


if __name__ == "__main__":
    main()
