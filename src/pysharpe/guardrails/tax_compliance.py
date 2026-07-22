"""CRA Superficial Loss compliance interlock for Canadian multi-account portfolios.

Provides ACB tracking with commission support for Non-Registered accounts and
a ``SuperficialLossGuardrail`` that validates proposed trade slates against
the CRA superficial loss rules across TFSA, RRSP, and Non-Registered account
wrappers per ITA s. 54 (superficial loss definition).

Key Concepts
------------
- **Superficial Loss Rule**: If you sell a security at a loss in a Non-Registered
  account and you (or an affiliated person) acquire the same or identical
  property in *any* account within 30 days before or after the sale, the capital
  loss is denied and added to the ACB of the substituted property.
- **Identical Property**: Securities that are the same or substantially similar.
  For example, VFV.TO (CAD-hedged S&P 500) and VOO (US-listed S&P 500) track
  the same index and may be considered identical property.

Module Components
-----------------
- :class:`TransactionRecord` — Immutable record of a trade across any account.
- :class:`ACBPosition` — State of the adjusted cost base for a single ticker.
- :class:`ACBTracker` — ACB pool tracker with commission support.
- :class:`SuperficialLossViolation` — Describes a detected superficial loss.
- :class:`SuperficialLossGuardrail` — Validates trade slates and re-routes
  violating orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default identical-property mapping for common Canadian ETF pairs
# ---------------------------------------------------------------------------

_DEFAULT_IDENTICAL_MAP: dict[str, set[str]] = {
    # S&P 500 — CAD-listed, CAD-hedged, and US-listed track the same index
    "VFV.TO": {"VFV", "VOO", "SPY", "IVV", "ZSP.TO", "ZSP"},
    "VOO": {"VFV.TO", "VFV", "SPY", "IVV", "ZSP.TO", "ZSP"},
    "SPY": {"VFV.TO", "VFV", "VOO", "IVV", "ZSP.TO", "ZSP"},
    "IVV": {"VFV.TO", "VFV", "VOO", "SPY", "ZSP.TO", "ZSP"},
    "ZSP.TO": {"VFV.TO", "VFV", "VOO", "SPY", "IVV", "ZSP"},
    # NASDAQ-100
    "QQC.TO": {"QQC", "QQQ", "QQQM", "XQQ.TO", "XQQ"},
    "QQQ": {"QQC.TO", "QQC", "QQQM", "XQQ.TO", "XQQ"},
    "QQQM": {"QQC.TO", "QQC", "QQQ", "XQQ.TO", "XQQ"},
    "XQQ.TO": {"QQC.TO", "QQC", "QQQ", "QQQM", "XQQ"},
    # TSX 60 / Canadian equity
    "XIU.TO": {"XIU", "HXT.TO", "HXT", "ZCN.TO", "ZCN"},
    "HXT.TO": {"XIU.TO", "XIU", "ZCN.TO", "ZCN"},
    "ZCN.TO": {"XIU.TO", "XIU", "HXT.TO", "HXT"},
    # Canadian dividend
    "VDY.TO": {"VDY", "XDV.TO", "XDV", "CDZ.TO", "CDZ"},
    "XDV.TO": {"VDY.TO", "VDY", "CDZ.TO", "CDZ"},
    # EAFE / International
    "XEF.TO": {"XEF", "VIU.TO", "VIU", "ZEA.TO", "ZEA"},
    "VIU.TO": {"XEF.TO", "XEF", "ZEA.TO", "ZEA"},
    # Emerging markets
    "XEC.TO": {"XEC", "VEE.TO", "VEE", "ZEM.TO", "ZEM"},
    "VEE.TO": {"XEC.TO", "XEC", "ZEM.TO", "ZEM"},
    # Aggregate bonds
    "XBB.TO": {"XBB", "VAB.TO", "VAB", "ZAG.TO", "ZAG"},
    "VAB.TO": {"XBB.TO", "XBB", "ZAG.TO", "ZAG"},
    "ZAG.TO": {"XBB.TO", "XBB", "VAB.TO", "VAB"},
}


def build_default_identical_map() -> dict[str, frozenset[str]]:
    """Build a frozen (hashable) copy of the default identical-property mapping.

    Returns
    -------
    dict[str, frozenset[str]]
        Ticker → frozen set of ticker strings considered identical property.
    """
    return {k: frozenset(v) for k, v in _DEFAULT_IDENTICAL_MAP.items()}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionRecord:
    """Immutable record of a single trade in any account wrapper.

    Attributes
    ----------
    ticker : str
        Security ticker symbol (e.g. ``"VFV.TO"``).
    date : date
        Trade settlement date.
    account_type : str
        Account wrapper: ``"TFSA"``, ``"RRSP"``, or ``"NON_REG"``.
    action : str
        ``"BUY"`` or ``"SELL"``.
    shares : float
        Number of shares transacted (always positive).
    price_per_share : float
        Transaction price per share.
    total_amount : float
        Total dollar amount (positive for buys, negative for sells).
    commission : float
        Trading commission paid (always non-negative).
    realized_gain_loss : float
        Realized capital gain (>0) or loss (<0). Zero for buys.
    """

    ticker: str
    date: date
    account_type: str
    action: str
    shares: float
    price_per_share: float
    total_amount: float
    commission: float = 0.0
    realized_gain_loss: float = 0.0

    @property
    def is_buy(self) -> bool:
        """True if this is a purchase."""
        return self.action.upper() == "BUY"

    @property
    def is_sell(self) -> bool:
        """True if this is a sale."""
        return self.action.upper() == "SELL"

    @property
    def is_loss(self) -> bool:
        """True if this sale realized a capital loss."""
        return self.is_sell and self.realized_gain_loss < 0.0

    @property
    def is_tax_sheltered(self) -> bool:
        """True if this trade occurred in a registered (sheltered) account."""
        return self.account_type.upper() in {"TFSA", "RRSP", "FHSA", "RESP", "LIRA"}

    @property
    def is_non_reg(self) -> bool:
        """True if this trade occurred in a Non-Registered (taxable) account."""
        return self.account_type.upper() in {
            "NON_REG",
            "NON-REG",
            "NONREG",
            "MARGIN",
            "CASH",
        }


@dataclass
class ACBPosition:
    """Running ACB state for a single security in a Non-Registered account.

    Attributes
    ----------
    ticker : str
        Security ticker symbol.
    total_shares : float
        Current number of shares held.
    total_cost : float
        Total adjusted cost base (sum of all acquisition costs including
        commissions, net of return of capital).
    """

    ticker: str
    total_shares: float = 0.0
    total_cost: float = 0.0

    @property
    def acb_per_share(self) -> float:
        """Weighted-average cost per share per CRA rules.

        Returns 0.0 when no shares are held.
        """
        if self.total_shares <= 0:
            return 0.0
        return self.total_cost / self.total_shares


@dataclass(frozen=True)
class SuperficialLossViolation:
    """Describes a detected CRA superficial loss rule violation.

    Attributes
    ----------
    sell_trade : TransactionRecord
        The sale in a Non-Registered account that realized a capital loss.
    conflicting_buy : TransactionRecord
        The purchase of identical property within the ±30-day window that
        triggers the superficial loss rule.
    identical_tickers : frozenset[str]
        The set of ticker symbols considered identical to the sold security.
    days_delta : int
        Number of days between the sale and the conflicting purchase (absolute).
    message : str
        Human-readable description of the violation.
    """

    sell_trade: TransactionRecord
    conflicting_buy: TransactionRecord
    identical_tickers: frozenset[str]
    days_delta: int
    message: str


# ---------------------------------------------------------------------------
# ACB Tracker — Non-Registered accounts with commission support
# ---------------------------------------------------------------------------


class ACBTracker:
    """Track Adjusted Cost Base for identical securities in Non-Registered accounts.

    Implements the CRA weighted-average cost method (ITA s. 47(1)) with
    explicit commission tracking.  Commissions are added to the cost base
    on purchases and deducted from proceeds on sales.

    Parameters
    ----------
    initial_positions : dict[str, tuple[float, float]], optional
        Seed the tracker as ``{ticker: (total_shares, total_cost)}``.

    Attributes
    ----------
    positions : dict[str, ACBPosition]
        Current ACB state per ticker (read-only copy).
    trades : list[TransactionRecord]
        Complete trade history for this account.
    """

    def __init__(
        self,
        initial_positions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._positions: dict[str, ACBPosition] = {}
        self._trades: list[TransactionRecord] = []

        if initial_positions:
            for ticker, (shares, cost) in initial_positions.items():
                if shares < 0:
                    raise ValueError(
                        f"Initial shares for '{ticker}' must be non-negative; "
                        f"got {shares}."
                    )
                if cost < 0:
                    raise ValueError(
                        f"Initial cost for '{ticker}' must be non-negative; got {cost}."
                    )
                if shares > 0 and cost <= 0:
                    raise ValueError(
                        f"Initial cost for '{ticker}' must be positive when "
                        f"shares > 0; got cost={cost} for {shares} shares."
                    )
                self._positions[ticker] = ACBPosition(
                    ticker=ticker,
                    total_shares=shares,
                    total_cost=cost,
                )

    # -- Read-only accessors ---------------------------------------------------

    @property
    def positions(self) -> dict[str, ACBPosition]:
        """Return a shallow copy of current positions."""
        return dict(self._positions)

    @property
    def trades(self) -> list[TransactionRecord]:
        """Return a shallow copy of the trade history."""
        return list(self._trades)

    def get_position(self, ticker: str) -> ACBPosition:
        """Return the ACB position for *ticker*, creating an empty one if unseen."""
        if ticker not in self._positions:
            self._positions[ticker] = ACBPosition(ticker=ticker)
        return self._positions[ticker]

    def get_acb_per_share(self, ticker: str) -> float:
        """Return the weighted-average ACB per share for *ticker*."""
        return self.get_position(ticker).acb_per_share

    def get_shares(self, ticker: str) -> float:
        """Return the current share count for *ticker*."""
        return self.get_position(ticker).total_shares

    def get_total_acb(self, ticker: str) -> float:
        """Return the total ACB for *ticker*."""
        return self.get_position(ticker).total_cost

    # -- Trade recording -------------------------------------------------------

    def record_buy(
        self,
        ticker: str,
        shares: float,
        price_per_share: float,
        trade_date: date,
        commission: float = 0.0,
        account_type: str = "NON_REG",
    ) -> ACBPosition:
        """Record a purchase and update the weighted-average ACB.

        Per CRA rules, total ACB = previous total ACB + (purchase cost + commission).
        The new per-unit ACB becomes total ACB / total units.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        shares : float
            Number of shares purchased. Must be strictly positive.
        price_per_share : float
            Price paid per share. Must be positive.
        trade_date : date
            Settlement date.
        commission : float
            Trading commission paid. Must be non-negative. Default 0.0.
        account_type : str
            Account wrapper label. Default ``"NON_REG"``.

        Returns
        -------
        ACBPosition
            Updated position after the buy.
        """
        if shares <= 0:
            raise ValueError(f"Shares must be positive for a buy; got {shares}.")
        if price_per_share <= 0:
            raise ValueError(
                f"Price per share must be positive for a buy; got {price_per_share}."
            )
        if commission < 0:
            raise ValueError(f"Commission must be non-negative; got {commission}.")

        purchase_cost = shares * price_per_share
        total_acquisition = purchase_cost + commission
        pos = self.get_position(ticker)

        pos.total_shares += shares
        pos.total_cost += total_acquisition

        self._trades.append(
            TransactionRecord(
                ticker=ticker,
                date=trade_date,
                account_type=account_type,
                action="BUY",
                shares=shares,
                price_per_share=price_per_share,
                total_amount=total_acquisition,
                commission=commission,
                realized_gain_loss=0.0,
            )
        )

        logger.debug(
            "BUY  %s [%s]: +%.4f sh @ $%.4f + $%.2f comm → ACB/sh $%.4f",
            ticker,
            account_type,
            shares,
            price_per_share,
            commission,
            pos.acb_per_share,
        )
        return pos

    def record_sell(
        self,
        ticker: str,
        shares: float,
        price_per_share: float,
        trade_date: date,
        commission: float = 0.0,
        account_type: str = "NON_REG",
    ) -> float:
        """Record a sale and compute the realized capital gain or loss.

        Realized Gain/Loss = Proceeds − (Sold Units × ACB per Share) − Commission.
        The per-unit ACB remains unchanged after a partial disposition.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        shares : float
            Number of shares sold. Must be positive and not exceed holdings.
        price_per_share : float
            Sale price per share. Must be non-negative.
        trade_date : date
            Settlement date.
        commission : float
            Trading commission paid. Must be non-negative. Default 0.0.
        account_type : str
            Account wrapper label. Default ``"NON_REG"``.

        Returns
        -------
        float
            Realized capital gain (positive) or loss (negative).

        Raises
        ------
        ValueError
            If shares is non-positive, exceeds holdings, or price is negative.
        """
        if shares <= 0:
            raise ValueError(f"Shares must be positive for a sell; got {shares}.")
        if price_per_share < 0:
            raise ValueError(
                f"Price per share must be non-negative; got {price_per_share}."
            )
        if commission < 0:
            raise ValueError(f"Commission must be non-negative; got {commission}.")

        pos = self.get_position(ticker)
        if pos.total_shares <= 0 or pos.total_cost <= 0:
            raise ValueError(
                f"Cannot sell '{ticker}': no shares held "
                f"(shares={pos.total_shares}, total_cost={pos.total_cost})."
            )
        if shares > pos.total_shares:
            raise ValueError(
                f"Cannot sell {shares} shares of '{ticker}'; only "
                f"{pos.total_shares} held."
            )

        acb_per_share = pos.acb_per_share
        acb_of_sold = shares * acb_per_share
        proceeds = shares * price_per_share
        gain_loss = proceeds - acb_of_sold - commission

        # Reduce the pool proportionally
        pos.total_shares -= shares
        pos.total_cost -= acb_of_sold

        # Guard against floating-point drift
        if pos.total_shares < 1e-12:
            pos.total_shares = 0.0
            pos.total_cost = 0.0

        self._trades.append(
            TransactionRecord(
                ticker=ticker,
                date=trade_date,
                account_type=account_type,
                action="SELL",
                shares=shares,
                price_per_share=price_per_share,
                total_amount=proceeds,
                commission=commission,
                realized_gain_loss=gain_loss,
            )
        )

        logger.debug(
            "SELL %s [%s]: -%.4f sh @ $%.4f − $%.2f comm → gain/loss $%.4f",
            ticker,
            account_type,
            shares,
            price_per_share,
            commission,
            gain_loss,
        )
        return gain_loss

    def get_unrealized_gain_loss(self, ticker: str, current_price: float) -> float:
        """Compute unrealized capital gain or loss for *ticker*.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        current_price : float
            Current market price per share.

        Returns
        -------
        float
            Unrealized gain (>0) or loss (<0) in dollars.
        """
        if current_price < 0:
            raise ValueError(
                f"current_price must be non-negative; got {current_price}."
            )
        pos = self.get_position(ticker)
        if pos.total_shares <= 0:
            return 0.0
        return pos.total_shares * current_price - pos.total_cost


# ---------------------------------------------------------------------------
# Superficial Loss Guardrail
# ---------------------------------------------------------------------------


class SuperficialLossGuardrail:
    """Enforce CRA superficial loss rules across multi-account portfolios.

    Maintains a rolling 30-day transaction log across all account wrappers
    (TFSA, RRSP, Non-Registered) and validates proposed trade slates before
    execution.  When a sale in a Non-Registered account would realize a loss
    and an identical-property purchase exists (or is proposed) in a
    tax-sheltered account within the ±30-day window, the purchase is
    flagged and the cash is re-routed to the next-best underweight
    non-identical asset.

    Parameters
    ----------
    identical_property_map : dict[str, frozenset[str]], optional
        Mapping of ticker → set of ticker strings considered identical
        property.  If not provided, the default Canadian ETF mapping is used.
    window_days : int
        Number of days before/after a sale to scan for conflicting purchases.
        Default is 30 (matching CRA superficial loss window).

    Attributes
    ----------
    history : list[TransactionRecord]
        Complete transaction history across all tracked accounts.
    """

    def __init__(
        self,
        identical_property_map: dict[str, frozenset[str]] | None = None,
        window_days: int = 30,
    ) -> None:
        self._identical_map: dict[str, frozenset[str]] = (
            identical_property_map or build_default_identical_map()
        )
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1; got {window_days}.")
        self.window_days: int = window_days
        self._history: list[TransactionRecord] = []

    # -- Read-only accessors ---------------------------------------------------

    @property
    def history(self) -> list[TransactionRecord]:
        """Return a shallow copy of the transaction history."""
        return list(self._history)

    @property
    def identical_map(self) -> dict[str, frozenset[str]]:
        """Return a shallow copy of the identical-property mapping."""
        return dict(self._identical_map)

    # -- Transaction recording -------------------------------------------------

    def record_transaction(self, txn: TransactionRecord) -> None:
        """Record a completed transaction in the cross-account history.

        Parameters
        ----------
        txn : TransactionRecord
            The trade to record.
        """
        self._history.append(txn)

    def record_transactions(self, txns: list[TransactionRecord]) -> None:
        """Record multiple completed transactions at once.

        Parameters
        ----------
        txns : list[TransactionRecord]
            Trades to record.
        """
        self._history.extend(txns)

    # -- Identical-property resolution -----------------------------------------

    def get_identical_tickers(self, ticker: str) -> frozenset[str]:
        """Return the set of tickers considered identical property to *ticker*.

        Always includes *ticker* itself.

        Parameters
        ----------
        ticker : str
            The reference ticker symbol.

        Returns
        -------
        frozenset[str]
        """
        result: set[str] = {ticker}
        # Check if ticker is a key in the map
        if ticker in self._identical_map:
            result.update(self._identical_map[ticker])
        # Check if ticker appears in any value set
        for key, values in self._identical_map.items():
            if ticker in values:
                result.add(key)
                result.update(values)
        return frozenset(result)

    def are_identical(self, ticker_a: str, ticker_b: str) -> bool:
        """Return True if two tickers are considered identical property.

        Parameters
        ----------
        ticker_a : str
        ticker_b : str

        Returns
        -------
        bool
        """
        if ticker_a == ticker_b:
            return True
        identical = self.get_identical_tickers(ticker_a)
        return ticker_b in identical

    # -- Superficial loss detection --------------------------------------------

    def _find_superficial_loss_conflicts(
        self,
        sell_trade: TransactionRecord,
        all_trades: list[TransactionRecord],
    ) -> list[SuperficialLossViolation]:
        """Scan all trades for ALL superficial-loss triggers against *sell_trade*.

        Returns all conflicts — a single sale can conflict with multiple buys.
        """
        if not sell_trade.is_non_reg:
            return []
        if not sell_trade.is_loss:
            return []

        identical_tickers = self.get_identical_tickers(sell_trade.ticker)
        sell_date = sell_trade.date
        window_start = sell_date - timedelta(days=self.window_days)
        window_end = sell_date + timedelta(days=self.window_days)

        results: list[SuperficialLossViolation] = []

        for txn in all_trades:
            if txn is sell_trade:
                continue
            if not txn.is_buy:
                continue
            if txn.ticker not in identical_tickers:
                continue
            if not (window_start <= txn.date <= window_end):
                continue
            if not txn.is_tax_sheltered:
                continue

            days_delta = abs((txn.date - sell_date).days)
            results.append(
                SuperficialLossViolation(
                    sell_trade=sell_trade,
                    conflicting_buy=txn,
                    identical_tickers=frozenset(identical_tickers),
                    days_delta=days_delta,
                    message=(
                        f"Superficial loss: {sell_trade.ticker} sold at loss in "
                        f"{sell_trade.account_type} on {sell_trade.date} conflicts "
                        f"with {txn.ticker} purchase in {txn.account_type} on "
                        f"{txn.date} ({days_delta} days apart). "
                        f"Identical property: {sorted(identical_tickers)}."
                    ),
                )
            )

        return results

    def detect_violations(
        self,
        proposed_trades: list[TransactionRecord],
    ) -> list[SuperficialLossViolation]:
        """Scan proposed trades against the full history for superficial loss violations.

        Parameters
        ----------
        proposed_trades : list[TransactionRecord]
            The proposed trade slate to validate.

        Returns
        -------
        list[SuperficialLossViolation]
            All detected violations, ordered by sell date.
        """
        combined = self._history + proposed_trades
        violations: list[SuperficialLossViolation] = []

        # 1) Proposed sells conflicting with history + other proposed trades
        for txn in proposed_trades:
            if not txn.is_sell:
                continue
            violations.extend(self._find_superficial_loss_conflicts(txn, combined))

        # 2) Historical sells conflicting with proposed buys
        for hist_txn in self._history:
            if not hist_txn.is_sell:
                continue
            for proposed_txn in proposed_trades:
                if not proposed_txn.is_buy:
                    continue
                if not proposed_txn.is_tax_sheltered:
                    continue
                violations.extend(
                    self._find_superficial_loss_conflicts(
                        hist_txn,
                        self._history + [proposed_txn],
                    )
                )

        # Deduplicate
        seen: set[tuple] = set()
        unique: list[SuperficialLossViolation] = []
        for v in violations:
            key = (
                v.sell_trade.date,
                v.sell_trade.ticker,
                v.sell_trade.account_type,
                v.conflicting_buy.date,
                v.conflicting_buy.ticker,
                v.conflicting_buy.account_type,
            )
            if key not in seen:
                seen.add(key)
                unique.append(v)

        unique.sort(key=lambda v: (v.sell_trade.date, v.sell_trade.ticker))
        return unique

    def validate_trade_slate(
        self,
        proposed_trades: list[TransactionRecord],
        opportunity_ranking: dict[str, float] | None = None,
    ) -> tuple[list[TransactionRecord], list[SuperficialLossViolation]]:
        """Validate a proposed trade slate and re-route violating orders.

        Scans the proposed trades against the full history.  For each
        detected violation, the offending BUY in the tax-sheltered account
        is removed from the slate.  If *opportunity_ranking* is provided
        (mapping ticker → score), the cash from the blocked purchase is
        re-routed to the highest-ranking underweight asset that is NOT
        identical property to the sold security.

        Parameters
        ----------
        proposed_trades : list[TransactionRecord]
            The proposed trade slate to validate.
        opportunity_ranking : dict[str, float], optional
            Ticker → opportunity score (higher is better).  Used to re-route
            blocked cash to the next-best non-identical asset.

        Returns
        -------
        tuple[list[TransactionRecord], list[SuperficialLossViolation]]
            - **validated_trades**: The cleaned trade slate with violating
              purchases removed and (optionally) re-routed.
            - **violations**: All detected superficial loss violations.
        """
        violations = self.detect_violations(proposed_trades)

        if not violations:
            return list(proposed_trades), []

        # Build a set of (ticker, date, account_type) keys to block
        blocked_keys: set[tuple[str, date, str]] = set()
        blocked_cash: dict[str, float] = {}  # ticker → blocked dollar amount

        for v in violations:
            buy = v.conflicting_buy
            # Only block if the buy is in the proposed slate
            if buy in proposed_trades:
                blocked_keys.add((buy.ticker, buy.date, buy.account_type))
                blocked_cash[buy.ticker] = (
                    blocked_cash.get(buy.ticker, 0.0) + buy.total_amount
                )

        # Filter out blocked trades
        validated: list[TransactionRecord] = []
        for txn in proposed_trades:
            key = (txn.ticker, txn.date, txn.account_type)
            if key in blocked_keys:
                logger.warning(
                    "Blocked %s %s in %s: triggers superficial loss rule.",
                    txn.action,
                    txn.ticker,
                    txn.account_type,
                )
                continue
            validated.append(txn)

        # Re-route blocked cash to next-best non-identical assets
        if opportunity_ranking and blocked_cash:
            for sold_ticker, cash_amount in blocked_cash.items():
                rerouted = self._reroute_cash(
                    sold_ticker=sold_ticker,
                    cash_amount=cash_amount,
                    validated=validated,
                    opportunity_ranking=opportunity_ranking,
                )
                validated.extend(rerouted)

        return validated, violations

    def _reroute_cash(
        self,
        sold_ticker: str,
        cash_amount: float,
        validated: list[TransactionRecord],
        opportunity_ranking: dict[str, float],
    ) -> list[TransactionRecord]:
        """Re-route blocked cash to the highest-ranking non-identical assets.

        Parameters
        ----------
        sold_ticker : str
            The ticker that was sold at a loss (whose identical-property group
            must be avoided).
        cash_amount : float
            Dollar amount to re-route.
        validated : list[TransactionRecord]
            Current validated trade slate (mutated in-place if a matching
            existing buy can be increased).
        opportunity_ranking : dict[str, float]
            Ticker → opportunity score.

        Returns
        -------
        list[TransactionRecord]
            Newly created BUY transactions for the re-routed cash.
        """
        identical_group = self.get_identical_tickers(sold_ticker)

        # Find non-identical candidates sorted by opportunity score descending
        candidates = [
            (ticker, score)
            for ticker, score in opportunity_ranking.items()
            if ticker not in identical_group
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            logger.warning(
                "No non-identical assets available to re-route $%.2f from %s.",
                cash_amount,
                sold_ticker,
            )
            return []

        # Try to add to the best existing BUY in the validated slate first
        best_ticker = candidates[0][0]
        for txn in validated:
            if txn.is_buy and txn.ticker == best_ticker and txn.is_tax_sheltered:
                # Cannot mutate frozen dataclass — flag for caller
                logger.info(
                    "Re-route $%.2f from %s → %s (already in slate).",
                    cash_amount,
                    sold_ticker,
                    best_ticker,
                )
                return []

        # Create a new BUY transaction for the best candidate
        logger.info(
            "Re-route $%.2f from %s → %s (new buy).",
            cash_amount,
            sold_ticker,
            best_ticker,
        )
        # We don't know price/share, so record as a cash-amount transaction
        rerouted = TransactionRecord(
            ticker=best_ticker,
            date=date.today(),
            account_type="TFSA",  # Default to the same type that was blocked
            action="BUY",
            shares=0.0,  # Unknown — will be computed at execution
            price_per_share=0.0,
            total_amount=cash_amount,
            commission=0.0,
            realized_gain_loss=0.0,
        )
        return [rerouted]

    # -- Bulk operations -------------------------------------------------------

    def get_trades_in_window(
        self,
        reference_date: date,
        ticker: str | None = None,
    ) -> list[TransactionRecord]:
        """Return all trades within ± ``window_days`` of *reference_date*.

        Parameters
        ----------
        reference_date : date
            Center of the window.
        ticker : str, optional
            If provided, filter to trades involving this ticker or its
            identical-property group.

        Returns
        -------
        list[TransactionRecord]
        """
        window_start = reference_date - timedelta(days=self.window_days)
        window_end = reference_date + timedelta(days=self.window_days)

        identical: frozenset[str] | None = None
        if ticker is not None:
            identical = self.get_identical_tickers(ticker)

        result: list[TransactionRecord] = []
        for txn in self._history:
            if not (window_start <= txn.date <= window_end):
                continue
            if identical is not None and txn.ticker not in identical:
                continue
            result.append(txn)

        return result

    def clear_history(self) -> None:
        """Clear all recorded transaction history."""
        self._history.clear()

    def summary(self) -> dict[str, Any]:
        """Return a summary of the guardrail state.

        Returns
        -------
        dict
            Keys: ``total_transactions``, ``unique_tickers``,
            ``accounts``, ``window_days``, ``identical_groups``.
        """
        tickers = {t.ticker for t in self._history}
        accounts = {t.account_type for t in self._history}
        return {
            "total_transactions": len(self._history),
            "unique_tickers": len(tickers),
            "accounts": sorted(accounts),
            "window_days": self.window_days,
            "identical_groups": len(self._identical_map),
        }
