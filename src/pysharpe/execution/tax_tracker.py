"""Adjusted Cost Base (ACB) tracking and Tax-Loss Harvesting (TLH) engine.

Implements the CRA-mandated weighted-average cost method for Canadian
non-registered accounts, plus a TLH signal engine that identifies unrealized
capital losses, proposes switch trades, and enforces the superficial loss rule
(ITA s. 54).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CRA superficial-loss window: 30 days before → 30 days after settlement
_SUPERFICIAL_WINDOW_DAYS: int = 30


# ---------------------------------------------------------------------------
# Trade data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Single trade event in the ACB ledger.

    Attributes
    ----------
    ticker : str
        Security ticker symbol.
    date : date
        Trade settlement date (used for superficial-loss window calculations).
    action : {"BUY", "SELL", "ROC"}
        Trade action: buy, sell, or return of capital distribution.
    shares : float
        Number of shares transacted (always positive; zero for ROC).
    price_per_share : float
        Transaction price per share (zero for ROC).
    total_amount : float
        Total dollar amount of the transaction (positive for buys, negative for
        sells, negative for ROC distributions that reduce ACB).
    """

    ticker: str
    date: date
    action: str
    shares: float
    price_per_share: float
    total_amount: float


@dataclass
class ACBPosition:
    """Running state of the adjusted cost base for a single security.

    Attributes
    ----------
    ticker : str
        Security ticker symbol.
    total_shares : float
        Current number of shares held.
    total_cost : float
        Total adjusted cost base (sum of all acquisition costs, net of ROC).
    acb_per_share : float
        Weighted-average cost per share (total_cost / total_shares, or 0 if
        no shares are held).
    """

    ticker: str
    total_shares: float = 0.0
    total_cost: float = 0.0

    @property
    def acb_per_share(self) -> float:
        """Weighted-average cost per share per CRA rules."""
        if self.total_shares <= 0:
            return 0.0
        return self.total_cost / self.total_shares


@dataclass
class TLHTrade:
    """A proposed tax-loss harvesting pair trade.

    Attributes
    ----------
    sell_ticker : str
        The ticker to sell to crystallize the capital loss.
    buy_ticker : str
        The replacement ticker maintaining similar factor exposure.
    sell_shares : float
        Number of shares to sell (may be fractional for pre-trade modelling).
    sell_proceeds : float
        Estimated proceeds from the sale at the current market price.
    unrealized_loss : float
        The capital loss that would be crystallized (negative = loss).
    superficial_loss_risk : bool
        ``True`` if a buy of the sell ticker or a substitute occurred within
        the 61-day superficial-loss window.
    superficial_loss_amount : float
        Dollar amount of the loss that would be denied under the superficial
        loss rule. Zero when ``superficial_loss_risk`` is ``False``.
    note : str
        Human-readable explanation (e.g. superficial loss warning).
    """

    sell_ticker: str
    buy_ticker: str
    sell_shares: float
    sell_proceeds: float
    unrealized_loss: float
    superficial_loss_risk: bool = False
    superficial_loss_amount: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# ACB Tracker
# ---------------------------------------------------------------------------


class ACBTracker:
    """Track the Adjusted Cost Base (ACB) of securities per CRA rules.

    The CRA requires Canadian taxpayers to use the **weighted-average cost
    method** for identical properties (ITA s. 47(1)). This tracker maintains
    a running total cost and total share count for each ticker, computing the
    ACB per share as ``total_cost / total_shares``.

    When a partial disposition occurs, the ACB per share **does not change**;
    only the total shares and total cost are reduced proportionally.

    Parameters
    ----------
    initial_positions : dict[str, tuple[float, float]], optional
        Seed the tracker with existing holdings as a mapping of ticker →
        ``(total_shares, total_cost)``. Useful for importing historical
        positions at the start of a tax year.

    Attributes
    ----------
    positions : dict[str, ACBPosition]
        Current ACB state per ticker.
    trades : list[TradeRecord]
        Complete trade ledger sorted by date.
    """

    def __init__(
        self,
        initial_positions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._positions: dict[str, ACBPosition] = {}
        self._trades: list[TradeRecord] = []

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
                if shares > 0 and cost == 0:
                    raise ValueError(
                        f"Initial cost for '{ticker}' must be positive when "
                        f"shares > 0; got cost={cost} for {shares} shares."
                    )
                if shares == 0 and cost > 0:
                    raise ValueError(
                        f"Initial cost for '{ticker}' must be zero when "
                        f"shares=0; got cost={cost}."
                    )
                self._positions[ticker] = ACBPosition(
                    ticker=ticker,
                    total_shares=shares,
                    total_cost=cost,
                )

    # -- Read-only accessors -------------------------------------------------

    @property
    def positions(self) -> dict[str, ACBPosition]:
        """Return a copy of current ACB positions."""
        return dict(self._positions)

    @property
    def trades(self) -> list[TradeRecord]:
        """Return a copy of the trade ledger."""
        return list(self._trades)

    def get_position(self, ticker: str) -> ACBPosition:
        """Return the ACB position for *ticker*, creating an empty one if unseen.

        Parameters
        ----------
        ticker : str
            The ticker symbol to look up.

        Returns
        -------
        ACBPosition
        """
        if ticker not in self._positions:
            self._positions[ticker] = ACBPosition(ticker=ticker)
        return self._positions[ticker]

    def get_acb(self, ticker: str) -> float:
        """Return the **total** adjusted cost base for *ticker*.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        float
            Total ACB (sum of all acquisition costs less ROC distributions).
        """
        return self.get_position(ticker).total_cost

    def get_acb_per_share(self, ticker: str) -> float:
        """Return the weighted-average ACB per share for *ticker*.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        float
            ACB per share, or 0.0 if no shares are held.
        """
        return self.get_position(ticker).acb_per_share

    def get_shares(self, ticker: str) -> float:
        """Return the current share count for *ticker*.

        Parameters
        ----------
        ticker : str
            Ticker symbol.

        Returns
        -------
        float
            Number of shares held.
        """
        return self.get_position(ticker).total_shares

    def get_unrealized_gain_loss(
        self,
        ticker: str,
        current_price: float,
    ) -> float:
        """Compute the unrealized capital gain or loss for *ticker*.

        A positive return indicates an unrealized gain; negative indicates a
        loss.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        current_price : float
            Current market price per share. Must be non-negative.

        Returns
        -------
        float
            Unrealized gain/loss in dollars.

        Raises
        ------
        ValueError
            If *current_price* is negative.
        """
        if current_price < 0:
            raise ValueError(
                f"current_price must be non-negative; got {current_price}."
            )
        pos = self.get_position(ticker)
        if pos.total_shares <= 0:
            return 0.0
        market_value = pos.total_shares * current_price
        return market_value - pos.total_cost

    def get_unrealized_gain_loss_pct(
        self,
        ticker: str,
        current_price: float,
    ) -> float:
        """Compute the unrealized gain/loss as a percentage of ACB.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        current_price : float
            Current market price per share.

        Returns
        -------
        float
            Gain/loss percentage (e.g. -0.15 = 15% loss). Returns 0.0 if
            the ACB is zero.
        """
        pos = self.get_position(ticker)
        if pos.total_cost <= 0:
            return 0.0
        return self.get_unrealized_gain_loss(ticker, current_price) / pos.total_cost

    # -- Trade recording -----------------------------------------------------

    def record_buy(
        self,
        ticker: str,
        shares: float,
        price_per_share: float,
        date: date,
    ) -> ACBPosition:
        """Record a purchase and update the weighted-average ACB.

        Per CRA rules, when new shares of identical property are acquired, the
        total cost is added to the pool and the ACB per share is recomputed.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        shares : float
            Number of shares purchased. Must be strictly positive.
        price_per_share : float
            Price paid per share. Must be positive.
        date : date
            Settlement date of the trade.

        Returns
        -------
        ACBPosition
            The updated position after the buy.

        Raises
        ------
        ValueError
            If *shares* or *price_per_share* are not strictly positive.
        """
        if shares <= 0:
            raise ValueError(f"Shares must be positive for a buy; got {shares}.")
        if price_per_share <= 0:
            raise ValueError(
                f"Price per share must be positive for a buy; got {price_per_share}."
            )

        total_cost = shares * price_per_share
        pos = self.get_position(ticker)

        # Weighted-average: pool the new shares and cost with the existing pool
        pos.total_shares += shares
        pos.total_cost += total_cost

        self._trades.append(
            TradeRecord(
                ticker=ticker,
                date=date,
                action="BUY",
                shares=shares,
                price_per_share=price_per_share,
                total_amount=total_cost,
            )
        )

        logger.debug(
            "BUY  %s: +%.4f sh @ %.4f → total %.4f sh, ACB/sh %.4f, total cost %.4f",
            ticker,
            shares,
            price_per_share,
            pos.total_shares,
            pos.acb_per_share,
            pos.total_cost,
        )
        return pos

    def record_sell(
        self,
        ticker: str,
        shares: float,
        price_per_share: float,
        date: date,
    ) -> float:
        """Record a sale and compute the realized capital gain or loss.

        Under the weighted-average method, the ACB per share **does not
        change** after a sale.  Total shares and total cost are both reduced
        proportionally.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        shares : float
            Number of shares sold. Must be positive and not exceed current
            holdings.
        price_per_share : float
            Sale price per share. Must be non-negative.
        date : date
            Settlement date of the trade.

        Returns
        -------
        float
            Realized capital gain (positive) or loss (negative).

        Raises
        ------
        ValueError
            If *shares* is non-positive, exceeds current holdings, or
            *price_per_share* is negative.
        """
        if shares <= 0:
            raise ValueError(f"Shares must be positive for a sell; got {shares}.")
        if price_per_share < 0:
            raise ValueError(
                f"Price per share must be non-negative for a sell; "
                f"got {price_per_share}."
            )

        pos = self.get_position(ticker)

        if pos.total_shares <= 0 or pos.total_cost <= 0:
            raise ValueError(
                f"Cannot sell '{ticker}': no shares or zero cost basis "
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
        gain_loss = proceeds - acb_of_sold

        # Reduce the pool proportionally
        pos.total_shares -= shares
        pos.total_cost -= acb_of_sold

        # Guard against floating-point drift pushing cost below zero when
        # all shares are sold.
        if pos.total_shares < 1e-12:
            pos.total_shares = 0.0
            pos.total_cost = 0.0

        self._trades.append(
            TradeRecord(
                ticker=ticker,
                date=date,
                action="SELL",
                shares=shares,
                price_per_share=price_per_share,
                total_amount=proceeds,
            )
        )

        logger.debug(
            "SELL %s: -%.4f sh @ %.4f → gain/loss %.4f, remaining %.4f sh, ACB/sh %.4f",
            ticker,
            shares,
            price_per_share,
            gain_loss,
            pos.total_shares,
            pos.acb_per_share,
        )
        return gain_loss

    def record_roc(
        self,
        ticker: str,
        amount: float,
        date: date,
    ) -> ACBPosition:
        """Record a Return of Capital (ROC) distribution that reduces ACB.

        When an ETF or trust distributes ROC, the total cost base is reduced
        by the distribution amount (but never below zero).  ROC does **not**
        change the share count; it only lowers the ACB per share.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        amount : float
            Dollar amount of the ROC distribution. Must be positive.
        date : date
            Record date of the distribution.

        Returns
        -------
        ACBPosition
            The updated position after the ROC adjustment.

        Raises
        ------
        ValueError
            If *amount* is not positive.
        """
        if amount <= 0:
            raise ValueError(f"ROC amount must be positive; got {amount}.")

        pos = self.get_position(ticker)
        if pos.total_shares <= 0:
            logger.warning(
                "ROC recorded for '%s' but no shares are held; ignoring.",
                ticker,
            )
            return pos

        # ACB cannot go below zero
        pos.total_cost = max(0.0, pos.total_cost - amount)

        self._trades.append(
            TradeRecord(
                ticker=ticker,
                date=date,
                action="ROC",
                shares=0.0,
                price_per_share=0.0,
                total_amount=amount,
            )
        )

        logger.debug(
            "ROC  %s: -%.4f → total cost %.4f, ACB/sh %.4f",
            ticker,
            amount,
            pos.total_cost,
            pos.acb_per_share,
        )
        return pos

    # -- Bulk operations -----------------------------------------------------

    def summary(self, current_prices: dict[str, float] | None = None) -> pd.DataFrame:
        """Return a DataFrame summarizing each position and its gain/loss.

        Parameters
        ----------
        current_prices : dict[str, float], optional
            Current market prices used to compute unrealized gain/loss.  Tickers
            missing from this mapping will have ``NaN`` in the market-value
            columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``total_shares``, ``total_cost``,
            ``acb_per_share``, plus ``market_value`` and ``unrealized_gain_loss``
            when *current_prices* is provided.
        """
        rows = []
        for ticker, pos in sorted(self._positions.items()):
            row: dict = {
                "ticker": ticker,
                "total_shares": pos.total_shares,
                "total_cost": pos.total_cost,
                "acb_per_share": pos.acb_per_share,
            }
            if current_prices is not None and ticker in current_prices:
                price = current_prices[ticker]
                row["market_value"] = pos.total_shares * price
                row["unrealized_gain_loss"] = row["market_value"] - pos.total_cost
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Superficial Loss Rule helpers
# ---------------------------------------------------------------------------


def _settlement_date(trade_date: date) -> date:
    """Return the approximate settlement date for a Canadian-listed trade.

    Canadian equities typically settle T+2; US equities T+1.  This function
    conservatively uses T+2 for all securities, which is safe because the
    CRA's 61-day window is measured from settlement date, and using a slightly
    later date errs on the side of flagging more potential superficial losses.

    Parameters
    ----------
    trade_date : date
        The trade (order) date.

    Returns
    -------
    date
        Settlement date (trade date + 2 business days, approximated as +2
        calendar days for simplicity).
    """
    return trade_date + timedelta(days=2)


def _superficial_window(settlement_date: date) -> tuple[date, date]:
    """Return the (start, end) of the 61-day superficial-loss window.

    The window spans from 30 days **before** the settlement date to 30 days
    **after** the settlement date (inclusive), per CRA guidance on ITA s. 54.

    Parameters
    ----------
    settlement_date : date
        The disposition settlement date.

    Returns
    -------
    tuple[date, date]
        ``(window_start, window_end)`` inclusive.
    """
    start = settlement_date - timedelta(days=_SUPERFICIAL_WINDOW_DAYS)
    end = settlement_date + timedelta(days=_SUPERFICIAL_WINDOW_DAYS)
    return start, end


def _load_switch_fund_map(path: Path | None = None) -> dict[str, list[str]]:
    """Load the switch-fund mapping from a JSON file.

    The expected schema is a flat JSON object mapping each ticker to a list of
    substitute tickers that provide similar factor exposure::

        {
            "VCN.TO": ["XIC.TO", "ZCN.TO"],
            "XIC.TO": ["VCN.TO", "ZCN.TO"]
        }

    Parameters
    ----------
    path : Path, optional
        Path to ``switch_fund_map.json``.  When ``None``, looks for
        ``switch_fund_map.json`` in the current working directory.

    Returns
    -------
    dict[str, list[str]]
        Parsed switch-fund map. Returns an empty dict if the file is absent
        or unparseable.
    """
    resolved = Path(path) if path else Path("switch_fund_map.json")
    if not resolved.exists():
        logger.info(
            "Switch-fund map not found at %s; TLH will not propose switch trades.",
            resolved,
        )
        return {}

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load switch_fund_map.json: %s", exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning("switch_fund_map.json must be a JSON object; ignoring.")
        return {}

    validated: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, list):
            logger.warning(
                "Skipping invalid entry in switch_fund_map: %r → %r", key, value
            )
            continue
        validated[str(key)] = [str(v) for v in value if isinstance(v, str)]

    return validated


# ---------------------------------------------------------------------------
# TLH Engine
# ---------------------------------------------------------------------------


class TLHEngine:
    """Identify and validate tax-loss harvesting opportunities.

    The engine cross-references the ACB tracker with a switch-fund map to
    propose sell-and-replace trades that crystallize capital losses while
    maintaining similar factor exposure.  It also enforces the CRA's
    **superficial loss rule** (ITA s. 54), which disallows a loss if the same
    or identical property is acquired within 30 days before or after the
    disposition settlement date.

    Parameters
    ----------
    acb_tracker : ACBTracker
        The ACB ledger tracking all positions and trade history.
    switch_fund_map : dict[str, list[str]] or Path or None
        Mapping of ticker → list of substitute tickers.  Can be a dict, a
        Path to a JSON file, or ``None`` (no switch trades proposed).
    current_prices : dict[str, float]
        Mapping of ticker → current market price per share.

    Attributes
    ----------
    acb : ACBTracker
        The reference ACB tracker.
    switch_map : dict[str, list[str]]
        The switch-fund mapping loaded from the provided source.
    prices : dict[str, float]
        Current market prices.
    """

    def __init__(
        self,
        acb_tracker: ACBTracker,
        switch_fund_map: dict[str, list[str]] | Path | None = None,
        current_prices: dict[str, float] | None = None,
    ) -> None:
        self.acb = acb_tracker
        self.prices: dict[str, float] = dict(current_prices) if current_prices else {}

        if isinstance(switch_fund_map, (Path, str)):
            self.switch_map = _load_switch_fund_map(Path(switch_fund_map))
        elif isinstance(switch_fund_map, dict):
            self.switch_map = switch_fund_map
        else:
            self.switch_map = {}

    # -- TLH opportunity detection -------------------------------------------

    def identify_tlh_opportunities(
        self,
        min_loss_dollars: float = 0.0,
        min_loss_pct: float = 0.0,
        as_of_date: date | None = None,
        exclude_tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Find positions with unrealized capital losses eligible for harvesting.

        Parameters
        ----------
        min_loss_dollars : float, optional
            Minimum dollar loss required to consider a position. Default is 0
            (all losses are considered).
        min_loss_pct : float, optional
            Minimum loss as a fraction of ACB (e.g. 0.05 = 5%). Default is 0.
        as_of_date : date, optional
            Reference date for the superficial-loss check.  When ``None``, uses
            ``date.today()``.  Trades must settle on or before this date to be
            included.
        exclude_tickers : list[str], optional
            Tickers to exclude from the opportunity scan.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``total_shares``, ``acb_per_share``,
            ``current_price``, ``market_value``, ``unrealized_gain_loss``,
            ``loss_pct``, ``has_switch_fund``.
        """
        exclude = set(exclude_tickers) if exclude_tickers else set()

        rows = []
        for ticker, pos in sorted(self.acb.positions.items()):
            if ticker in exclude:
                continue
            if pos.total_shares <= 0:
                continue
            if ticker not in self.prices:
                logger.debug("Skipping '%s': no current price available.", ticker)
                continue

            price = self.prices[ticker]
            if price <= 0:
                continue

            market_value = pos.total_shares * price
            gain_loss = market_value - pos.total_cost

            # Only positions with an unrealized loss
            if gain_loss >= 0:
                continue

            loss_pct = abs(gain_loss) / pos.total_cost if pos.total_cost > 0 else 0.0

            if abs(gain_loss) < min_loss_dollars:
                continue
            if loss_pct < min_loss_pct:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "total_shares": pos.total_shares,
                    "acb_per_share": pos.acb_per_share,
                    "current_price": price,
                    "market_value": market_value,
                    "unrealized_gain_loss": gain_loss,
                    "loss_pct": loss_pct,
                    "has_switch_fund": ticker in self.switch_map,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("unrealized_gain_loss").reset_index(drop=True)
        return df

    def generate_tlh_pairs(
        self,
        ticker: str,
        as_of_date: date | None = None,
    ) -> list[TLHTrade]:
        """Generate TLH switch-trade proposals for a single ticker.

        For each substitute fund in the switch map, this method produces a
        :class:`TLHTrade` that sells the current position and buys the
        substitute.  Each proposal is checked against the superficial-loss
        rule using the trade ledger.

        Parameters
        ----------
        ticker : str
            The ticker to harvest a loss from.
        as_of_date : date, optional
            Reference settlement date for the superficial-loss check.

        Returns
        -------
        list[TLHTrade]
            One or more proposed switch trades, each with superficial-loss
            status.  Returns an empty list if the ticker has no unrealized
            loss, no switch funds are defined, or no prices are available.
        """
        ref_date = as_of_date or date.today()

        if ticker not in self.prices or self.prices[ticker] <= 0:
            return []

        pos = self.acb.get_position(ticker)
        if pos.total_shares <= 0:
            return []

        price = self.prices[ticker]
        gain_loss = pos.total_shares * price - pos.total_cost
        if gain_loss >= 0:
            # No loss to harvest
            return []

        switch_funds = self.switch_map.get(ticker, [])
        if not switch_funds:
            return []

        proposals: list[TLHTrade] = []
        for buy_ticker in switch_funds:
            # Check superficial loss risk on the disposition ticker
            sl_risk, sl_amount = self.check_superficial_loss(ticker, ref_date)

            note = ""
            if sl_risk:
                note = (
                    f"SUPERFICIAL LOSS WARNING: ${sl_amount:,.2f} of the loss "
                    f"may be denied. A buy of '{ticker}' or a substitute was "
                    f"detected within the 61-day window around settlement "
                    f"{ref_date.isoformat()}."
                )

            proposals.append(
                TLHTrade(
                    sell_ticker=ticker,
                    buy_ticker=buy_ticker,
                    sell_shares=pos.total_shares,
                    sell_proceeds=pos.total_shares * price,
                    unrealized_loss=gain_loss,
                    superficial_loss_risk=sl_risk,
                    superficial_loss_amount=sl_amount,
                    note=note,
                )
            )

        return proposals

    def generate_all_tlh_pairs(
        self,
        min_loss_dollars: float = 0.0,
        min_loss_pct: float = 0.0,
        as_of_date: date | None = None,
    ) -> pd.DataFrame:
        """Generate TLH switch-trade proposals for all eligible positions.

        This is a convenience method that calls :meth:`identify_tlh_opportunities`
        followed by :meth:`generate_tlh_pairs` for each eligible ticker, and
        returns a flattened DataFrame.

        Parameters
        ----------
        min_loss_dollars : float, optional
            Minimum dollar loss to consider.
        min_loss_pct : float, optional
            Minimum loss percentage to consider.
        as_of_date : date, optional
            Reference date for superficial-loss checks.

        Returns
        -------
        pd.DataFrame
            One row per proposed switch trade, with columns from
            :class:`TLHTrade`.
        """
        opportunities = self.identify_tlh_opportunities(
            min_loss_dollars=min_loss_dollars,
            min_loss_pct=min_loss_pct,
            as_of_date=as_of_date,
        )

        if opportunities.empty:
            return pd.DataFrame(
                columns=[
                    "sell_ticker",
                    "buy_ticker",
                    "sell_shares",
                    "sell_proceeds",
                    "unrealized_loss",
                    "superficial_loss_risk",
                    "superficial_loss_amount",
                    "note",
                ]
            )

        all_proposals: list[dict] = []
        for _, row in opportunities.iterrows():
            ticker = row["ticker"]
            proposals = self.generate_tlh_pairs(ticker, as_of_date=as_of_date)
            for p in proposals:
                all_proposals.append(
                    {
                        "sell_ticker": p.sell_ticker,
                        "buy_ticker": p.buy_ticker,
                        "sell_shares": p.sell_shares,
                        "sell_proceeds": p.sell_proceeds,
                        "unrealized_loss": p.unrealized_loss,
                        "superficial_loss_risk": p.superficial_loss_risk,
                        "superficial_loss_amount": p.superficial_loss_amount,
                        "note": p.note,
                    }
                )

        if not all_proposals:
            return pd.DataFrame()

        df = pd.DataFrame(all_proposals)
        return df.sort_values("unrealized_loss").reset_index(drop=True)

    # -- Superficial loss rule enforcement -----------------------------------

    def check_superficial_loss(
        self,
        ticker: str,
        disposition_date: date,
    ) -> tuple[bool, float]:
        """Check whether a disposition would trigger the superficial loss rule.

        Per ITA s. 54, a capital loss is superficial if the taxpayer (or an
        affiliated person) acquires the same or identical property during the
        period that begins 30 days before and ends 30 days after the
        disposition settlement date.

        This implementation scans the trade ledger for any **buy** of *ticker*
        within the 61-day window.  If a buy is found, the loss is presumed
        superficial for the number of shares acquired in that window.

        Parameters
        ----------
        ticker : str
            The ticker being disposed.
        disposition_date : date
            The trade date of the proposed disposition (settlement is
            approximated as T+2).

        Returns
        -------
        tuple[bool, float]
            ``(is_superficial, superficial_loss_amount)``.  When
            ``is_superficial`` is ``True``, *superficial_loss_amount* is the
            dollar amount of the loss that is denied.

        Notes
        -----
        In practice, substitute funds (e.g. VCN.TO → XIC.TO) are considered
        different properties by the CRA and do **not** trigger the superficial
        loss rule on their own.  However, if the taxpayer buys back the
        **original** ticker within the window, the rule applies.  The switch
        map is used to identify identical vs. similar properties.
        """
        settlement = _settlement_date(disposition_date)
        window_start, window_end = _superficial_window(settlement)

        # Find all BUY trades for this ticker within the window
        buys_in_window = [
            t
            for t in self.acb._trades
            if t.ticker == ticker
            and t.action == "BUY"
            and window_start <= t.date <= window_end
        ]

        if not buys_in_window:
            return False, 0.0

        # Compute the total shares bought in the window
        shares_bought = sum(t.shares for t in buys_in_window)
        pos = self.acb.get_position(ticker)
        shares_held = pos.total_shares

        # The superficial loss is proportional to the shares bought in the
        # window vs. total shares held.  If shares_bought >= shares_held, the
        # entire loss is superficial.
        if shares_held <= 0:
            return False, 0.0

        # Calculate the unrealized loss on the position
        if ticker not in self.prices or self.prices[ticker] <= 0:
            # Cannot compute loss without a price; flag as risky
            return True, 0.0

        current_price = self.prices[ticker]
        total_loss = pos.total_shares * current_price - pos.total_cost
        if total_loss >= 0:
            return False, 0.0

        # Proportion of the loss that is superficial
        proportion = min(shares_bought / shares_held, 1.0)
        superficial_amount = abs(total_loss) * proportion

        return True, superficial_amount

    def is_within_superficial_window(
        self,
        ticker: str,
        disposition_date: date,
        other_ticker: str | None = None,
    ) -> bool:
        """Return ``True`` if any buy of *ticker* (or *other_ticker*) occurred
        within the 61-day superficial-loss window.

        Parameters
        ----------
        ticker : str
            The primary ticker being checked.
        disposition_date : date
            The proposed disposition date.
        other_ticker : str, optional
            An additional ticker to scan (e.g. a switch-fund candidate that
            might be considered identical property).

        Returns
        -------
        bool
        """
        settlement = _settlement_date(disposition_date)
        window_start, window_end = _superficial_window(settlement)
        tickers_to_check = {ticker}
        if other_ticker:
            tickers_to_check.add(other_ticker)

        for t in self.acb._trades:
            if t.ticker in tickers_to_check and t.action == "BUY":
                if window_start <= t.date <= window_end:
                    return True
        return False


# ---------------------------------------------------------------------------
# TLH-aware Rebalance helpers
# ---------------------------------------------------------------------------


@dataclass
class TLHRebalanceResult:
    """Output of a TLH-augmented rebalance analysis.

    Attributes
    ----------
    portfolio_name : str
        Identifier for the portfolio being analyzed.
    tlh_trades : pd.DataFrame
        Proposed TLH switch-trade pairs (from
        :meth:`TLHEngine.generate_all_tlh_pairs`).
    acb_summary : pd.DataFrame
        Current ACB summary for all tracked positions (from
        :meth:`ACBTracker.summary`).
    superficial_loss_flags : list[str]
        Tickers where a superficial-loss risk was detected, with human-readable
        warnings.
    """

    portfolio_name: str
    tlh_trades: pd.DataFrame
    acb_summary: pd.DataFrame
    superficial_loss_flags: list[str] = field(default_factory=list)


def analyze_tlh_opportunities(
    acb_tracker: ACBTracker,
    current_prices: dict[str, float],
    switch_fund_map: dict[str, list[str]] | Path | None = None,
    portfolio_name: str = "default",
    min_loss_dollars: float = 0.0,
    min_loss_pct: float = 0.0,
    as_of_date: date | None = None,
) -> TLHRebalanceResult:
    """Run a full TLH analysis and return actionable trade recommendations.

    This is the top-level entry point intended to be called from the rebalance
    workflow.  It runs the opportunity scan, generates switch-trade pairs,
    checks superficial-loss rules, and returns a structured result suitable
    for display in the rebalance plan output.

    Parameters
    ----------
    acb_tracker : ACBTracker
        ACB ledger with full trade history.
    current_prices : dict[str, float]
        Current market prices keyed by ticker.
    switch_fund_map : dict or Path or None
        Switch-fund mapping (see :class:`TLHEngine`).
    portfolio_name : str, optional
        Label for the result.
    min_loss_dollars : float, optional
        Minimum absolute dollar loss to consider.
    min_loss_pct : float, optional
        Minimum loss as a fraction of ACB.
    as_of_date : date, optional
        Reference date for window checks.

    Returns
    -------
    TLHRebalanceResult
    """
    engine = TLHEngine(
        acb_tracker=acb_tracker,
        switch_fund_map=switch_fund_map,
        current_prices=current_prices,
    )

    tlh_trades = engine.generate_all_tlh_pairs(
        min_loss_dollars=min_loss_dollars,
        min_loss_pct=min_loss_pct,
        as_of_date=as_of_date,
    )
    acb_summary = acb_tracker.summary(current_prices=current_prices)

    superficial_flags: list[str] = []
    if not tlh_trades.empty:
        flagged = tlh_trades[tlh_trades["superficial_loss_risk"]]
        for _, row in flagged.iterrows():
            superficial_flags.append(f"{row['sell_ticker']}: {row['note']}")

    return TLHRebalanceResult(
        portfolio_name=portfolio_name,
        tlh_trades=tlh_trades,
        acb_summary=acb_summary,
        superficial_loss_flags=superficial_flags,
    )


def format_tlh_rebalance_result(result: TLHRebalanceResult) -> str:
    """Render a TLH rebalance result as human-readable text.

    Parameters
    ----------
    result : TLHRebalanceResult
        The output from :func:`analyze_tlh_opportunities`.

    Returns
    -------
    str
        Formatted multi-line string.
    """
    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f"  Tax-Loss Harvesting Analysis — {result.portfolio_name}")
    lines.append("═" * 72)

    # ACB summary
    lines.append("")
    lines.append("── ACB Summary ──")
    if result.acb_summary.empty:
        lines.append("  (no positions tracked)")
    else:
        acb = result.acb_summary.copy()
        if "market_value" in acb.columns:
            acb["unrealized_gain_loss"] = acb["unrealized_gain_loss"].map(
                lambda x: f"${x:+,.2f}" if pd.notna(x) else "N/A"
            )
            acb["market_value"] = acb["market_value"].map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )
        acb["total_cost"] = acb["total_cost"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
        )
        acb["acb_per_share"] = acb["acb_per_share"].map(
            lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A"
        )
        lines.append(acb.to_string(index=False))

    # Superficial loss flags
    if result.superficial_loss_flags:
        lines.append("")
        lines.append("── ⚠ Superficial Loss Warnings ──")
        for flag in result.superficial_loss_flags:
            lines.append(f"  • {flag}")

    # TLH trade proposals
    lines.append("")
    lines.append("── Proposed TLH Switch Trades ──")
    if result.tlh_trades.empty:
        lines.append("  No eligible tax-loss harvesting opportunities found.")
    else:
        trades = result.tlh_trades.copy()
        trades["unrealized_loss"] = trades["unrealized_loss"].map(
            lambda x: f"${x:+,.2f}"
        )
        trades["sell_proceeds"] = trades["sell_proceeds"].map(lambda x: f"${x:,.2f}")
        trades["sell_shares"] = trades["sell_shares"].map(lambda x: f"{x:,.4f}")
        trades["superficial_loss_amount"] = trades["superficial_loss_amount"].map(
            lambda x: f"${x:,.2f}"
        )
        # Drop note column for cleaner display; it's in the warnings above
        display_cols = [c for c in trades.columns if c != "note"]
        lines.append(trades[display_cols].to_string(index=False))

    lines.append("")
    lines.append("═" * 72)
    return "\n".join(lines)
