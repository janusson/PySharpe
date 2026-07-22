"""Adjusted Cost Base (ACB) tracking for Canadian TFSA portfolios.

Implements the CRA-mandated weighted-average cost method (ITA s. 47(1)) and
return-of-capital adjustment rules for Canadian accounts.

Tax-Loss Harvesting (TLH) is categorically prohibited in TFSA accounts per
Canadian treaty guidelines and is not implemented in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

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
    realized_gain_loss: float = 0.0


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
    ) -> tuple[ACBPosition, float]:
        """Record a Return of Capital (ROC) distribution that reduces ACB.

        When an ETF or trust distributes ROC, the total cost base is reduced
        by the distribution amount.  If the ROC exceeds the remaining ACB, the
        excess is realized immediately as a capital gain per CRA rules, and the
        ACB is reset to zero.

        ROC does **not** change the share count; it only lowers the ACB per
        share.

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
        tuple[ACBPosition, float]
            A ``(position, realized_capital_gain)`` pair.  The realized gain is
            zero unless the ROC amount exceeds the remaining ACB.

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
            return pos, 0.0

        # CRA: ROC in excess of ACB is realized as a capital gain
        realized_gain = max(0.0, amount - pos.total_cost)
        pos.total_cost = max(0.0, pos.total_cost - amount)

        self._trades.append(
            TradeRecord(
                ticker=ticker,
                date=date,
                action="ROC",
                shares=0.0,
                price_per_share=0.0,
                total_amount=amount,
                realized_gain_loss=realized_gain,
            )
        )

        logger.debug(
            "ROC  %s: -%.4f → total cost %.4f, ACB/sh %.4f, realized gain %.4f",
            ticker,
            amount,
            pos.total_cost,
            pos.acb_per_share,
            realized_gain,
        )
        return pos, realized_gain

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
