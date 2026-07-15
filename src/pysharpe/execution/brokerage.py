"""Export buy orders to brokerage-specific CSV upload formats.

Bridges the gap between PySharpe's mathematical allocation engine and the
actual trade-execution workflow at Canadian discount brokerages.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from pysharpe.execution.rebalance import RebalancePlan


class Brokerage(enum.Enum):
    """Supported brokerage CSV upload formats."""

    QUESTRADE = "questrade"
    WEALTHSIMPLE = "wealthsimple"
    INTERACTIVE_BROKERS = "interactive_brokers"


@dataclass(frozen=True)
class BrokerageExportConfig:
    """Order-level parameters applied during CSV export.

    Attributes
    ----------
    order_type : str
        Order type included in the exported CSV. ``"MKT"`` (market) and
        ``"LMT"`` (limit) are the most common.  Default is ``"MKT"``.
    limit_price : float or None
        Limit price per share.  Only meaningful when *order_type* is
        ``"LMT"``.  When ``None`` (default), the column is left blank.
    time_in_force : str
        How long the order remains active.  ``"DAY"`` is the safest default
        and covers most brokerage platforms.  Default is ``"DAY"``.
    account : str or None
        Account identifier required by Interactive Brokers BasketTrader.
        When ``None`` (default), the column is omitted.
    """

    order_type: str = "MKT"
    limit_price: float | None = None
    time_in_force: str = "DAY"
    account: str | None = None

    @property
    def limit_price_str(self) -> str:
        """Return the limit price as a 2-decimal string, or empty."""
        if self.limit_price is None:
            return ""
        return f"{self.limit_price:.2f}"


# ---------------------------------------------------------------------------
# Column access helpers
# ---------------------------------------------------------------------------
_EXPECTED_COLUMNS = frozenset(
    {
        "ticker",
        "recommended_allocation",
        "recommended_shares",
        "latest_price",
    }
)


def _validate_buy_orders(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` when required columns are missing."""
    missing = _EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"buy_orders DataFrame is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def _prepare_buy_frame(
    source: RebalancePlan | pd.DataFrame,
    *,
    account: str | None = None,
) -> pd.DataFrame:
    """Normalise the input into a cleaned buy-orders ``DataFrame``.

    Parameters
    ----------
    source : RebalancePlan or pd.DataFrame
        Either a full rebalance plan or a pre-filtered buy-orders frame.
    account : str or None
        When the plan is multi-account, only include orders for this account.

    Returns
    -------
    pd.DataFrame
        Buy orders with at least ``ticker``, ``recommended_allocation``,
        ``recommended_shares``, and ``latest_price`` columns.
    """
    # Avoid circular import — RebalancePlan is only used for type narrowing.
    from pysharpe.execution.rebalance import RebalancePlan  # noqa: PLC0415

    if isinstance(source, RebalancePlan):
        if account is not None and source.is_multi_account:
            assert source.account_allocations is not None
            if account not in source.account_allocations:
                available = ", ".join(sorted(source.account_allocations))
                raise ValueError(
                    f"Account {account!r} not found in plan. "
                    f"Available accounts: {available}"
                )
            df = source.account_allocations[account].copy()
            df = df.loc[df["recommended_allocation"] > 0]
        else:
            df = source.buy_orders
    else:
        df = source.copy()

    _validate_buy_orders(df)
    return df


# ---------------------------------------------------------------------------
# Per-brokerage formatters
# ---------------------------------------------------------------------------


_QUESTRADE_HEADER = "Symbol,Action,Quantity,Order Type,Limit Price,Duration\n"


def _format_questrade(df: pd.DataFrame, config: BrokerageExportConfig) -> str:
    """Render buy orders as a Questrade MyQuestrade CSV.

    Questrade's trade-upload CSV expects:

    - **Symbol** (ticker)
    - **Action** (``BUY`` or ``SELL``)
    - **Quantity** (whole shares; fractional shares are floored)
    - **Order Type** (``MKT`` or ``LMT``)
    - **Limit Price** (blank for market orders)
    - **Duration** (``DAY``)

    References
    ----------
    https://www.questrade.com/learning/questrade-basics/upload-trades
    """
    limit = config.limit_price_str if config.order_type.upper() == "LMT" else ""
    rows = []
    for _, row in df.iterrows():
        shares = int(row["recommended_shares"])
        if shares <= 0:
            continue
        rows.append(
            {
                "Symbol": row["ticker"],
                "Action": "BUY",
                "Quantity": shares,
                "Order Type": config.order_type.upper(),
                "Limit Price": limit,
                "Duration": config.time_in_force.upper(),
            }
        )

    if not rows:
        return _QUESTRADE_HEADER

    result = pd.DataFrame(rows)
    # Ensure column ordering matches Questrade's expected format
    result = result[
        ["Symbol", "Action", "Quantity", "Order Type", "Limit Price", "Duration"]
    ]
    return result.to_csv(index=False)


_WEALTHSIMPLE_HEADER = "Ticker,Action,Shares,Estimated Price,Estimated Cost (CAD)\n"


def _format_wealthsimple(df: pd.DataFrame, config: BrokerageExportConfig) -> str:
    """Render buy orders as a Wealthsimple Trade reference CSV.

    Wealthsimple Trade does not provide a native batch-upload CSV feature.
    This format produces a human-readable reference table that users can
    follow during manual order entry.

    Columns
    -------
    - **Ticker**
    - **Action** (``BUY``)
    - **Shares** (may be fractional for fractional-share eligible ETFs)
    - **Estimated Price** (from ``latest_price``)
    - **Estimated Cost (CAD)**
    """
    rows = []
    for _, row in df.iterrows():
        shares = row["recommended_shares"]
        if shares <= 0:
            continue
        price = row.get("latest_price", 0.0)
        rows.append(
            {
                "Ticker": row["ticker"],
                "Action": "BUY",
                "Shares": f"{shares:.4f}".rstrip("0").rstrip("."),
                "Estimated Price": f"${price:,.2f}",
                "Estimated Cost (CAD)": f"${shares * price:,.2f}",
            }
        )

    if not rows:
        return _WEALTHSIMPLE_HEADER

    result = pd.DataFrame(rows)
    result = result[
        ["Ticker", "Action", "Shares", "Estimated Price", "Estimated Cost (CAD)"]
    ]
    return result.to_csv(index=False)


_IBKR_HEADER_BASE = "Symbol,Action,Quantity,Order Type,Limit Price,Time-in-Force"


def _format_interactive_brokers(
    df: pd.DataFrame, config: BrokerageExportConfig
) -> str:
    """Render buy orders as an Interactive Brokers BasketTrader CSV.

    IBKR's BasketTrader import expects:

    - **Symbol** (ticker)
    - **Action** (``BUY`` or ``SELL``)
    - **Quantity** (whole shares for non-fractional symbols)
    - **Order Type** (``MKT``, ``LMT``, ``REL``, etc.)
    - **Limit Price** (blank for market orders)
    - **Time-in-Force** (``DAY``, ``GTC``, ``IOC``)
    - **Account** (optional; IBKR account ID)

    References
    ----------
    https://www.interactivebrokers.com/en/trading/basket-trader.php
    """
    limit = config.limit_price_str if config.order_type.upper() == "LMT" else ""
    rows = []
    for _, row in df.iterrows():
        shares = int(row["recommended_shares"])
        if shares <= 0:
            continue
        entry = {
            "Symbol": row["ticker"],
            "Action": "BUY",
            "Quantity": shares,
            "Order Type": config.order_type.upper(),
            "Limit Price": limit,
            "Time-in-Force": config.time_in_force.upper(),
        }
        if config.account is not None:
            entry["Account"] = config.account
        rows.append(entry)

    has_account = config.account is not None
    if not rows:
        header = _IBKR_HEADER_BASE
        if has_account:
            header += ",Account"
        return header + "\n"

    result = pd.DataFrame(rows)
    cols = ["Symbol", "Action", "Quantity", "Order Type", "Limit Price", "Time-in-Force"]
    if has_account:
        cols.append("Account")
    result = result[cols]
    return result.to_csv(index=False)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
_FORMATTERS = {
    Brokerage.QUESTRADE: _format_questrade,
    Brokerage.WEALTHSIMPLE: _format_wealthsimple,
    Brokerage.INTERACTIVE_BROKERS: _format_interactive_brokers,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_buy_orders(
    source: RebalancePlan | pd.DataFrame,
    brokerage: Brokerage,
    *,
    config: BrokerageExportConfig | None = None,
    account: str | None = None,
) -> str:
    """Convert buy orders into a brokerage-specific CSV upload string.

    Parameters
    ----------
    source : RebalancePlan or pd.DataFrame
        Either a :class:`RebalancePlan` produced by
        :func:`~pysharpe.execution.rebalance.build_rebalance_plan`, or a
        pre-filtered ``DataFrame`` containing only rows with positive
        ``recommended_allocation`` (equivalent to ``plan.buy_orders``).
    brokerage : Brokerage
        Target brokerage format.
    config : BrokerageExportConfig or None, optional
        Order-level parameters (order type, limit price, time-in-force,
        account).  When ``None``, market-day-order defaults are used.
    account : str or None, optional
        For multi-account plans, specify which account's buy orders to
        export.  Ignored for single-account plans or raw DataFrames.

    Returns
    -------
    str
        CSV-formatted content ready to be written to a ``.csv`` file and
        uploaded to the brokerage platform.

    Raises
    ------
    ValueError
        If the input DataFrame is missing required columns, or if the
        requested *account* is not present in a multi-account plan.

    Examples
    --------
    >>> from pysharpe import build_rebalance_plan, Brokerage, export_buy_orders
    >>> plan = build_rebalance_plan("growth", new_cash=5000, ...)
    >>> csv_content = export_buy_orders(plan, Brokerage.QUESTRADE)
    >>> Path("questrade_orders.csv").write_text(csv_content)

    For a multi-account plan, specify the target account:

    >>> csv_content = export_buy_orders(plan, Brokerage.INTERACTIVE_BROKERS,
    ...                                 account="TFSA")
    """
    if config is None:
        config = BrokerageExportConfig()

    df = _prepare_buy_frame(source, account=account)
    formatter = _FORMATTERS[brokerage]
    return formatter(df, config)
