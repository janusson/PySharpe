"""Equity curve visualization utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import utils as viz_utils

if TYPE_CHECKING:  # pragma: no cover - type checking aide
    import matplotlib.pyplot as plt
    import pandas as pd

    from pysharpe.data.fetcher import PriceFetcher

logger = logging.getLogger(__name__)


def plot_equity_curves(
    optimized: pd.Series,
    baseline: pd.Series,
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Plot an equity curve comparing an optimized strategy against a baseline.

    Args:
        optimized: Series of the portfolio value over time for the optimized strategy.
        baseline: Series of the portfolio value over time for the baseline strategy.
        ax: Optional axes to draw onto. A new figure/axes pair is created when ``None``.
        show: When ``True`` call ``plt.show()`` before returning.
        title: Optional plot title override.

    Returns:
        Matplotlib axes containing the plot.
    """
    if ax is None:
        plt = viz_utils.require_matplotlib()
        _, ax = plt.subplots(figsize=(10, 6))
    else:
        plt = viz_utils.require_matplotlib()

    # Align indexes if possible, though matplotlib handles it gracefully usually
    ax.plot(
        optimized.index,
        optimized.values,
        label="Sharpe-Optimized Portfolio",
        linewidth=2,
        color="blue",
    )

    ax.plot(
        baseline.index,
        baseline.values,
        label="Buy-and-Hold Baseline",
        linewidth=2,
        linestyle="--",
        color="orange",
    )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)", fontweight="bold")

    if title is None:
        title = "Equity Curve Comparison: Optimized vs. Baseline"
    ax.set_title(title, fontweight="bold")

    # Format y-axis as dollars
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax


def plot_comparative_returns(
    tickers: list[str],
    fetcher: PriceFetcher,
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Plot cumulative percentage returns for multiple tickers from a common
    inception date.

    Historical adjusted close prices are fetched for every ticker and aligned
    to the inception date of the youngest asset by dropping rows with missing
    data. Cumulative percentage returns are then plotted against a common
    timeline.

    Args:
        tickers: List of ticker symbols to compare.
        fetcher: A :class:`~pysharpe.data.fetcher.PriceFetcher` instance used
            to download price history.
        ax: Optional axes to draw onto. A new figure/axes pair is created when
            ``None``.
        show: When ``True`` call ``plt.show()`` before returning.
        title: Optional plot title override.

    Returns:
        Matplotlib axes containing the plot.
    """
    import pandas as pd

    if ax is None:
        plt = viz_utils.require_matplotlib()
        _, ax = plt.subplots(figsize=(10, 6))
    else:
        plt = viz_utils.require_matplotlib()

    # Pull max available Adj Close for each ticker.
    price_series: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = fetcher.fetch_history(ticker, period="max", interval="1d")
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            continue
        if df.empty:
            logger.warning("Skipping %s: no data returned", ticker)
            continue
        # Prefer adjusted close when available, otherwise fall back to close.
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        price_series[ticker] = df[col].squeeze()

    if not price_series:
        raise ValueError("No valid price data for any of the requested tickers.")

    # Combine into a single DataFrame and align to the youngest asset's inception.
    combined = pd.DataFrame(price_series)
    common_data = combined.dropna()

    if common_data.empty:
        raise ValueError("No overlapping date range across the requested tickers.")

    # Normalize to cumulative percentage returns from the common inception date.
    cumulative_returns = (common_data / common_data.iloc[0] - 1) * 100

    # Plot each ticker.
    for ticker in cumulative_returns.columns:
        ax.plot(
            cumulative_returns.index,
            cumulative_returns[ticker].values,
            label=ticker,
            linewidth=2,
        )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontweight="bold")

    if title is None:
        start_date = common_data.index[0].strftime("%Y-%m-%d")
        title = f"Cumulative Returns from {start_date}"
    ax.set_title(title, fontweight="bold")

    # Format y-axis as percentage and add a 0 % reference line.
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:+.0f}%"))
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax


def plot_holdings_history(
    collated_prices: pd.DataFrame,
    *,
    min_trading_days: int = 20,
    ax: plt.Axes | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Plot cumulative percentage returns for every holding in collated price data.

    All price series are aligned to the latest common inception date by dropping
    rows with missing data. If the resulting intersection spans fewer than
    *min_trading_days* trading days a :class:`~pysharpe.exceptions.PySharpeError`
    is raised with a descriptive message listing each ticker's available date
    range so the user can diagnose which ticker is limiting the view.

    Args:
        collated_prices: DataFrame with dates as the index and tickers as columns.
        min_trading_days: Minimum number of overlapping trading days required
            for the plot to be considered meaningful.
        ax: Optional axes to draw onto. A new figure/axes pair is created when
            ``None``.
        show: When ``True`` call ``plt.show()`` before returning.
        title: Optional plot title override.

    Returns:
        Matplotlib axes containing the plot.

    Raises:
        PySharpeError: If the overlapping date range across all tickers is
            shorter than *min_trading_days*.
    """
    from pysharpe.exceptions import PySharpeError

    if collated_prices.empty:
        raise PySharpeError("Collated price data is empty — nothing to plot.")

    if ax is None:
        plt = viz_utils.require_matplotlib()
        _, ax = plt.subplots(figsize=(12, 7))
    else:
        plt = viz_utils.require_matplotlib()

    # Find the intersection where *all* tickers have data.
    common_data = collated_prices.dropna()

    if common_data.empty:
        # Build a diagnostic message showing each ticker's date range.
        ranges: list[str] = []
        for col in collated_prices.columns:
            valid = collated_prices[col].dropna()
            if valid.empty:
                ranges.append(f"  {col}: no data")
            else:
                ranges.append(
                    f"  {col}: {valid.index[0].strftime('%Y-%m-%d')} "
                    f"to {valid.index[-1].strftime('%Y-%m-%d')} "
                    f"({len(valid)} days)"
                )
        raise PySharpeError(
            "No overlapping date range across the requested tickers. "
            "Each ticker's available range:\n" + "\n".join(ranges)
        )

    overlap_days = len(common_data)
    if overlap_days < min_trading_days:
        ranges: list[str] = []
        for col in collated_prices.columns:
            valid = collated_prices[col].dropna()
            start = valid.index[0].strftime("%Y-%m-%d") if not valid.empty else "N/A"
            end = valid.index[-1].strftime("%Y-%m-%d") if not valid.empty else "N/A"
            ranges.append(f"  {col}: {start} to {end}")
        raise PySharpeError(
            f"Intersection of dates across holdings is only {overlap_days} "
            f"trading days (minimum {min_trading_days} required for a "
            f"meaningful view). Each ticker's available range:\n" + "\n".join(ranges)
        )

    # Normalize to cumulative percentage returns from the common inception.
    cumulative_returns = (common_data / common_data.iloc[0] - 1) * 100

    for ticker in cumulative_returns.columns:
        ax.plot(
            cumulative_returns.index,
            cumulative_returns[ticker].values,
            label=ticker,
            linewidth=1.5,
        )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontweight="bold")

    if title is None:
        start_date = common_data.index[0].strftime("%Y-%m-%d")
        end_date = common_data.index[-1].strftime("%Y-%m-%d")
        ticker_count = len(collated_prices.columns)
        title = (
            f"Holdings History — Cumulative Returns from {start_date} "
            f"to {end_date} ({ticker_count} tickers)"
        )
    ax.set_title(title, fontweight="bold")

    # Format y-axis as percentage.
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:+.0f}%"))
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax
