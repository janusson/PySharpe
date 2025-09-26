"""Utilities for retrieving market data from Yahoo Finance."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import pandas as pd


def fetch_price_history(
    symbols: Iterable[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices for the provided symbols.

    Parameters
    ----------
    symbols:
        An iterable of ticker symbols understood by Yahoo Finance.
    start:
        Optional start date for the price history. Defaults to letting Yahoo
        Finance decide (usually max available history).
    end:
        Optional end date for the price history. Defaults to the current date.
    interval:
        Yahoo Finance interval string (``"1d"`` by default).

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by date with one column per symbol representing the
        adjusted close prices.
    """

    if not symbols:
        raise ValueError("At least one ticker symbol must be provided.")

    # Import lazily so that the package can be imported without yfinance installed.
    import yfinance as yf

    tickers = " ".join(symbols)
    data = yf.download(tickers, start=start, end=end, interval=interval, progress=False)

    if data.empty:
        raise ValueError("No price data was returned for the requested parameters.")

    # yfinance returns a multi-index column when multiple tickers are provided.
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Adj Close"]
    else:
        data = data.rename("Adj Close")

    return data.sort_index()
