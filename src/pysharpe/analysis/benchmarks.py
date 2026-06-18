"""Baseline benchmarks for portfolio comparison."""

from __future__ import annotations

import logging

import pandas as pd

from pysharpe import metrics
from pysharpe.data.fetcher import (
    DuckDBCachedPriceFetcher,
    YFinancePriceFetcher,
    apply_fx_conversion,
)

logger = logging.getLogger(__name__)

# Standard Canadian all-in-one asset allocation ETFs
CANADIAN_BENCHMARKS = {
    "VEQT.TO": "Vanguard All-Equity ETF (100/0)",
    "XEQT.TO": "iShares Core Equity ETF (100/0)",
    "VGRO.TO": "Vanguard Growth ETF (80/20)",
    "XGRO.TO": "iShares Core Growth ETF (80/20)",
    "VBAL.TO": "Vanguard Balanced ETF (60/40)",
    "XBAL.TO": "iShares Core Balanced ETF (60/40)",
}


def fetch_benchmark_metrics(
    tickers: list[str],
    start_date: str,
    end_date: str,
    base_currency: str = "CAD",
) -> pd.DataFrame:
    """Fetch benchmark data and compute performance metrics.

    Args:
        tickers: List of benchmark tickers (e.g., ["VEQT.TO", "VGRO.TO"]).
        start_date: ISO8601 start date for the analysis period.
        end_date: ISO8601 end date for the analysis period.
        base_currency: Target currency for evaluation (default "CAD").

    Returns:
        DataFrame with Ticker, Annualized Return, Annualized Volatility, and Sharpe Ratio.
    """

    if not tickers:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
            ]
        )

    fetcher = DuckDBCachedPriceFetcher(YFinancePriceFetcher())

    price_data: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            df = fetcher.fetch_history(
                ticker,
                period="max",
                interval="1d",
                start=start_date,
                end=end_date,
            )
            if not df.empty:
                # Ensure index is naive DatetimeIndex for consistency with pysharpe standards
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                price_data[ticker] = df["Close"]
        except Exception as exc:
            logger.warning("Failed to fetch benchmark %s: %s", ticker, exc)

    if not price_data:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
            ]
        )

    prices_df = pd.DataFrame(price_data)

    # Apply FX conversion if needed
    prices_df = apply_fx_conversion(
        prices_df, base_currency=base_currency, fetcher=fetcher
    )

    # Compute returns
    returns = metrics.compute_returns(prices_df)

    # Calculate metrics
    ann_return = metrics.annualize_return(returns)
    ann_vol = metrics.annualize_volatility(returns)
    sharpe = metrics.sharpe_ratio(returns)

    # Handle single ticker edge case (metrics might return float instead of Series)
    if isinstance(ann_return, float):
        results = [
            {
                "Ticker": tickers[0],
                "Annualized Return": ann_return,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
            }
        ]
    else:
        results = []
        for ticker in prices_df.columns:
            results.append(
                {
                    "Ticker": ticker,
                    "Annualized Return": ann_return.get(ticker, 0.0),
                    "Annualized Volatility": ann_vol.get(ticker, 0.0),
                    "Sharpe Ratio": sharpe.get(ticker, 0.0),
                }
            )

    return pd.DataFrame(results)
