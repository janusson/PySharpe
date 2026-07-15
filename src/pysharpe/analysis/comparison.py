"""Stateless Head-to-Head fund comparison engine.

Evaluates two assets side-by-side using vectorised metrics without invoking
multi-asset optimisation solvers or Value Averaging (VA) allocation pipelines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pysharpe.metrics import (
    annualize_volatility,
    cagr,
    calmar_ratio,
    compute_returns,
    max_drawdown_duration,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

if TYPE_CHECKING:  # pragma: no cover
    from pysharpe.data.fetcher import PriceFetcher

logger = logging.getLogger(__name__)


def _rolling_tracking_error(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute 1-year rolling annualised tracking error between two series."""
    diff = returns_a - returns_b
    rolling_std = diff.rolling(window=window, min_periods=window).std(ddof=1)
    return rolling_std * np.sqrt(periods_per_year)  # pyright: ignore[reportReturnType]


def _rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    window: int = 252,
) -> pd.Series:
    """Compute 1-year rolling return correlation between two series."""
    return returns_a.rolling(window=window, min_periods=window).corr(returns_b)  # pyright: ignore[reportReturnType]


def _fetch_and_align(
    ticker_a: str,
    ticker_b: str,
    start_date: str | None,
    end_date: str | None,
    base_currency: str,
    fetcher: PriceFetcher | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch price histories for two tickers and return aligned DataFrames.

    Uses ``CollationService`` with ``YFinancePriceFetcher`` (DuckDB-cached)
    to retrieve daily close prices.  Proxy resolution via ``proxy_map.json``
    is handled automatically by ``CollationService``.  FX conversion is
    applied by the fetcher when ``base_currency`` differs from the native
    currency.

    Args:
        ticker_a: First ticker symbol.
        ticker_b: Second ticker symbol.
        start_date: Optional ISO-format start date.
        end_date: Optional ISO-format end date.
        base_currency: Target currency for price conversion.
        fetcher: Optional ``PriceFetcher`` for test injection.  When ``None``
            a ``YFinancePriceFetcher`` is used.

    Returns:
        Tuple of (prices_a, prices_b) as single-column DataFrames with
        aligned, overlapping date indices.
    """
    from pysharpe.data.collation import CollationService
    from pysharpe.data.fetcher import YFinancePriceFetcher

    if fetcher is None:
        fetcher = YFinancePriceFetcher({"auto_adjust": True})

    service = CollationService(fetcher)

    period = "max"
    interval = "1d"

    for ticker in (ticker_a, ticker_b):
        service.download_portfolio_prices(
            [ticker],
            period=period,
            interval=interval,
            start=start_date,
            end=end_date,
        )

    collated = service.collate_portfolio(
        f"_head_to_head_{ticker_a}_{ticker_b}",
        [ticker_a, ticker_b],
    )

    if (
        collated.empty
        or ticker_a not in collated.columns
        or ticker_b not in collated.columns
    ):
        raise ValueError(
            f"Could not retrieve overlapping price data for {ticker_a} and {ticker_b}."
        )

    prices_a = collated[[ticker_a]].dropna()
    prices_b = collated[[ticker_b]].dropna()

    # Align both on the intersection of dates.
    common_index = prices_a.index.intersection(prices_b.index)
    if len(common_index) < 2:
        raise ValueError(
            f"Insufficient overlapping data for {ticker_a} and {ticker_b} "
            f"(only {len(common_index)} common dates)."
        )

    prices_a = prices_a.loc[common_index]
    prices_b = prices_b.loc[common_index]

    return prices_a, prices_b


def _compute_comparison(
    prices_a: pd.DataFrame,
    prices_b: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Core stateless computation: given two aligned price DataFrames, produce
    the side-by-side comparison table.

    This inner function is separated from ``compare_two_funds`` so that tests
    can inject synthetic price data directly without any network fetcher.
    """
    series_a: pd.Series = prices_a.iloc[:, 0]  # type: ignore[assignment]
    series_b: pd.Series = prices_b.iloc[:, 0]  # type: ignore[assignment]

    returns_a = compute_returns(series_a)
    returns_b = compute_returns(series_b)
    assert isinstance(returns_a, pd.Series)
    assert isinstance(returns_b, pd.Series)

    # --- Scalar metrics ---
    cagr_a = cagr(series_a)
    cagr_b = cagr(series_b)

    vol_a = annualize_volatility(returns_a, periods_per_year=252)
    vol_b = annualize_volatility(returns_b, periods_per_year=252)

    mdd_a = maximum_drawdown(series_a)
    mdd_b = maximum_drawdown(series_b)

    mdd_dur_a = max_drawdown_duration(series_a)
    mdd_dur_b = max_drawdown_duration(series_b)

    sharpe_a = sharpe_ratio(
        returns_a, risk_free_rate=risk_free_rate, periods_per_year=252
    )
    sharpe_b = sharpe_ratio(
        returns_b, risk_free_rate=risk_free_rate, periods_per_year=252
    )

    sortino_a = sortino_ratio(
        returns_a, risk_free_rate=risk_free_rate, periods_per_year=252
    )
    sortino_b = sortino_ratio(
        returns_b, risk_free_rate=risk_free_rate, periods_per_year=252
    )

    calmar_a = calmar_ratio(series_a)
    calmar_b = calmar_ratio(series_b)

    te_series = _rolling_tracking_error(returns_a, returns_b, window=252)
    te_mean = float(te_series.mean())

    corr_series = _rolling_correlation(returns_a, returns_b, window=252)
    corr_mean = float(corr_series.mean())

    # --- Assemble result ---
    result = pd.DataFrame(
        {
            "Metric": [
                "CAGR",
                "Annualized Volatility",
                "Max Drawdown Depth",
                "Max Drawdown Duration (days)",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "1Y Rolling Tracking Error (mean)",
                "1Y Rolling Return Correlation (mean)",
            ],
            ticker_a: [
                cagr_a,
                vol_a,
                mdd_a,
                mdd_dur_a,
                sharpe_a,
                sortino_a,
                calmar_a,
                te_mean,
                corr_mean,
            ],
            ticker_b: [
                cagr_b,
                vol_b,
                mdd_b,
                mdd_dur_b,
                sharpe_b,
                sortino_b,
                calmar_b,
                te_mean,
                corr_mean,
            ],
        }
    )
    result.set_index("Metric", inplace=True)
    return result


def compare_two_funds(
    ticker_a: str,
    ticker_b: str,
    start_date: str | None = None,
    end_date: str | None = None,
    base_currency: str = "CAD",
    risk_free_rate: float = 0.0,
    fetcher: PriceFetcher | None = None,
) -> pd.DataFrame:
    """Compare two funds side-by-side using standardised risk/return metrics.

    Fetches daily close prices for ``ticker_a`` and ``ticker_b``, converts to
    ``base_currency`` if needed, and returns a comparison DataFrame containing:

    - **CAGR**: Compound Annual Growth Rate.
    - **Annualized Volatility**: Standard deviation of daily returns annualised.
    - **Max Drawdown Depth**: Largest peak-to-trough decline as a percentage.
    - **Max Drawdown Duration**: Longest drawdown in trading days.
    - **Sharpe Ratio**: Annualised excess return per unit of total risk.
    - **Sortino Ratio**: Annualised excess return per unit of downside risk.
    - **Calmar Ratio**: Annualised return divided by the absolute max drawdown.
    - **1Y Rolling Tracking Error**: Mean of the 1-year rolling tracking error.
    - **1Y Rolling Correlation**: Mean of the 1-year rolling return correlation
      between the two assets.

    Args:
        ticker_a: First fund ticker symbol.
        ticker_b: Second fund ticker symbol.
        start_date: Optional ISO-format start date (inclusive).
        end_date: Optional ISO-format end date (inclusive).
        base_currency: Target currency for price conversion (default ``"CAD"``).
        risk_free_rate: Annual risk-free rate as a decimal (default ``0.0``).
        fetcher: Optional ``PriceFetcher`` for test injection.  When ``None``
            a ``YFinancePriceFetcher`` with auto-adjust is used.

    Returns:
        DataFrame with metrics as rows and funds as columns.  Pair-wise metrics
        (Tracking Error, Correlation) appear as the same value in both columns
        since they describe the relationship rather than an individual fund.

    Raises:
        ValueError: If insufficient overlapping data exists for the two tickers.
    """
    prices_a, prices_b = _fetch_and_align(
        ticker_a, ticker_b, start_date, end_date, base_currency, fetcher=fetcher
    )
    return _compute_comparison(prices_a, prices_b, ticker_a, ticker_b, risk_free_rate)


__all__ = ["compare_two_funds"]
