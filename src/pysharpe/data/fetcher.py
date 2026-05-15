"""Price history fetching abstractions."""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def apply_fx_conversion(
    prices: pd.DataFrame,
    base_currency: str = "CAD",
    fetcher: Optional[PriceFetcher] = None,
) -> pd.DataFrame:
    """Adjust price data for foreign exchange rates.

    Args:
        prices: Historical price data with tickers as columns.
        base_currency: The target currency for all assets (default "CAD").
        fetcher: Optional PriceFetcher to use for downloading FX data.

    Returns:
        DataFrame with all prices converted to the base currency.

    Raises:
        ValueError: If FX data cannot be downloaded or is empty.
    """

    if fetcher is None:
        fetcher = YFinancePriceFetcher()

    adjusted_prices = prices.copy()

    start_date = prices.index.min().strftime("%Y-%m-%d")
    end_date = prices.index.max().strftime("%Y-%m-%d")

    for ticker in prices.columns:
        try:
            # Ticker.info can be slow/flaky, but is the source of truth for currency
            ticker_info = yf.Ticker(ticker).info
            currency = ticker_info.get("currency")
        except Exception as exc:
            logger.warning(
                "Could not determine currency for %s: %s. Assuming %s.",
                ticker,
                exc,
                base_currency,
            )
            currency = base_currency

        if not currency or currency.upper() == base_currency.upper():
            continue

        fx_ticker = f"{currency}{base_currency}=X"
        logger.warning(
            "Notice: %s is priced in %s. Prices converted to %s using daily %s rates.",
            ticker,
            currency,
            base_currency,
            fx_ticker,
        )

        try:
            fx_data = fetcher.fetch_history(
                fx_ticker,
                period="max",
                interval="1d",
                start=start_date,
                end=end_date,
            )
            if fx_data.empty:
                raise ValueError(f"FX data for {fx_ticker} is empty.")
            fx_series = fx_data["Close"]
        except Exception as exc:
            if isinstance(exc, ValueError) and "is empty" in str(exc):
                raise
            raise ValueError(
                f"Failed to download FX data for {fx_ticker}: {exc}"
            ) from exc

        # Reindex to match prices.index, using ffill and then bfill
        if fx_series.index.tz is not None:
            fx_series.index = fx_series.index.tz_localize(None)

        # Ensure prices index is also naive if it has timezones
        prices_index = prices.index
        if hasattr(prices_index, "tz") and prices_index.tz is not None:
            prices_index = prices_index.tz_localize(None)

        aligned_fx = fx_series.reindex(prices_index).ffill().bfill()
        adjusted_prices[ticker] = adjusted_prices[ticker] * aligned_fx

    return adjusted_prices


class PriceHistoryError(RuntimeError):
    """Raised when a price history request fails."""


class PriceFetcher(ABC):
    """Abstract base class for price history providers."""

    @abstractmethod
    def fetch_history(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a price history for ``ticker``.

        Args:
            ticker: Symbol recognised by the downstream API.
            period: Rolling window used when ``start``/``end`` are omitted.
            interval: Sampling frequency (for example ``"1d"`` or ``"1wk"``).
            start: Optional ISO8601 start date.
            end: Optional ISO8601 end date.

        Returns:
            Price history indexed by timestamp.

        Raises:
            PriceHistoryError: Implementations should wrap provider-specific
                failures in :class:`PriceHistoryError` for consistency.
        """

        raise NotImplementedError


class DuckDBCachedPriceFetcher(PriceFetcher):
    """A caching decorator for PriceFetcher that uses DuckDB.

    Data is stored in a local DuckDB file and is considered fresh for 24 hours.
    """

    def __init__(
        self, fetcher: PriceFetcher, db_path: str = "pysharpe_cache.db"
    ) -> None:
        self.fetcher = fetcher
        self.db_path = db_path
        self._init_db()

    def _lazy_module(self):
        try:
            import duckdb
        except ImportError as exc:
            raise PriceHistoryError(
                "duckdb must be installed to use DuckDBCachedPriceFetcher."
            ) from exc
        return duckdb

    def _init_db(self):
        duckdb = self._lazy_module()
        with duckdb.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_key VARCHAR PRIMARY KEY,
                    ticker VARCHAR,
                    fetch_time TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_prices (
                    cache_key VARCHAR,
                    date_val TIMESTAMP WITH TIME ZONE,
                    open_val DOUBLE,
                    high_val DOUBLE,
                    low_val DOUBLE,
                    close_val DOUBLE,
                    volume_val DOUBLE,
                    dividends_val DOUBLE,
                    stock_splits_val DOUBLE
                )
            """)

    def _generate_key(
        self,
        ticker: str,
        period: str,
        interval: str,
        start: Optional[str],
        end: Optional[str],
    ) -> str:
        params = {
            "ticker": ticker,
            "period": period or "",
            "interval": interval,
            "start": start or "",
            "end": end or "",
        }
        key_string = json.dumps(params, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def fetch_history(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        duckdb = self._lazy_module()
        cache_key = self._generate_key(ticker, period, interval, start, end)

        with duckdb.connect(self.db_path) as conn:
            # Check if cache is valid (< 24 hours old)
            query = """
                SELECT fetch_time FROM cache_metadata
                WHERE cache_key = ? AND fetch_time >= CURRENT_TIMESTAMP - INTERVAL '24' HOUR
            """
            result = conn.execute(query, [cache_key]).fetchone()

            if result:
                logger.info("Cache hit for %s", ticker)
                df = conn.execute(
                    """
                    SELECT date_val AS "Date", open_val AS "Open", high_val AS "High",
                           low_val AS "Low", close_val AS "Close", volume_val AS "Volume",
                           dividends_val AS "Dividends", stock_splits_val AS "Stock Splits"
                    FROM cached_prices
                    WHERE cache_key = ?
                    ORDER BY date_val
                """,
                    [cache_key],
                ).df()

                if not df.empty:
                    df.set_index("Date", inplace=True)
                    # DuckDB returns pandas Timestamps, but timezone might need explicit conversion or be UTC standard
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    return df

            logger.info("Cache miss or expired for %s", ticker)

            # Fetch fresh data
            df = self.fetcher.fetch_history(
                ticker=ticker, period=period, interval=interval, start=start, end=end
            )

            if df.empty:
                return df

            # Ensure incoming index is UTC to avoid timezone issues when querying back from DuckDB
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("UTC")
                else:
                    df.index = df.index.tz_localize("UTC")

            # Store in cache
            conn.execute("DELETE FROM cache_metadata WHERE cache_key = ?", [cache_key])
            conn.execute("DELETE FROM cached_prices WHERE cache_key = ?", [cache_key])

            conn.execute(
                """
                INSERT INTO cache_metadata (cache_key, ticker, fetch_time)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                [cache_key, ticker],
            )

            df_to_store = df.copy()
            index_name = df_to_store.index.name or "Date"
            df_to_store = df_to_store.reset_index()

            # Rename columns to match db schema to avoid case sensitivity/spaces issues
            # Missing columns will be NaN which DuckDB maps to NULL
            expected_cols = {
                index_name: "date_val",
                "Open": "open_val",
                "High": "high_val",
                "Low": "low_val",
                "Close": "close_val",
                "Volume": "volume_val",
                "Dividends": "dividends_val",
                "Stock Splits": "stock_splits_val",
            }
            df_to_store = df_to_store.rename(columns=expected_cols)
            df_to_store["cache_key"] = cache_key

            # Ensure all schema columns exist
            for col in [
                "date_val",
                "open_val",
                "high_val",
                "low_val",
                "close_val",
                "volume_val",
                "dividends_val",
                "stock_splits_val",
            ]:
                if col not in df_to_store.columns:
                    df_to_store[col] = 0.0

            conn.register("temp_df", df_to_store)
            conn.execute("""
                INSERT INTO cached_prices
                SELECT cache_key, date_val, open_val, high_val, low_val, close_val, volume_val, dividends_val, stock_splits_val
                FROM temp_df
            """)

            return df


class YFinancePriceFetcher(PriceFetcher):
    """yfinance-backed fetcher with thin logging and validation.

    Example:
        >>> from pysharpe.data.fetcher import YFinancePriceFetcher
        >>> fetcher = YFinancePriceFetcher({"auto_adjust": True})
        >>> isinstance(fetcher, YFinancePriceFetcher)
        True
    """

    def __init__(self, history_kwargs: Optional[Dict[str, object]] = None) -> None:
        self._history_overrides = history_kwargs or {}

    def _lazy_module(self):
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:
            raise PriceHistoryError(
                "yfinance must be installed to download market data."
            ) from exc
        return yf

    def fetch_history(
        self,
        ticker: str,
        *,
        period: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download price data for ``ticker`` using yfinance.

        Args:
            ticker: Symbol to request.
            period: Rolling window requested when explicit dates are missing.
            interval: Sampling frequency for the returned data.
            start: Optional start date in ISO format.
            end: Optional end date in ISO format.

        Returns:
            DataFrame indexed by timestamp with pricing columns provided by
            `yfinance`.

        Raises:
            PriceHistoryError: If the download fails or returns an empty frame.

        Example:
            >>> from pysharpe.data.fetcher import YFinancePriceFetcher
            >>> fetcher = YFinancePriceFetcher()
            >>> fetcher.fetch_history  # doctest: +ELLIPSIS
            <bound method ...>
        """

        yf = self._lazy_module()
        request_payload: Dict[str, object] = {"interval": interval}
        if start:
            request_payload["start"] = start
        if end:
            request_payload["end"] = end
        if not start and not end:
            request_payload["period"] = period

        request_payload.update(self._history_overrides)

        logger.info("Fetching history for %s", ticker)
        try:
            history = yf.Ticker(ticker).history(**request_payload)
        except Exception as exc:  # pragma: no cover - network issues
            raise PriceHistoryError(
                f"Failed to download history for {ticker}: {exc}"
            ) from exc

        if history.empty:
            raise PriceHistoryError(f"No data returned for {ticker}")

        return history
