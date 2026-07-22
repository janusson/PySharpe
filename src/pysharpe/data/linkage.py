"""Data Linkage and Feature Engineering using DuckDB.

This module provides tools to join financial market data with external macro datasets.
It utilizes an embedded DuckDB instance to efficiently perform complex SQL aggregations,
window functions (rolling averages), and temporal shifts (lead/lag) directly in the database.
"""

import logging
from typing import Optional

import duckdb
import pandas as pd

from .fetcher import PriceFetcher, YFinancePriceFetcher

logger = logging.getLogger(__name__)


class DataLinker:
    """Links disparate datasets and engineers features using an embedded DuckDB database."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize the DuckDB connection.

        Args:
            db_path (str): Path to the DuckDB file. Defaults to ":memory:" for an in-memory database.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=self.db_path)
        logger.info(f"Initialized DuckDB connection at {self.db_path}")

    def register_data(self, name: str, data: pd.DataFrame) -> None:
        """Register a pandas DataFrame as a virtual table in DuckDB.

        Args:
            name (str): The name to assign to the virtual table.
            data (pd.DataFrame): The DataFrame to register.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")
        self.conn.register(name, data)
        logger.debug(f"Registered DataFrame as virtual table: '{name}'")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query and return the result as a pandas DataFrame.

        Args:
            query (str): The SQL query string.

        Returns:
            pd.DataFrame: The query result.
        """
        logger.debug(f"Executing query:\n{query}")
        return self.conn.execute(query).df()

    def get_enhanced_market_data(
        self,
        market_table: str = "market_data",
        macro_table: Optional[str] = None,
        rolling_window: int = 7,
        short_window: int = 20,
        long_window: int = 50,
    ) -> pd.DataFrame:
        """Join market data with macro data and engineer temporal features using SQL window functions.

        This demonstrates "Data Linkage" by joining two disparate datasets on a date/time key,
        and computes features like rolling averages, lagged variables, and moving average crossover signals.

        Args:
            market_table (str): The registered name of the market data table. Must have 'date' and 'price' columns.
            macro_table (Optional[str]): The registered name of the macro data table. Must have 'date' column.
            rolling_window (int): The window size for a general rolling average.
            short_window (int): The window size for the short moving average in crossover calculation.
            long_window (int): The window size for the long moving average in crossover calculation.

        Returns:
            pd.DataFrame: The joined and feature-engineered dataset, including:
            - 'price_rolling_avg': Rolling average of prices.
            - 'price_lag_1': Price lagged by one period.
            - 'short_ma': Short moving average of prices.
            - 'long_ma': Long moving average of prices.
            - 'ma_crossover_signal': 1 if short_ma > long_ma, -1 if short_ma < long_ma, else 0.
        """
        # Base CTE for market data with a rolling average window function
        query = f"""
            WITH market_features AS (
                SELECT
                    *,
                    AVG(price) OVER (
                        ORDER BY date
                        ROWS BETWEEN {rolling_window - 1} PRECEDING AND CURRENT ROW
                    ) AS price_rolling_avg,
                    LAG(price, 1) OVER (ORDER BY date) AS price_lag_1,
                    AVG(price) OVER (
                        ORDER BY date
                        ROWS BETWEEN {short_window - 1} PRECEDING AND CURRENT ROW
                    ) AS short_ma,
                    AVG(price) OVER (
                        ORDER BY date
                        ROWS BETWEEN {long_window - 1} PRECEDING AND CURRENT ROW
                    ) AS long_ma
                FROM {market_table}
            ),
            final_features AS (
                SELECT
                    *,
                    CASE
                        WHEN short_ma > long_ma THEN 1
                        WHEN short_ma < long_ma THEN -1
                        ELSE 0
                    END AS ma_crossover_signal
                FROM market_features
            )
        """

        if macro_table:
            # If macro data is provided, join and add lagged macro features
            query += f""",
            joined_data AS (
                SELECT
                    f.*,
                    mac.* EXCLUDE (date)
                FROM final_features f
                LEFT JOIN {macro_table} mac ON f.date = mac.date
            )
            SELECT
                *,
                -- Example: creating a lag of the first non-date macro column generically
                -- In a real scenario, you'd specify exact columns, but here we assume the macro table
                -- has been pre-filtered to the target indicator.
                -- For demonstration, we simply return the joined set which duckdb easily handles.
            FROM joined_data
            ORDER BY date;
            """
        else:
            query += """
            SELECT * FROM final_features ORDER BY date;
            """

        return self.execute_query(query)

    def calculate_trend_signals(
        self,
        market_table: str = "market_data",
        short_window: int = 20,
        long_window: int = 50,
    ) -> pd.DataFrame:
        """Calculate trend signals for a single ticker's market data.

        Computes moving-average crossover signals and 30-day vs historical
        volatility ratios from price data already registered as *market_table*.

        The computation proceeds in two stages:

        1. **MA Crossover** — delegated to
           :meth:`get_enhanced_market_data`, which computes ``short_ma``,
           ``long_ma``, and ``ma_crossover_signal`` (1 = bullish,
           -1 = bearish, 0 = neutral).
        2. **Volatility ratio** — annualised 30-day rolling standard
           deviation of log returns divided by the full-period annualised
           standard deviation ::

              vol_ratio = σ_30d / σ_hist

           where σ = STDDEV_SAMP(log_returns) * √252.

        Parameters
        ----------
        market_table : str
            Registered DuckDB table name containing ``date`` and ``price``
            columns.
        short_window : int
            Look-back window (rows) for the short moving average.
        long_window : int
            Look-back window (rows) for the long moving average.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``price``, ``price_rolling_avg``,
            ``price_lag_1``, ``short_ma``, ``long_ma``,
            ``ma_crossover_signal``, ``vol_30d``, ``vol_hist``,
            ``volatility_ratio``.  Returns an empty DataFrame if fewer than
            2 rows are available (log returns require at least 2 prices).
        """
        enhanced = self.get_enhanced_market_data(
            market_table=market_table,
            short_window=short_window,
            long_window=long_window,
        )

        if enhanced.empty or len(enhanced) < 2:
            empty_cols = [
                "date",
                "price",
                "price_rolling_avg",
                "price_lag_1",
                "short_ma",
                "long_ma",
                "ma_crossover_signal",
                "vol_30d",
                "vol_hist",
                "volatility_ratio",
            ]
            return pd.DataFrame(columns=empty_cols)

        self.register_data("_trend_enhanced", enhanced)

        query = """
            WITH returns AS (
                SELECT
                    date,
                    price,
                    price_rolling_avg,
                    price_lag_1,
                    short_ma,
                    long_ma,
                    ma_crossover_signal,
                    LN(price / NULLIF(LAG(price, 1) OVER (ORDER BY date), 0)) AS log_ret
                FROM _trend_enhanced
            ),
            volatility AS (
                SELECT
                    date,
                    price,
                    price_rolling_avg,
                    price_lag_1,
                    short_ma,
                    long_ma,
                    ma_crossover_signal,
                    STDDEV_SAMP(log_ret) OVER (
                        ORDER BY date
                        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                    ) * SQRT(252) AS vol_30d,
                    STDDEV_SAMP(log_ret) OVER () * SQRT(252) AS vol_hist
                FROM returns
            )
            SELECT
                *,
                CASE
                    WHEN vol_hist IS NOT NULL AND vol_hist > 0
                    THEN vol_30d / vol_hist
                    ELSE NULL
                END AS volatility_ratio
            FROM volatility
            ORDER BY date
        """
        return self.execute_query(query)

    def close(self):
        """Close the database connection."""
        self.conn.close()


class HistoryLinker:
    """Stitches a target asset's price history with a proxy to simulate a longer track record."""

    def __init__(
        self,
        proxy_map: dict[str, str],
        fx_adjust: bool = False,
        fetcher: Optional[PriceFetcher] = None,
    ):
        """Initialize the HistoryLinker.

        Args:
            proxy_map (dict): Mapping of target tickers to their proxy tickers.
            fx_adjust (bool): Whether to perform FX adjustment via USDCAD=X.
            fetcher (Optional[PriceFetcher]): Provider for historical data.
        """
        self.proxy_map = proxy_map
        self.fx_adjust = fx_adjust
        self.fetcher = fetcher or YFinancePriceFetcher()
        self.crypto_tickers = {"BTC-USD", "ETH-USD"}  # Define known 24/7 tickers

    def _is_24_7_asset(self, ticker: str) -> bool:
        return ticker in self.crypto_tickers

    def get_stitched_series(self, target_ticker: str, start_date: str) -> pd.Series:
        """Fetch and stitch historical data for the target using its proxy.

        Args:
            target_ticker (str): The symbol of the asset to fetch and backfill.
            start_date (str): The beginning of the historical window (ISO format).

        Returns:
            pd.Series: The combined stitched price history.
        """
        logger.info("Fetching target data for %s", target_ticker)
        target_df = self.fetcher.fetch_history(
            target_ticker, period="max", interval="1d", start=start_date, end=None
        )
        target_price = target_df["Close"].dropna()
        target_price.name = target_ticker

        if target_ticker not in self.proxy_map:
            logger.info(
                "No proxy defined for %s. Returning original series.", target_ticker
            )
            return target_price

        proxy_ticker = self.proxy_map[target_ticker]
        logger.info("Fetching proxy data for %s", proxy_ticker)
        proxy_df = self.fetcher.fetch_history(
            proxy_ticker, period="max", interval="1d", start=start_date, end=None
        )
        proxy_price = proxy_df["Close"].dropna()

        # Handle 24/7 assets in proxy_price by aligning to an equity calendar
        if self._is_24_7_asset(proxy_ticker):
            logger.info(
                "Proxy %s is a 24/7 asset. Aligning to equity calendar.", proxy_ticker
            )
            # Fetch SPY to get a standard equity market calendar
            spy_df = self.fetcher.fetch_history(
                "SPY", period="max", interval="1d", start=start_date, end=None
            )
            equity_calendar = spy_df.index

            # Reindex proxy_price to equity_calendar using ffill
            proxy_price = proxy_price.reindex(equity_calendar, method="ffill")
            proxy_price = proxy_price.dropna()

        if self.fx_adjust:
            logger.info("Fetching USDCAD=X for FX adjustment")
            fx_df = self.fetcher.fetch_history(
                "USDCAD=X", period="max", interval="1d", start=start_date, end=None
            )
            fx_price = fx_df["Close"].dropna()

            # Reindex FX to match the proxy dates and forward/backward fill missing values
            fx_price = fx_price.reindex(proxy_price.index).ffill().bfill()

            # Multiply proxy price by 1 / USDCAD as requested
            proxy_price = proxy_price * (1.0 / fx_price)

        # Find the inception date (T0) of the target
        # Use intersection to ensure both series have data on the handover date
        common_dates = target_price.index.intersection(proxy_price.index)

        if common_dates.empty:
            logger.warning(
                "No overlapping dates for %s and %s. Returning original.",
                target_ticker,
                proxy_ticker,
            )
            return target_price

        t0 = common_dates.min()

        # Calculate the scalar to match prices at T0
        scalar = target_price.loc[t0] / proxy_price.loc[t0]
        logger.info(
            "Stitching %s with %s at %s (scalar: %.4f)",
            target_ticker,
            proxy_ticker,
            t0.date(),
            scalar,
        )

        # Scale the proxy data
        scaled_proxy = proxy_price * scalar

        # Splice: scaled proxy before T0, target from T0 onwards
        proxy_portion = scaled_proxy.loc[scaled_proxy.index < t0]
        target_portion = target_price.loc[target_price.index >= t0]

        stitched = pd.concat([proxy_portion, target_portion])
        stitched.name = target_ticker

        # Drop duplicated indices, keeping the target portion if there is an overlap issue
        stitched = stitched[~stitched.index.duplicated(keep="last")]

        return stitched.sort_index()
