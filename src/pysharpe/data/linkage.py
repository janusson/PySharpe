"""Data Linkage and Feature Engineering using DuckDB.

This module provides tools to join financial market data with external macro datasets.
It utilizes an embedded DuckDB instance to efficiently perform complex SQL aggregations,
window functions (rolling averages), and temporal shifts (lead/lag) directly in the database.
"""

import logging
from typing import Optional, Union

import duckdb
import pandas as pd

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
    ) -> pd.DataFrame:
        """Join market data with macro data and engineer temporal features using SQL window functions.

        This demonstrates "Data Linkage" by joining two disparate datasets on a date/time key,
        and computes features like rolling averages and lagged variables suitable for causal inference.

        Args:
            market_table (str): The registered name of the market data table. Must have 'date' and 'price' columns.
            macro_table (Optional[str]): The registered name of the macro data table. Must have 'date' column.
            rolling_window (int): The window size for rolling averages.

        Returns:
            pd.DataFrame: The joined and feature-engineered dataset.
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
                    LAG(price, 1) OVER (ORDER BY date) AS price_lag_1
                FROM {market_table}
            )
        """

        if macro_table:
            # If macro data is provided, join and add lagged macro features
            query += f""",
            joined_data AS (
                SELECT
                    m.*,
                    mac.* EXCLUDE (date)
                FROM market_features m
                LEFT JOIN {macro_table} mac ON m.date = mac.date
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
            SELECT * FROM market_features ORDER BY date;
            """

        return self.execute_query(query)

    def close(self):
        """Close the database connection."""
        self.conn.close()
