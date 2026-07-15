"""Tests for the DuckDB write-through price cache.

.. note::

    **Canadian Data Pipeline** — DuckDB caching wraps only
    YFinancePriceFetcher.  Cache keys include ticker, period, and
    interval parameters.  Cache invalidation is tested via parameter
    changes (different period/interval/ticker produce cache misses).
"""
import pandas as pd
import pytest

from pysharpe.data.fetcher import DuckDBCachedPriceFetcher, PriceFetcher


class MockFetcher(PriceFetcher):
    def __init__(self):
        self.call_count = 0

    def fetch_history(self, ticker, *, period, interval, start=None, end=None):
        self.call_count += 1
        # Create a dummy DataFrame
        dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [10, 11, 12],
                "High": [11, 12, 13],
                "Low": [9, 10, 11],
                "Close": [10.5, 11.5, 12.5],
                "Volume": [100, 200, 300],
                "Dividends": [0, 0, 0],
                "Stock Splits": [0, 0, 0],
            },
            index=dates,
        )
        df.index.name = "Date"
        return df


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_cache.db")


def test_duckdb_cache_miss_then_hit(temp_db_path):
    mock_fetcher = MockFetcher()
    cached_fetcher = DuckDBCachedPriceFetcher(
        fetcher=mock_fetcher, db_path=temp_db_path
    )

    # First call: cache miss
    df1 = cached_fetcher.fetch_history("AAPL", period="1mo", interval="1d")
    assert mock_fetcher.call_count == 1
    assert len(df1) == 3
    assert df1.index.name == "Date"
    assert "Close" in df1.columns

    # Second call with same params: cache hit
    df2 = cached_fetcher.fetch_history("AAPL", period="1mo", interval="1d")
    assert mock_fetcher.call_count == 1  # Should not increment
    assert len(df2) == 3
    pd.testing.assert_series_equal(df1["Close"], df2["Close"])


def test_duckdb_cache_different_params(temp_db_path):
    mock_fetcher = MockFetcher()
    cached_fetcher = DuckDBCachedPriceFetcher(
        fetcher=mock_fetcher, db_path=temp_db_path
    )

    # First call
    cached_fetcher.fetch_history("AAPL", period="1mo", interval="1d")
    assert mock_fetcher.call_count == 1

    # Call with different period
    cached_fetcher.fetch_history("AAPL", period="3mo", interval="1d")
    assert mock_fetcher.call_count == 2

    # Call with different ticker
    cached_fetcher.fetch_history("MSFT", period="1mo", interval="1d")
    assert mock_fetcher.call_count == 3
