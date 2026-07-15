"""Tests for the DuckDB-based data linker.

.. note::

    **Canadian Data Pipeline** — DataLinker provides SQL-based joining
    of market and macro data for enhanced feature engineering.  Used
    for analytical workflows, not for predictive trading models.
"""

import numpy as np
import pandas as pd
import pytest

from pysharpe.data.fetcher import PriceFetcher
from pysharpe.data.linkage import DataLinker, HistoryLinker


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2023-01-01", periods=10)
    # 10, 20, 30, ..., 100
    prices = np.arange(10, 110, 10)
    return pd.DataFrame(
        {"date": dates, "price": prices, "volume": np.random.randint(100, 1000, 10)}
    )


@pytest.fixture
def sample_macro_data():
    dates = pd.date_range("2023-01-01", periods=10)
    # Assume interest rates are slowly rising
    interest_rates = np.linspace(2.0, 3.0, 10)
    return pd.DataFrame({"date": dates, "interest_rate": interest_rates})


def test_data_linker_initialization():
    linker = DataLinker()
    assert linker.conn is not None
    linker.close()


def test_register_and_execute(sample_market_data):
    linker = DataLinker()
    linker.register_data("my_table", sample_market_data)

    result = linker.execute_query("SELECT COUNT(*) as cnt FROM my_table")
    assert result["cnt"].iloc[0] == 10

    linker.close()


def test_register_invalid_data():
    linker = DataLinker()
    with pytest.raises(TypeError):
        linker.register_data("invalid", [1, 2, 3])
    linker.close()


def test_get_enhanced_market_data_no_macro(sample_market_data):
    linker = DataLinker()
    linker.register_data("market_data", sample_market_data)

    # 3-day rolling window
    result = linker.get_enhanced_market_data(
        market_table="market_data", rolling_window=3
    )

    # Check columns
    assert "price_rolling_avg" in result.columns
    assert "price_lag_1" in result.columns

    # Day 1: lag is NaN, rolling is 10
    assert pd.isna(result["price_lag_1"].iloc[0])
    assert result["price_rolling_avg"].iloc[0] == 10.0

    # Day 2: lag is 10, rolling is (10+20)/2 = 15
    assert result["price_lag_1"].iloc[1] == 10.0
    assert result["price_rolling_avg"].iloc[1] == 15.0

    # Day 3: lag is 20, rolling is (10+20+30)/3 = 20
    assert result["price_lag_1"].iloc[2] == 20.0
    assert result["price_rolling_avg"].iloc[2] == 20.0

    # Day 4: lag is 30, rolling is (20+30+40)/3 = 30
    assert result["price_rolling_avg"].iloc[3] == 30.0

    linker.close()


def test_get_enhanced_market_data_with_macro(sample_market_data, sample_macro_data):
    linker = DataLinker()
    linker.register_data("market_data", sample_market_data)
    linker.register_data("macro_data", sample_macro_data)

    result = linker.get_enhanced_market_data(
        market_table="market_data", macro_table="macro_data", rolling_window=3
    )

    # Check joined columns
    assert "interest_rate" in result.columns
    assert len(result) == 10
    assert result["interest_rate"].iloc[0] == 2.0
    assert result["interest_rate"].iloc[-1] == 3.0

    linker.close()


class _MockFetcherForStitch(PriceFetcher):
    def fetch_history(self, ticker, *, period, interval, start=None, end=None):
        dates = pd.date_range("2020-01-01", "2020-01-10", tz="UTC")
        if ticker == "TARGET":
            return pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=dates[4:]
            )
        elif ticker == "PROXY":
            return pd.DataFrame(
                {"Close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]},
                index=dates,
            )
        elif ticker == "USDCAD=X":
            return pd.DataFrame({"Close": [2.0] * 10}, index=dates)
        return pd.DataFrame()


def test_stitched_series_no_fx():
    proxy_map = {"TARGET": "PROXY"}
    linker = HistoryLinker(
        proxy_map=proxy_map, fx_adjust=False, fetcher=_MockFetcherForStitch()
    )
    stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")
    assert len(stitched) == 10
    assert stitched.name == "TARGET"
    assert pytest.approx(stitched.iloc[4]) == 100.0
    assert pytest.approx(stitched.iloc[-1]) == 105.0
    assert pytest.approx(stitched.iloc[0]) == 10.0 * (100.0 / 14.0)


def test_stitched_series_with_fx():
    proxy_map = {"TARGET": "PROXY"}
    linker = HistoryLinker(
        proxy_map=proxy_map, fx_adjust=True, fetcher=_MockFetcherForStitch()
    )
    stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")
    assert len(stitched) == 10
    assert pytest.approx(stitched.iloc[4]) == 100.0
    assert pytest.approx(stitched.iloc[-1]) == 105.0
    assert pytest.approx(stitched.iloc[0]) == (10.0 * (1.0 / 2.0)) * (100.0 / 7.0)


def test_stitched_series_no_proxy_defined():
    proxy_map = {"OTHER": "PROXY"}
    linker = HistoryLinker(
        proxy_map=proxy_map, fx_adjust=False, fetcher=_MockFetcherForStitch()
    )
    stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")
    assert len(stitched) == 6
    assert stitched.index.min() == pd.Timestamp("2020-01-05", tz="UTC")
