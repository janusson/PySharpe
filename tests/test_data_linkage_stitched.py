import pandas as pd
import pytest

from pysharpe.data.fetcher import PriceFetcher
from pysharpe.data.linkage import HistoryLinker


class MockFetcher(PriceFetcher):
    def fetch_history(self, ticker, *, period, interval, start=None, end=None):
        dates = pd.date_range("2020-01-01", "2020-01-10", tz="UTC")
        if ticker == "TARGET":
            # Target only exists from day 5 onwards
            return pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=dates[4:]
            )
        elif ticker == "PROXY":
            # Proxy exists for all 10 days
            return pd.DataFrame(
                {"Close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]},
                index=dates,
            )
        elif ticker == "USDCAD=X":
            # FX rate is 2.0 flat
            return pd.DataFrame({"Close": [2.0] * 10}, index=dates)
        return pd.DataFrame()


def test_stitched_series_no_fx():
    proxy_map = {"TARGET": "PROXY"}
    linker = HistoryLinker(proxy_map=proxy_map, fx_adjust=False, fetcher=MockFetcher())

    stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

    # Handover date is 2020-01-05
    # Target price at T0 = 100.0
    # Proxy price at T0 = 14.0
    # Scalar = 100.0 / 14.0

    assert len(stitched) == 10
    assert stitched.name == "TARGET"
    assert pytest.approx(stitched.iloc[4]) == 100.0
    assert pytest.approx(stitched.iloc[-1]) == 105.0
    assert pytest.approx(stitched.iloc[0]) == 10.0 * (100.0 / 14.0)


def test_stitched_series_with_fx():
    proxy_map = {"TARGET": "PROXY"}
    linker = HistoryLinker(proxy_map=proxy_map, fx_adjust=True, fetcher=MockFetcher())

    stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

    # Handover date is 2020-01-05
    # Target price at T0 = 100.0
    # FX adjusted Proxy price at T0 = 14.0 * (1.0 / 2.0) = 7.0
    # Scalar = 100.0 / 7.0

    assert len(stitched) == 10
    assert pytest.approx(stitched.iloc[4]) == 100.0
    assert pytest.approx(stitched.iloc[-1]) == 105.0
    assert pytest.approx(stitched.iloc[0]) == (10.0 * (1.0 / 2.0)) * (100.0 / 7.0)


def test_stitched_series_no_proxy_defined():
    proxy_map = {"OTHER": "PROXY"}
    linker = HistoryLinker(proxy_map=proxy_map, fx_adjust=False, fetcher=MockFetcher())

    stitched = linker.get_stitched_series("TARGET", start_date="2020-01-01")

    # Should only return the 6 days of the target
    assert len(stitched) == 6
    assert stitched.index.min() == pd.Timestamp("2020-01-05", tz="UTC")
