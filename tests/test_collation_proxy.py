import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from pysharpe.config import PySharpeSettings
from pysharpe.data.collation import CollationService
from pysharpe.data.fetcher import PriceFetcher
from pysharpe.data.linkage import HistoryLinker


class MockFetcher(PriceFetcher):
    def fetch_history(self, ticker, *, period, interval, start=None, end=None):
        dates = pd.date_range("2020-01-01", "2020-01-10", tz="UTC")
        if ticker == "TARGET":
            df = pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=dates[4:]
            )
        elif ticker == "PROXY":
            df = pd.DataFrame(
                {"Close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]},
                index=dates,
            )
        elif ticker == "USDCAD=X":
            df = pd.DataFrame({"Close": [2.0] * 10}, index=dates)
        else:
            return pd.DataFrame()

        df.index.name = "Date"
        return df


def test_download_portfolios_uses_history_linker(tmp_path):
    proxy_map = {
        "TARGET": {"proxy": "PROXY", "fx_adjust": True, "start_date": "2020-01-01"}
    }
    settings = PySharpeSettings(data_dir=tmp_path, proxy_map=proxy_map)

    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    service = CollationService(
        MockFetcher(),
        settings=settings,
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    results = service.download_portfolio_prices(
        ["TARGET"], period="max", interval="1d", start=None, end=None
    )

    assert "TARGET" in results
    df = results["TARGET"]

    assert len(df) == 10
    # Day 0 value: Proxy 10.0 * (1/2.0) = 5.0 CAD. Scalar = (100.0 / 7.0) = 14.2857
    # 5.0 * 14.2857 = 71.4285
    assert pytest.approx(df["Close"].iloc[0]) == (10.0 * 0.5) * (100.0 / 7.0)
    assert pytest.approx(df["Close"].iloc[-1]) == 105.0
