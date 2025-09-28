"""Tests for the data.fetch module."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from pysharpe.data.fetch import fetch_price_history


def _patch_yfinance(monkeypatch, frame: pd.DataFrame) -> None:
    """Patch the ``yfinance`` module to return *frame*."""

    def _download(*args, **kwargs):  # noqa: D401 - simple stub for monkeypatch
        return frame

    fake_module = types.SimpleNamespace(download=_download)
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)


def test_fetch_price_history_single_symbol(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame(
        {
            "Adj Close": [101.0, 102.5, 103.0],
            "Open": [100.0, 101.0, 102.0],
        },
        index=dates,
    )

    _patch_yfinance(monkeypatch, data)

    frame = fetch_price_history(["AAPL"])

    assert list(frame.columns) == ["AAPL"]
    pd.testing.assert_index_equal(frame.index, dates)
    pd.testing.assert_series_equal(frame["AAPL"], data["Adj Close"], check_names=False)


def test_fetch_price_history_multiple_symbols(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], ["AAPL", "MSFT"]]
    )
    data = pd.DataFrame(
        [[101.0, 210.0, 100.0, 209.0], [102.0, 211.0, 101.0, 210.0]],
        index=dates,
        columns=columns,
    )

    _patch_yfinance(monkeypatch, data)

    frame = fetch_price_history(["AAPL", "MSFT"])

    assert list(frame.columns) == ["AAPL", "MSFT"]
    pd.testing.assert_frame_equal(
        frame,
        pd.DataFrame(
            {
                "AAPL": [101.0, 102.0],
                "MSFT": [210.0, 211.0],
            },
            index=dates,
        ),
    )


def test_fetch_price_history_generator_input(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    data = pd.DataFrame({"Adj Close": [50.0]}, index=dates)

    _patch_yfinance(monkeypatch, data)

    result = fetch_price_history((symbol for symbol in ["TSLA"]))
    assert list(result.columns) == ["TSLA"]


def test_fetch_price_history_missing_adj_close(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    data = pd.DataFrame({"Close": [50.0]}, index=dates)

    _patch_yfinance(monkeypatch, data)

    with pytest.raises(ValueError, match="Adj Close"):
        fetch_price_history(["TSLA"])
