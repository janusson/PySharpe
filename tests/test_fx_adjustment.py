"""Tests for FX adjustment logic in portfolio optimization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from pysharpe.data.fetcher import apply_fx_conversion


def test_apply_fx_conversion_no_conversion_needed(monkeypatch: pytest.MonkeyPatch):
    """Test that no conversion is applied when assets are already in the base currency."""
    import yfinance as yf

    mock_ticker = MagicMock()
    mock_ticker.info = {"currency": "CAD"}
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    prices = pd.DataFrame(
        {"AAA": [100.0, 101.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    adjusted = apply_fx_conversion(prices, base_currency="CAD")

    pd.testing.assert_frame_equal(adjusted, prices)


def test_apply_fx_conversion_with_usd_to_cad(monkeypatch: pytest.MonkeyPatch):
    """Test successful conversion from USD to CAD."""
    import yfinance as yf

    def mock_ticker_init(ticker: str):
        m = MagicMock()
        if ticker == "USD_ASSET":
            m.info = {"currency": "USD"}
        else:
            m.info = {"currency": "CAD"}
        return m

    monkeypatch.setattr(yf, "Ticker", mock_ticker_init)

    # Mock YFinancePriceFetcher
    mock_fetcher = MagicMock()
    # FX rates: 1.35 on Jan 1, 1.36 on Jan 2
    fx_data = pd.DataFrame(
        {"Close": [1.35, 1.36]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )
    mock_fetcher.fetch_history.return_value = fx_data
    monkeypatch.setattr(
        "pysharpe.data.fetcher.YFinancePriceFetcher", lambda: mock_fetcher
    )

    prices = pd.DataFrame(
        {"USD_ASSET": [100.0, 100.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    adjusted = apply_fx_conversion(prices, base_currency="CAD")

    expected = pd.DataFrame(
        {"USD_ASSET": [135.0, 136.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    pd.testing.assert_frame_equal(adjusted, expected)


def test_apply_fx_conversion_fails_on_empty_fx(monkeypatch: pytest.MonkeyPatch):
    """Test that ValueError is raised if FX data is empty."""
    import yfinance as yf

    mock_ticker = MagicMock()
    mock_ticker.info = {"currency": "USD"}
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_history.return_value = pd.DataFrame()
    monkeypatch.setattr(
        "pysharpe.data.fetcher.YFinancePriceFetcher", lambda: mock_fetcher
    )

    prices = pd.DataFrame(
        {"USD_ASSET": [100.0]},
        index=pd.to_datetime(["2023-01-01"]),
    )

    with pytest.raises(ValueError, match="FX data for USDCAD=X is empty"):
        apply_fx_conversion(prices, base_currency="CAD")


def test_apply_fx_conversion_fills_missing_fx_dates(monkeypatch: pytest.MonkeyPatch):
    """Test that missing FX dates are filled using ffill and bfill."""
    import yfinance as yf

    mock_ticker = MagicMock()
    mock_ticker.info = {"currency": "USD"}
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    mock_fetcher = MagicMock()
    # FX data only for Jan 2
    fx_data = pd.DataFrame(
        {"Close": [1.35]},
        index=pd.to_datetime(["2023-01-02"]),
    )
    mock_fetcher.fetch_history.return_value = fx_data
    monkeypatch.setattr(
        "pysharpe.data.fetcher.YFinancePriceFetcher", lambda: mock_fetcher
    )

    # Prices for Jan 1 and Jan 2
    prices = pd.DataFrame(
        {"USD_ASSET": [100.0, 100.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    # Jan 1 should be bfilled from Jan 2
    adjusted = apply_fx_conversion(prices, base_currency="CAD")

    expected = pd.DataFrame(
        {"USD_ASSET": [135.0, 135.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
    )

    pd.testing.assert_frame_equal(adjusted, expected)
