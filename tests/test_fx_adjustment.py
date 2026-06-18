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


def test_apply_fx_conversion_excludes_rows_with_no_fx_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """Rows where FX data is not yet available must be dropped, not bfilled.

    Using a future rate to convert a historical price (bfill) is lookahead bias.
    The correct fix is to trim the price history to start when FX data exists.
    """
    import yfinance as yf

    mock_ticker = MagicMock()
    mock_ticker.info = {"currency": "USD"}
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    mock_fetcher = MagicMock()
    # FX data only available from Jan 2 — Jan 1 is genuinely unknown
    fx_data = pd.DataFrame(
        {"Close": [1.35]},
        index=pd.to_datetime(["2023-01-02"]),
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

    # Jan 1 must be excluded — its FX rate was unknown; Jan 2 is converted correctly
    assert pd.Timestamp("2023-01-01") not in adjusted.index, (
        "Jan 1 should be excluded because FX data was not yet available (bfill is lookahead bias)."
    )
    assert pd.Timestamp("2023-01-02") in adjusted.index
    assert adjusted.loc[pd.Timestamp("2023-01-02"), "USD_ASSET"] == pytest.approx(135.0)


def test_apply_fx_conversion_trims_to_common_fx_window(monkeypatch: pytest.MonkeyPatch):
    """When two foreign tickers have different FX start dates, the output is trimmed
    to the latest common start — ensuring no row uses a bfilled rate for any ticker."""
    import yfinance as yf

    def mock_ticker_init(ticker: str):
        m = MagicMock()
        m.info = {"currency": "USD"}  # both assets are USD
        return m

    monkeypatch.setattr(yf, "Ticker", mock_ticker_init)

    # Two sequential FX fetches for the same USDCAD=X ticker but with different data windows
    call_count = {"n": 0}

    def mock_fetch(ticker, *, period, interval, start, end):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (for USD_A): FX available Jan 2 and Jan 3
            return pd.DataFrame(
                {"Close": [1.35, 1.36]},
                index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
            )
        else:
            # Second call (for USD_B): FX only available Jan 3
            return pd.DataFrame(
                {"Close": [1.36]},
                index=pd.to_datetime(["2023-01-03"]),
            )

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_history.side_effect = mock_fetch
    monkeypatch.setattr(
        "pysharpe.data.fetcher.YFinancePriceFetcher", lambda: mock_fetcher
    )

    prices = pd.DataFrame(
        {
            "USD_A": [100.0, 100.0, 100.0],
            "USD_B": [200.0, 200.0, 200.0],
        },
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )

    adjusted = apply_fx_conversion(prices, base_currency="CAD")

    # Only Jan 3 is valid for both tickers — Jan 1 and Jan 2 must be excluded
    assert list(adjusted.index) == [pd.Timestamp("2023-01-03")], (
        f"Expected only Jan 3 in output, got {list(adjusted.index)}"
    )
    assert adjusted.loc[pd.Timestamp("2023-01-03"), "USD_A"] == pytest.approx(136.0)
    assert adjusted.loc[pd.Timestamp("2023-01-03"), "USD_B"] == pytest.approx(272.0)
