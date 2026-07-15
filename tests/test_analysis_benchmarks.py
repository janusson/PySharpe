"""Tests for the benchmarks analysis module.

.. note::

    **Canadian ETF Context** — Benchmarks use CAD-denominated broad-market
    ETFs (e.g., VEQT.TO).  Returns are annualized at 252 trading days.
    Foreign withholding tax drag is accounted for in the tax-location
    engine, not in benchmark fetching.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from pysharpe.analysis.benchmarks import fetch_benchmark_metrics


def test_fetch_benchmark_metrics_empty():
    """Test with empty ticker list."""
    df = fetch_benchmark_metrics([], "2023-01-01", "2023-12-31")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "Ticker",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
    ]
    assert len(df) == 0


def test_fetch_benchmark_metrics_mocked(monkeypatch: pytest.MonkeyPatch):
    """Test with mocked price data."""

    # Mock DuckDBCachedPriceFetcher
    mock_fetcher = MagicMock()

    # Mock data for VEQT.TO
    veqt_data = pd.DataFrame(
        {"Close": [10.0, 10.1, 10.2]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )
    mock_fetcher.fetch_history.return_value = veqt_data

    # Mock apply_fx_conversion to return data as-is
    monkeypatch.setattr(
        "pysharpe.analysis.benchmarks.apply_fx_conversion", lambda df, **kwargs: df
    )

    # Mock DuckDBCachedPriceFetcher constructor
    monkeypatch.setattr(
        "pysharpe.analysis.benchmarks.DuckDBCachedPriceFetcher", lambda _: mock_fetcher
    )

    df = fetch_benchmark_metrics(["VEQT.TO"], "2023-01-01", "2023-01-03")

    assert len(df) == 1
    assert df.iloc[0]["Ticker"] == "VEQT.TO"
    assert "Annualized Return" in df.columns
    assert "Annualized Volatility" in df.columns
    assert "Sharpe Ratio" in df.columns
    assert df.iloc[0]["Annualized Return"] > 0
