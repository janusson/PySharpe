"""Tests for the simplified data collector module."""

from __future__ import annotations

import types

import pandas as pd

from pysharpe import data_collector


def _fake_yfinance(history_frame: pd.DataFrame):
    calls: list[dict[str, object]] = []

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs):  # noqa: D401 - stub method signature
            calls.append({"symbol": self.symbol, "kwargs": kwargs})
            return history_frame

    return types.SimpleNamespace(Ticker=_Ticker, calls=calls)


def test_get_csv_file_paths(tmp_path):
    (tmp_path / "alpha.csv").write_text("AAPL", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("IGNORE", encoding="utf-8")
    (tmp_path / "gamma.csv").write_text("MSFT", encoding="utf-8")

    files = data_collector.get_csv_file_paths(tmp_path)
    assert [path.name for path in files] == ["alpha.csv", "gamma.csv"]


def test_download_and_collate_prices(monkeypatch, tmp_path):
    history = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [100.0, 101.0],
        }
    )

    fake_yf = _fake_yfinance(history)
    monkeypatch.setattr(data_collector, "yf", fake_yf)

    portfolio_file = tmp_path / "demo.csv"
    portfolio_file.write_text("AAA\n", encoding="utf-8")

    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    frame = data_collector.process_portfolio(
        portfolio_file,
        price_history_dir=price_dir,
        export_dir=export_dir,
        period="1y",
        interval="1d",
    )

    assert "AAA" in frame.columns
    assert (price_dir / "AAA_hist.csv").exists()
    assert (export_dir / "demo_collated.csv").exists()
    assert fake_yf.calls[0]["kwargs"]["period"] == "1y"
    assert "start" not in fake_yf.calls[0]["kwargs"]

    fake_yf.calls.clear()

    data_collector.process_portfolio(
        portfolio_file,
        price_history_dir=price_dir,
        export_dir=export_dir,
        interval="1d",
        start="2023-01-01",
    )

    assert fake_yf.calls[0]["kwargs"]["start"] == "2023-01-01"
    assert "period" not in fake_yf.calls[0]["kwargs"]


def test_security_data_collector_download(monkeypatch, tmp_path):
    info_payload = {"shortName": "Test Corp", "longName": "Test Corporation"}
    news_payload = ["story"]

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.info = info_payload
            self.news = news_payload
            self.options = ["2024-06-21"]

        def history(self, **kwargs):  # noqa: D401 - stub
            return pd.DataFrame({"Close": [1.0]})

        def get_shares_full(self):  # noqa: D401 - stub
            return pd.DataFrame({"Shares": [100]})

    monkeypatch.setattr(data_collector, "yf", types.SimpleNamespace(Ticker=_FakeTicker))

    collector = data_collector.SecurityDataCollector("TEST")
    assert collector.get_company_name() == "Test Corp"
    output = collector.download_info(tmp_path)
    assert output.exists()

    price_frame = collector.download_price_history(tmp_path)
    assert not price_frame.empty
