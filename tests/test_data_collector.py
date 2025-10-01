"""Tests for the simplified data collector module."""

from __future__ import annotations

import types

import pandas as pd

from pysharpe import data_collector


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

    instances: list[object] = []

    class _StubFetcher:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def fetch_history(
            self,
            ticker: str,
            *,
            period: str,
            interval: str,
            start: str | None,
            end: str | None,
        ):
            self.calls.append(
                {
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "start": start,
                    "end": end,
                }
            )
            return history

    def _factory() -> _StubFetcher:
        instance = _StubFetcher()
        instances.append(instance)
        return instance

    monkeypatch.setattr(data_collector, "YFinancePriceFetcher", _factory)

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
    assert (export_dir / "demo_metadata.json").exists()
    assert instances[0].calls[0]["period"] == "1y"
    assert instances[0].calls[0]["start"] is None

    data_collector.process_portfolio(
        portfolio_file,
        price_history_dir=price_dir,
        export_dir=export_dir,
        interval="1d",
        start="2023-01-01",
    )

    # second invocation instantiates a new fetcher
    assert instances[-1].calls[0]["start"] == "2023-01-01"
    assert instances[-1].calls[0]["period"] == "max"


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


def test_read_tickers_from_file_strips_comments_and_duplicates(tmp_path):
    portfolio_file = tmp_path / "demo.csv"
    portfolio_file.write_text("# comment\nAAPL\nMSFT\nAAPL\n", encoding="utf-8")

    tickers = data_collector.read_tickers_from_file(portfolio_file)
    assert tickers == {"AAPL", "MSFT"}


def test_portfolio_ticker_reader_refresh_handles_empty_files(tmp_path):
    portfolio_dir = tmp_path / "portfolio"
    portfolio_dir.mkdir()

    (portfolio_dir / "growth.csv").write_text("AAPL\nMSFT\n", encoding="utf-8")
    reader = data_collector.PortfolioTickerReader(portfolio_dir)
    assert reader.get_portfolio_tickers("growth") == {"AAPL", "MSFT"}

    (portfolio_dir / "income.csv").write_text("T\n# placeholder\n", encoding="utf-8")
    reader.refresh()
    assert reader.get_portfolio_tickers("income") == {"T"}

    (portfolio_dir / "empty.csv").write_text("# comment only\n", encoding="utf-8")
    reader.refresh()
    assert reader.get_portfolio_tickers("empty") == set()
    assert "empty" not in reader.portfolio_tickers

    # ensure previously loaded portfolios remain intact after refresh
    assert reader.get_portfolio_tickers("growth") == {"AAPL", "MSFT"}
