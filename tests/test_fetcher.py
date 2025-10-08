import builtins
from types import SimpleNamespace

import pandas as pd
import pytest

from pysharpe.data.fetcher import PriceHistoryError, YFinancePriceFetcher


def test_lazy_module_raises_when_yfinance_missing(monkeypatch: pytest.MonkeyPatch):
    fetcher = YFinancePriceFetcher()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(PriceHistoryError):
        fetcher._lazy_module()


def test_fetch_history_builds_payload(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, dict] = {}

    class DummyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs):
            calls["kwargs"] = kwargs
            return pd.DataFrame({"Close": [1.0]})

    module = SimpleNamespace(Ticker=lambda symbol: DummyTicker(symbol))

    fetcher = YFinancePriceFetcher({"auto_adjust": True})
    monkeypatch.setattr(fetcher, "_lazy_module", lambda: module)

    frame = fetcher.fetch_history("AAPL", period="1y", interval="1d", start=None, end=None)

    assert not frame.empty
    payload = calls["kwargs"]
    assert payload["interval"] == "1d"
    assert payload["period"] == "1y"
    assert payload["auto_adjust"] is True


def test_fetch_history_prefers_explicit_dates(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, dict] = {}

    class DummyTicker:
        def history(self, **kwargs):
            calls["kwargs"] = kwargs
            return pd.DataFrame({"Close": [1.0]})

    module = SimpleNamespace(Ticker=lambda symbol: DummyTicker())
    fetcher = YFinancePriceFetcher()
    monkeypatch.setattr(fetcher, "_lazy_module", lambda: module)

    fetcher.fetch_history("AAPL", period="1y", interval="1d", start="2024-01-01", end="2024-02-01")

    payload = calls["kwargs"]
    assert "period" not in payload
    assert payload["start"] == "2024-01-01"
    assert payload["end"] == "2024-02-01"


def test_fetch_history_raises_on_empty(monkeypatch: pytest.MonkeyPatch):
    class DummyTicker:
        def history(self, **kwargs):
            return pd.DataFrame()

    fetcher = YFinancePriceFetcher()
    monkeypatch.setattr(fetcher, "_lazy_module", lambda: SimpleNamespace(Ticker=lambda symbol: DummyTicker()))

    with pytest.raises(PriceHistoryError):
        fetcher.fetch_history("AAPL", period="1y", interval="1d", start=None, end=None)


def test_fetch_history_wraps_exceptions(monkeypatch: pytest.MonkeyPatch):
    class DummyTicker:
        def history(self, **kwargs):
            raise RuntimeError("boom")

    fetcher = YFinancePriceFetcher()
    monkeypatch.setattr(fetcher, "_lazy_module", lambda: SimpleNamespace(Ticker=lambda symbol: DummyTicker()))

    with pytest.raises(PriceHistoryError):
        fetcher.fetch_history("AAPL", period="1y", interval="1d", start=None, end=None)
