"""Focused tests for the CollationService helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pysharpe.data import CollationService, PriceFetcher


class _NoopFetcher(PriceFetcher):
    """Stub fetcher that should never be invoked in these tests."""

    def fetch_history(self, *args, **kwargs):  # type: ignore[override]
        raise AssertionError("fetch_history should not be called in collation-only tests")


def _write_price_history(path: Path, rows: list[tuple[str, float | None]]) -> None:
    frame = pd.DataFrame({"Date": [row[0] for row in rows], "Close": [row[1] for row in rows]})
    frame.to_csv(path, index=False)


def test_collate_portfolio_filters_invalid_columns(tmp_path):
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    price_dir.mkdir()
    export_dir.mkdir()

    _write_price_history(
        price_dir / "AAA_hist.csv",
        [
            ("2024-01-01", 10.0),
            ("2024-01-02", 11.0),
        ],
    )

    _write_price_history(
        price_dir / "BBB_hist.csv",
        [
            ("2024-01-01", None),
            ("2024-01-02", None),
        ],
    )

    service = CollationService(
        _NoopFetcher(),
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    frame = service.collate_portfolio("demo", ("AAA", "BBB"))

    assert list(frame.columns) == ["AAA"]
    metadata = json.loads((export_dir / "demo_metadata.json").read_text(encoding="utf-8"))
    assert metadata["included_tickers"] == ["AAA"]
    assert metadata["dropped_tickers"] == ["BBB"]
    # Change: Added regression test ensuring performance tweak keeps metadata consistent.


def test_collate_portfolio_returns_empty_when_no_data(tmp_path):
    service = CollationService(
        _NoopFetcher(),
        price_history_dir=tmp_path / "price_hist",
        export_dir=tmp_path / "exports",
    )

    frame = service.collate_portfolio("missing", ("AAA",))
    assert frame.empty


def test_download_portfolio_prices_writes_csv(tmp_path):
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    history = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Close": [1.0, 1.1]})

    class _Fetcher(PriceFetcher):
        def fetch_history(self, *_args, **_kwargs):  # type: ignore[override]
            return history

    service = CollationService(
        _Fetcher(),
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    result = service.download_portfolio_prices(["AAA"], period="1mo", interval="1d", start=None, end=None)
    assert "AAA" in result
    assert (price_dir / "AAA_hist.csv").exists()
