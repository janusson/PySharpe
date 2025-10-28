"""Focused tests for the CollationService helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandas.testing as tm

from pysharpe.data import CollationService, PriceFetcher
from pysharpe.data.collation import load_raw, parse_records
from pysharpe.data.fetcher import PriceHistoryError


class _NoopFetcher(PriceFetcher):
    """Stub fetcher that should never be invoked in these tests."""

    def fetch_history(self, *args, **kwargs):  # type: ignore[override]
        raise AssertionError("fetch_history should not be called in collation-only tests")


def _write_price_history(path: Path, rows: list[tuple[str, float | None]]) -> None:
    frame = pd.DataFrame({"Date": [row[0] for row in rows], "Close": [row[1] for row in rows]})
    frame.to_csv(path, index=False)


def _golden_path(name: str) -> Path:
    return Path(__file__).parent / "golden" / name


def test_load_raw_matches_golden_fixture():
    path = _golden_path("aaa_history_raw.csv")
    frame = load_raw(path)
    expected = pd.read_csv(path)
    tm.assert_frame_equal(frame, expected)


def test_parse_records_matches_golden_fixture():
    raw = load_raw(_golden_path("aaa_history_raw.csv"))
    parsed = parse_records(raw, "AAA")
    expected = pd.read_csv(_golden_path("aaa_history_parsed.csv")).set_index("Date")
    expected.index = expected.index.astype(str)
    expected.index.name = "Date"
    tm.assert_frame_equal(parsed, expected)


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


def test_download_portfolio_prices_skips_errors(tmp_path):
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    class _Fetcher(PriceFetcher):
        def fetch_history(self, ticker, *_args, **_kwargs):  # type: ignore[override]
            if ticker == "BAD":
                raise PriceHistoryError("failed")
            return pd.DataFrame({"Date": ["2024-01-01"], "Close": [1.0]})

    service = CollationService(
        _Fetcher(),
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    result = service.download_portfolio_prices(["BAD", "GOOD"], period="1mo", interval="1d", start=None, end=None)
    assert "GOOD" in result and "BAD" not in result
    assert (price_dir / "GOOD_hist.csv").exists()


def test_load_price_frame_missing_file_logs_warning(tmp_path, caplog):
    service = CollationService(
        _NoopFetcher(),
        price_history_dir=tmp_path / "price_hist",
        export_dir=tmp_path / "exports",
    )

    with caplog.at_level("WARNING"):
        result = service._load_price_frame("AAA")

    assert result is None
    assert "Price history missing" in caplog.text


def test_load_price_frame_invalid_columns(tmp_path, caplog):
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"
    price_dir.mkdir()
    export_dir.mkdir()

    pd.DataFrame({"Date": ["2024-01-01"], "Value": [1.0]}).to_csv(price_dir / "AAA_hist.csv", index=False)

    service = CollationService(
        _NoopFetcher(),
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    with caplog.at_level("ERROR"):
        result = service._load_price_frame("AAA")

    assert result is None
    assert "Unexpected columns" in caplog.text


def test_load_price_frame_invalid_dates(tmp_path, caplog):
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"
    price_dir.mkdir()
    export_dir.mkdir()

    pd.DataFrame({"Date": ["not-a-date"], "Close": [1.0]}).to_csv(price_dir / "AAA_hist.csv", index=False)

    service = CollationService(
        _NoopFetcher(),
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    with caplog.at_level("ERROR"):
        result = service._load_price_frame("AAA")

    assert result is None
    assert "Unable to parse dates" in caplog.text


def test_load_price_frame_handles_timezone_and_empty(tmp_path, caplog):
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"
    price_dir.mkdir()
    export_dir.mkdir()

    pd.DataFrame(
        {
            "Date": ["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
            "Close": [1.0, None],
        }
    ).to_csv(price_dir / "AAA_hist.csv", index=False)

    service = CollationService(
        _NoopFetcher(),
        price_history_dir=price_dir,
        export_dir=export_dir,
    )

    frame = service._load_price_frame("AAA")

    assert list(frame.columns) == ["AAA"]
    assert list(frame.index) == ["2024-01-01"]
