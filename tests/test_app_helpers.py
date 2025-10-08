import app
import pandas as pd
import pytest
from pysharpe.optimization import PortfolioWeights

from app import (
    MetricResults,
    _clean_numeric_frame,
    _prepare_weight_chart_data,
    _resolve_field_frame,
    _select_price_data,
    compute_metrics,
)


def call_cached(func, *args, **kwargs):
    target = getattr(func, "__wrapped__", func)
    return target(*args, **kwargs)


def test_resolve_field_frame_multiindex_extracts_preferred_block():
    raw = pd.DataFrame(
        data=[
            [100.0, 200.0, 1_000_000, 2_000_000],
            [101.0, 201.0, 1_050_000, 2_050_000],
        ],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        columns=pd.MultiIndex.from_product(
            [["Adj Close", "Volume"], ["AAPL", "MSFT"]],
        ),
    )

    extracted = _resolve_field_frame(raw, ("Adj Close", "Close"))

    assert list(extracted.columns) == ["AAPL", "MSFT"]
    pd.testing.assert_index_equal(extracted.index, raw.index)


def test_resolve_field_frame_single_level_matches_substring():
    raw = pd.DataFrame(
        data=[
            [100.0, 1_000_000],
            [99.0, 980_000],
        ],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        columns=["AAPL Close", "AAPL Volume"],
    )

    extracted = _resolve_field_frame(raw, ("Adj Close", "Close"))

    assert list(extracted.columns) == ["AAPL Close"]


def test_resolve_field_frame_handles_empty_input():
    result = _resolve_field_frame(pd.DataFrame(), ("Adj Close", "Close"))

    assert result.empty


def test_resolve_field_frame_returns_numeric_when_no_match():
    raw = pd.DataFrame(
        data=[
            [100.0, 101.0],
            [102.0, 103.0],
        ],
        columns=["Open", "High"],
    )

    extracted = _resolve_field_frame(raw, ("Adj Close", "Close"))

    assert list(extracted.columns) == ["Open", "High"]


def test_clean_numeric_frame_deduplicates_and_fills():
    frame = pd.DataFrame(
        data=[
            [100.0, None],
            [None, 102.0],
        ],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        columns=["AAPL", "AAPL"],
    )

    cleaned = _clean_numeric_frame(frame)

    assert list(cleaned.columns) == ["AAPL", "AAPL.1"]
    assert cleaned.isna().sum().sum() == 0


def test_clean_numeric_frame_returns_empty_input():
    frame = pd.DataFrame()

    result = _clean_numeric_frame(frame)

    assert result.empty and result is frame


def test_select_price_data_prefers_close_columns():
    numeric_df = pd.DataFrame(
        data=[
            [100.0, 200.0, 500.0],
            [101.0, 210.0, 505.0],
        ],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        columns=["AAPL Close", "AAPL Volume", "Indicator"],
    )

    price_data = _select_price_data(numeric_df)

    assert list(price_data.columns) == ["AAPL Close"]


def test_select_price_data_falls_back_to_non_volume_columns():
    numeric_df = pd.DataFrame(
        data=[
            [100.0, 200.0],
            [101.0, 205.0],
        ],
        index=pd.date_range("2023-01-01", periods=2, freq="D"),
        columns=["Momentum", "Volume"],
    )

    price_data = _select_price_data(numeric_df)

    assert list(price_data.columns) == ["Momentum"]


def test_prepare_weight_chart_data_filters_non_positive_weights():
    series = pd.Series({"AAPL": 0.6, "MSFT": 0.4, "CASH": 0.0, "BOND": -0.1})

    chart_df = _prepare_weight_chart_data(series)

    assert list(chart_df["Ticker"]) == ["AAPL", "MSFT"]
    assert chart_df["Weight"].tolist() == [0.6, 0.4]


def test_compute_metrics_reindexes_outputs():
    price_frame = pd.DataFrame(
        data=[
            [100.0, 200.0],
            [102.0, 202.0],
            [103.0, 203.0],
        ],
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
        columns=["MSFT", "AAPL"],
    )

    results = compute_metrics(price_frame)

    assert isinstance(results, MetricResults)
    assert list(results.expected.index) == ["MSFT", "AAPL"]
    assert list(results.volatility.index) == ["MSFT", "AAPL"]
    assert list(results.sharpe.index) == ["MSFT", "AAPL"]


@pytest.mark.parametrize(
    "price_frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"AAPL": [100.0]}, index=pd.date_range("2023-01-01", periods=1)),
    ],
)
def test_compute_metrics_rejects_insufficient_history(price_frame):
    with pytest.raises(ValueError):
        compute_metrics(price_frame)


def test_load_prices_extracts_adj_close(monkeypatch: pytest.MonkeyPatch):
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern")
    data = pd.DataFrame(
        data=[
            [100.0, 200.0, 1_000_000, 2_000_000],
            [101.0, 201.0, 1_050_000, 2_050_000],
        ],
        columns=["Adj Close", "Volume", "Adj Close.1", "Volume.1"],
        index=dates,
    )
    monkeypatch.setattr(
        app._STREAMLIT_SERVICE,
        "download_portfolio_prices",
        lambda tickers, **kwargs: {"AAPL": data.iloc[:, [0, 1]], "MSFT": data.iloc[:, [2, 3]]},
    )
    portfolio_name = app._make_portfolio_name(("AAPL", "MSFT"))
    collated_path = app._STREAMLIT_SERVICE.export_dir / f"{portfolio_name}_collated.csv"
    collated_path.parent.mkdir(parents=True, exist_ok=True)

    def fake_collate(name, tickers):  # noqa: D401 - test helper
        frame = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0],
                "MSFT": [200.0, 201.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        frame.to_csv(collated_path)
        return frame

    monkeypatch.setattr(app._STREAMLIT_SERVICE, "collate_portfolio", fake_collate)
    monkeypatch.setattr(app, "_load_collated_from_disk", lambda path: None)

    result = call_cached(app.load_prices, ["AAPL", "MSFT"], "2024-01-01", "2024-02-01")

    assert isinstance(result, app.PortfolioData)
    assert list(result.prices.columns) == ["AAPL", "MSFT"]
    assert result.prices.notna().all().all()
    assert result.prices.index.tz is None
    assert result.collated.shape[1] == 2
    assert result.warnings == ()
    assert result.used_cache is False


def test_load_prices_returns_empty_when_no_numeric(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app, "_load_collated_from_disk", lambda path: None)
    monkeypatch.setattr(app._STREAMLIT_SERVICE, "download_portfolio_prices", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError):
        call_cached(app.load_prices, ["AAPL"], "2024-01-01", "2024-02-01")


def test_gather_metadata_handles_success(monkeypatch: pytest.MonkeyPatch):
    class DummyTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {"shortName": f"{symbol} Inc", "exchange": "NASDAQ", "currency": "USD"}

    monkeypatch.setattr(app.yf, "Ticker", lambda ticker: DummyTicker(ticker))

    metadata = call_cached(app.gather_metadata, ["AAPL"])

    assert metadata.loc["AAPL", "name"] == "AAPL Inc"


def test_gather_metadata_handles_failure(monkeypatch: pytest.MonkeyPatch):
    class DummyTicker:
        def __init__(self, symbol: str) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(app.yf, "Ticker", lambda ticker: DummyTicker(ticker))

    metadata = call_cached(app.gather_metadata, ["AAPL"])

    assert metadata.loc["AAPL", "name"] == "Lookup failed"


def test_load_preview_data_combines_volume(monkeypatch: pytest.MonkeyPatch):
    columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], ["AAPL", "MSFT"]])
    data = pd.DataFrame(
        data=[
            [100.0, 200.0, 1_000_000, 2_000_000],
            [101.0, 201.0, 1_050_000, 2_050_000],
        ],
        columns=columns,
    )
    monkeypatch.setattr(app.yf, "download", lambda *args, **kwargs: data)

    preview = call_cached(app.load_preview_data, ["AAPL", "MSFT"], pd.Timestamp("2024-03-01"))

    assert any(col.endswith("Volume") for col in preview.columns)


def test_load_preview_data_returns_empty_when_download_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    preview = call_cached(app.load_preview_data, ["AAPL"], pd.Timestamp("2024-03-01"))

    assert preview.empty


def test_optimise_weights_returns_none_for_empty_returns():
    metrics_result = MetricResults(
        returns=pd.DataFrame(),
        expected=pd.Series(dtype=float),
        volatility=pd.Series(dtype=float),
        sharpe=pd.Series(dtype=float),
    )

    assert app.optimise_weights(metrics_result) is None


def test_optimise_weights_invokes_frontier(monkeypatch: pytest.MonkeyPatch):
    metrics_result = MetricResults(
        returns=pd.DataFrame(
            {
                "AAPL": [0.01, 0.02],
                "MSFT": [0.015, -0.01],
            }
        ),
        expected=pd.Series({"AAPL": 0.1, "MSFT": 0.2}),
        volatility=pd.Series({"AAPL": 0.15, "MSFT": 0.2}),
        sharpe=pd.Series({"AAPL": 0.6, "MSFT": 0.7}),
    )

    class DummyFrontier:
        def __init__(self, mu, cov_matrix) -> None:
            self.mu = mu
            self.cov_matrix = cov_matrix

        def max_sharpe(self):
            return {"AAPL": 0.5, "MSFT": 0.5}

        def clean_weights(self):
            return {"AAPL": 0.6, "MSFT": 0.4}

    monkeypatch.setattr(app, "EfficientFrontier", DummyFrontier)

    weights = app.optimise_weights(metrics_result)

    assert isinstance(weights, PortfolioWeights)
    assert pytest.approx(weights.allocations["AAPL"]) == 0.6
