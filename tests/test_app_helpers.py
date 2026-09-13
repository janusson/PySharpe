"""Tests for the Streamlit app helpers.

.. note::

    **Canadian TFSA Context** — The Streamlit dashboard displays CAD-
    denominated ETF portfolio analytics.  No single-stock ticker
    recommendations are generated.  All price data is synthetic in tests.
"""

import numpy as np
import pandas as pd
import pytest
from pypfopt.risk_models import CovarianceShrinkage

import app
from app import (
    MetricResults,
    _clean_numeric_frame,
    _normalize_weights,
    _prepare_weight_chart_data,
    _resolve_field_frame,
    _select_price_data,
    compute_metrics,
    compute_overlapping_date_range,
    equal_weight_allocation,
)
from pysharpe.app.analytics import (
    AdjustedMetricResults,
    compute_adjusted_metrics,
    evaluate_adjusted_performance,
)
from pysharpe.optimization.expected_returns import shrinkage_expected_return
from pysharpe.optimization.tax_location import (
    AssetLocationEngine,
    AssetTaxCharacteristics,
    TaxProfile,
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


def test_equal_weight_allocation_is_strict_one_over_n():
    weights = equal_weight_allocation(["VFV.TO", "VCN.TO", "VDY.TO", "QQC.TO"])

    assert weights == pytest.approx(
        {
            "VFV.TO": 0.25,
            "VCN.TO": 0.25,
            "VDY.TO": 0.25,
            "QQC.TO": 0.25,
        }
    )
    assert sum(weights.values()) == pytest.approx(1.0)


def test_equal_weight_allocation_handles_empty_and_single():
    assert equal_weight_allocation([]) == {}
    assert equal_weight_allocation(["VCN.TO"]) == {"VCN.TO": 1.0}


def test_normalize_weights_rescales_to_unity():
    weights = _normalize_weights({"A": 0.6, "B": 0.4, "C": 0.2})

    assert weights == pytest.approx({"A": 0.5, "B": 1 / 3, "C": 1 / 6})


def test_normalize_weights_falls_back_to_equal_on_zero_sum():
    weights = _normalize_weights({"A": 0.0, "B": 0.0, "C": 0.0})

    assert weights == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def test_compute_overlapping_date_range_takes_max_min_window():
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0],
            "BBB": [np.nan, 200.0, 201.0, np.nan],
            "CCC": [300.0, 301.0, 302.0, 303.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )

    start, end = compute_overlapping_date_range(prices)

    assert start == pd.Timestamp("2024-01-02")
    assert end == pd.Timestamp("2024-01-03")


def test_compute_overlapping_date_range_rejects_empty_or_disjoint():
    empty = pd.DataFrame()
    assert compute_overlapping_date_range(empty) is None

    all_nan = pd.DataFrame(
        {"AAA": [np.nan, np.nan]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    assert compute_overlapping_date_range(all_nan) is None

    disjoint = pd.DataFrame(
        {
            "AAA": [1.0, 1.0, np.nan, np.nan],
            "BBB": [np.nan, np.nan, 2.0, 2.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    assert compute_overlapping_date_range(disjoint) is None


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


def _seeded_price_frame(n: int = 40, seed: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "VFV.TO": 100 * (1 + rng.normal(0.0010, 0.012, n)).cumprod(),
            "VCN.TO": 100 * (1 + rng.normal(0.0006, 0.009, n)).cumprod(),
            "QQC.TO": 100 * (1 + rng.normal(0.0013, 0.015, n)).cumprod(),
        },
        index=idx,
    )


def test_compute_adjusted_metrics_applies_shrinkage_and_tax_drag():
    """Harmonized metrics use Bayes-Stein shrunk returns reduced by MER and
    the AssetLocationEngine drag, with Sharpe recomputed at the 2% rf."""
    prices = _seeded_price_frame()
    profile = TaxProfile(marginal_tax_rate=0.45)
    chars = {
        "VFV.TO": AssetTaxCharacteristics(
            "VFV.TO", dividend_yield=0.015, is_cad_wrapped_us_equity=True, mer=0.0009
        ),
        "VCN.TO": AssetTaxCharacteristics("VCN.TO", dividend_yield=0.025, mer=0.0005),
        "QQC.TO": AssetTaxCharacteristics(
            "QQC.TO", dividend_yield=0.012, is_cad_wrapped_us_equity=True, mer=0.0020
        ),
    }

    results = compute_adjusted_metrics(
        prices, tax_profile=profile, asset_characteristics=chars
    )

    assert isinstance(results, AdjustedMetricResults)
    assert list(results.expected.index) == ["VFV.TO", "VCN.TO", "QQC.TO"]
    # Bayes-Stein shrinkage compresses the cross-section vs raw means.
    raw = prices.pct_change().dropna().mean() * 252
    assert results.expected_pre_drag.std() <= raw.std()
    # Every asset is penalized by MER + Non-Registered income-tax drag.
    assert (results.expected < results.expected_pre_drag).all()
    engine = AssetLocationEngine(profile)
    for ticker in prices.columns:
        expected_adj = engine.compute_tax_adjusted_return(
            float(results.expected_pre_drag[ticker]), chars[ticker], "NON_REG"
        )
        assert results.expected[ticker] == pytest.approx(expected_adj)
    # Sharpe is recomputed from the adjusted returns at the 2% risk-free rate.
    nonzero = results.volatility > 0
    np.testing.assert_allclose(
        results.sharpe[nonzero].to_numpy(),
        ((results.expected[nonzero] - 0.02) / results.volatility[nonzero]).to_numpy(),
    )


def test_compute_adjusted_metrics_joins_reference_for_shrinkage():
    """Benchmark columns are shrunk jointly with the asset universe so they
    are pulled toward the same cross-sectional grand mean."""
    prices = _seeded_price_frame(seed=5)
    rng = np.random.default_rng(5)
    n = len(prices)
    bm = pd.DataFrame(
        {"VEQT.TO": 100 * (1 + rng.normal(0.0060, 0.010, n)).cumprod()},
        index=prices.index,
    )

    results = compute_adjusted_metrics(prices, reference_prices=bm)

    assert "VEQT.TO" in results.expected.index
    joint = pd.concat([prices, bm], axis=1).dropna()
    raw = joint.pct_change().dropna().mean() * 252
    grand = float(raw.mean())
    shrunk = float(results.expected_pre_drag["VEQT.TO"])
    assert float(raw["VEQT.TO"]) > grand  # sanity: benchmark is the extreme asset
    assert abs(shrunk - grand) <= abs(float(raw["VEQT.TO"]) - grand) + 1e-12
    # Without a tax profile no drag is applied; portfolio assets keep the
    # portfolio-only shrunk values.
    for ticker in prices.columns:
        assert results.expected[ticker] == pytest.approx(
            results.expected_pre_drag[ticker]
        )


def test_compute_adjusted_metrics_rejects_empty_or_disjoint_input():
    with pytest.raises(ValueError):
        compute_adjusted_metrics(pd.DataFrame())

    prices = _seeded_price_frame()
    disjoint = pd.DataFrame(
        {"BM.TO": [100.0, 101.0]}, index=pd.date_range("2024-06-01", periods=2)
    )
    with pytest.raises(ValueError):
        compute_adjusted_metrics(prices, reference_prices=disjoint)


def test_evaluate_adjusted_performance_uses_net_of_drag_returns():
    """Portfolio rows use shrunk, tax/MER-adjusted returns and Ledoit-Wolf
    covariance, with weights normalized and Sharpe at the 2% rf."""
    prices = _seeded_price_frame(seed=9)
    profile = TaxProfile(marginal_tax_rate=0.45)
    chars = {
        "VFV.TO": AssetTaxCharacteristics("VFV.TO", dividend_yield=0.02, mer=0.0010),
        "VCN.TO": AssetTaxCharacteristics("VCN.TO", dividend_yield=0.02, mer=0.0010),
        "QQC.TO": AssetTaxCharacteristics("QQC.TO", dividend_yield=0.02, mer=0.0010),
    }
    mu = shrinkage_expected_return(prices, shrinkage_floor=0.3)

    exp, vol, sharpe = evaluate_adjusted_performance(
        {"VFV.TO": 0.6, "VCN.TO": 0.3, "QQC.TO": 0.1},
        prices,
        expected_returns=mu,
        tax_profile=profile,
        asset_characteristics=chars,
        account="NON_REG",
    )

    engine = AssetLocationEngine(profile)
    mu_adj = pd.Series(
        {
            t: engine.compute_tax_adjusted_return(float(mu[t]), chars[t], "NON_REG")
            for t in prices.columns
        }
    )
    w = np.array([0.6, 0.3, 0.1])
    assert exp == pytest.approx(w @ mu_adj.to_numpy())
    cov = CovarianceShrinkage(prices.dropna()).ledoit_wolf()
    assert vol == pytest.approx(np.sqrt(w @ cov @ w))
    assert sharpe == pytest.approx((exp - 0.02) / vol)
    # Unnormalized weights are rescaled to sum to 1 first.
    exp_norm, vol_norm, sharpe_norm = evaluate_adjusted_performance(
        {"VFV.TO": 6.0, "VCN.TO": 3.0, "QQC.TO": 1.0},
        prices,
        expected_returns=mu,
        tax_profile=profile,
        asset_characteristics=chars,
        account="NON_REG",
    )
    assert (exp_norm, vol_norm, sharpe_norm) == pytest.approx((exp, vol, sharpe))


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
        lambda tickers, **kwargs: {
            "AAPL": data.iloc[:, [0, 1]],
            "MSFT": data.iloc[:, [2, 3]],
        },
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
    monkeypatch.setattr(
        app._STREAMLIT_SERVICE, "download_portfolio_prices", lambda *args, **kwargs: {}
    )

    with pytest.raises(RuntimeError):
        call_cached(app.load_prices, ["AAPL"], "2024-01-01", "2024-02-01")


def test_gather_metadata_handles_success(monkeypatch: pytest.MonkeyPatch):
    class DummyTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {
                "shortName": f"{symbol} Inc",
                "exchange": "NASDAQ",
                "currency": "USD",
            }

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

    preview = call_cached(
        app.load_preview_data, ["AAPL", "MSFT"], pd.Timestamp("2024-03-01")
    )

    assert any(col.endswith("Volume") for col in preview.columns)


def test_load_preview_data_returns_empty_when_download_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(app.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    preview = call_cached(app.load_preview_data, ["AAPL"], pd.Timestamp("2024-03-01"))

    assert preview.empty
