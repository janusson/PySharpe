"""Unit tests for the head-to-head fund comparison engine.

Uses ONLY synthetic price data with fixed ``numpy.random`` seeds — no live
network calls via ``yfinance`` or ``requests``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.analysis.comparison import (
    _compute_comparison,
    _fetch_and_align,
    _rolling_correlation,
    _rolling_tracking_error,
    compare_two_funds,
)
from pysharpe.metrics import (
    annualize_volatility,
    cagr,
    calmar_ratio,
    compute_returns,
    max_drawdown_duration,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)

# ---------------------------------------------------------------------------
# Helper: build synthetic aligned price DataFrames
# ---------------------------------------------------------------------------


def _make_price_df(
    seed: int,
    periods: int = 504,
    start: str = "2022-01-03",
    drift: float = 0.0005,
    vol: float = 0.012,
    initial: float = 100.0,
    name: str = "FUND",
) -> pd.DataFrame:
    """Build a single-column DataFrame of synthetic daily prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    daily_returns = rng.normal(loc=drift, scale=vol, size=periods - 1)
    prices = initial * np.cumprod(1 + daily_returns)
    prices = np.concatenate([[initial], prices])
    return pd.DataFrame({name: prices}, index=dates)


# ---------------------------------------------------------------------------
# Tests for new metrics functions
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_basic(self):
        rng = np.random.default_rng(99)
        rets = pd.Series(rng.normal(0.001, 0.02, 500), dtype=float)
        result = sortino_ratio(rets, risk_free_rate=0.03, periods_per_year=252)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_all_positive_returns(self):
        """Downside deviation is zero when no return falls below target."""
        rets = pd.Series([0.01, 0.02, 0.015, 0.03], dtype=float)
        with pytest.raises(ValueError, match="Downside deviation is zero"):
            sortino_ratio(rets, periods_per_year=252)

    def test_zero_vol_scenario(self):
        """Zero returns → no downside deviation → error."""
        rets = pd.Series([0.0, 0.0, 0.0], dtype=float)
        with pytest.raises(ValueError, match="Downside deviation is zero"):
            sortino_ratio(rets, periods_per_year=252)


class TestCalmarRatio:
    def test_basic(self):
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        rng = np.random.default_rng(7)
        vals = pd.Series(
            100 * (1 + rng.normal(0.0003, 0.01, 252)).cumprod(),
            index=dates,
            dtype=float,
        )
        result = calmar_ratio(vals)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_no_drawdown(self):
        """When the series never declines, Calmar is infinite."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        vals = pd.Series([100, 101, 102, 103, 104], index=dates, dtype=float)
        result = calmar_ratio(vals)
        assert result == float("inf")

    def test_always_declining(self):
        """When the series always declines, Calmar is negative."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        vals = pd.Series([100, 99, 98, 97, 96], index=dates, dtype=float)
        result = calmar_ratio(vals)
        assert result < 0


class TestTrackingError:
    def test_basic(self):
        rng = np.random.default_rng(3)
        ra = pd.Series(rng.normal(0.001, 0.02, 500), dtype=float)
        rb = pd.Series(rng.normal(0.0005, 0.018, 500), dtype=float)
        result = tracking_error(ra, rb, periods_per_year=252)
        assert isinstance(result, float)
        assert result > 0

    def test_identical_series(self):
        """Tracking error for two identical return series is zero."""
        rets = pd.Series([0.01, -0.005, 0.02, 0.0], dtype=float)
        result = tracking_error(rets, rets.copy(), periods_per_year=252)
        assert np.isclose(result, 0.0)

    def test_length_mismatch(self):
        ra = pd.Series([0.01, 0.02], dtype=float)
        rb = pd.Series([0.01], dtype=float)
        with pytest.raises(ValueError, match="same length"):
            tracking_error(ra, rb)


class TestMaxDrawdownDuration:
    def test_basic(self):
        vals = pd.Series([100, 90, 95, 80, 105], dtype=float)
        # Three consecutive days where value < all-time high (90, 95, 80).
        assert max_drawdown_duration(vals) == 3

    def test_no_drawdown(self):
        vals = pd.Series([100, 101, 102], dtype=float)
        assert max_drawdown_duration(vals) == 0

    def test_empty(self):
        assert max_drawdown_duration(pd.Series([], dtype=float)) == 0

    def test_multiple_drawdowns(self):
        """Only the longest contiguous drawdown period is returned."""
        vals = pd.Series([100, 98, 96, 99, 95, 93, 97, 101], dtype=float)
        # All values from index 1 through 6 are below the ATH of 100,
        # forming one continuous 6-day drawdown.
        result = max_drawdown_duration(vals)
        assert result == 6


# ---------------------------------------------------------------------------
# Rolling helper tests
# ---------------------------------------------------------------------------


class TestRollingTrackingError:
    def test_basic(self):
        rng = np.random.default_rng(11)
        ra = pd.Series(rng.normal(0.001, 0.02, 600), dtype=float)
        rb = pd.Series(rng.normal(0.0005, 0.018, 600), dtype=float)
        te = _rolling_tracking_error(ra, rb, window=252)
        assert len(te) == 600
        # First 251 values should be NaN (min_periods=252)
        assert te.iloc[:251].isna().all()
        assert te.iloc[251:].notna().all()
        assert (te.iloc[251:] >= 0).all()

    def test_identical_series(self):
        rets = pd.Series(np.linspace(0.001, 0.01, 500), dtype=float)
        te = _rolling_tracking_error(rets, rets.copy(), window=100)
        # After warmup, tracking error of identical series should be ~0
        assert np.allclose(te.dropna().to_numpy(dtype=float), 0.0, atol=1e-10)


class TestRollingCorrelation:
    def test_basic(self):
        rng = np.random.default_rng(13)
        ra = pd.Series(rng.normal(0.001, 0.02, 600), dtype=float)
        rb = ra * 0.8 + pd.Series(rng.normal(0.0, 0.01, 600), dtype=float)
        corr = _rolling_correlation(ra, rb, window=252)
        assert corr.iloc[:251].isna().all()
        assert corr.iloc[251:].notna().all()
        assert (corr.dropna().between(-1, 1)).all()


# ---------------------------------------------------------------------------
# Core _compute_comparison tests (stateless, no network)
# ---------------------------------------------------------------------------


class TestComputeComparison:
    @pytest.fixture(autouse=True)
    def _clear_settings_cache(self):
        """Ensure cached settings don't leak between tests."""
        from pysharpe.config import get_settings

        get_settings.cache_clear()

    def test_basic_output_structure(self):
        prices_a = _make_price_df(seed=1, periods=504, name="AAA")
        prices_b = _make_price_df(seed=2, periods=504, name="BBB")

        result = _compute_comparison(prices_a, prices_b, "AAA", "BBB")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["AAA", "BBB"]
        assert result.index.name == "Metric"

        expected_metrics = {
            "CAGR",
            "Annualized Volatility",
            "Max Drawdown Depth",
            "Max Drawdown Duration (days)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "1Y Rolling Tracking Error (mean)",
            "1Y Rolling Return Correlation (mean)",
        }
        assert set(result.index) == expected_metrics
        assert result.notna().all().all()

    def test_identical_series(self):
        """Two identical funds should have equal metrics + zero tracking error."""
        prices = _make_price_df(seed=5, periods=504, name="AAA")
        prices_copy = prices.copy()
        prices_copy.columns = ["BBB"]

        result = _compute_comparison(prices, prices_copy, "AAA", "BBB")

        # All per-fund metrics should be identical
        assert np.isclose(result.loc["CAGR", "AAA"], result.loc["CAGR", "BBB"])
        assert np.isclose(
            result.loc["Annualized Volatility", "AAA"],
            result.loc["Annualized Volatility", "BBB"],
        )
        assert np.isclose(
            result.loc["Max Drawdown Depth", "AAA"],
            result.loc["Max Drawdown Depth", "BBB"],
        )
        assert np.isclose(
            result.loc["Sharpe Ratio", "AAA"], result.loc["Sharpe Ratio", "BBB"]
        )
        assert np.isclose(
            result.loc["Sortino Ratio", "AAA"], result.loc["Sortino Ratio", "BBB"]
        )
        # Tracking error should be zero for identical series.
        assert np.isclose(
            result.loc["1Y Rolling Tracking Error (mean)", "AAA"], 0.0, atol=1e-10
        )
        # Correlation should be 1.0
        assert np.isclose(
            result.loc["1Y Rolling Return Correlation (mean)", "AAA"], 1.0, atol=1e-6
        )

    def test_metrics_are_consistent_with_direct_calls(self):
        """Spot-check: comparison metrics match direct metrics.py calls."""
        prices_a = _make_price_df(seed=42, periods=504, name="AAA")
        prices_b = _make_price_df(seed=43, periods=504, name="BBB")

        result = _compute_comparison(prices_a, prices_b, "AAA", "BBB")

        series_a = prices_a.iloc[:, 0]
        series_b = prices_b.iloc[:, 0]
        rets_a = compute_returns(series_a)
        rets_b = compute_returns(series_b)

        # CAGR
        assert np.isclose(result.loc["CAGR", "AAA"], cagr(series_a))
        assert np.isclose(result.loc["CAGR", "BBB"], cagr(series_b))

        # Volatility
        assert np.isclose(
            result.loc["Annualized Volatility", "AAA"],
            annualize_volatility(rets_a, periods_per_year=252),
        )
        assert np.isclose(
            result.loc["Annualized Volatility", "BBB"],
            annualize_volatility(rets_b, periods_per_year=252),
        )

        # Max DD
        assert np.isclose(
            result.loc["Max Drawdown Depth", "AAA"], maximum_drawdown(series_a)
        )
        assert np.isclose(
            result.loc["Max Drawdown Depth", "BBB"], maximum_drawdown(series_b)
        )

        # Max DD Duration
        assert result.loc[
            "Max Drawdown Duration (days)", "AAA"
        ] == max_drawdown_duration(series_a)
        assert result.loc[
            "Max Drawdown Duration (days)", "BBB"
        ] == max_drawdown_duration(series_b)

        # Sharpe
        assert np.isclose(
            result.loc["Sharpe Ratio", "AAA"],
            sharpe_ratio(rets_a, risk_free_rate=0.0, periods_per_year=252),
        )

        # Sortino
        assert np.isclose(
            result.loc["Sortino Ratio", "AAA"],
            sortino_ratio(rets_a, risk_free_rate=0.0, periods_per_year=252),
        )

        # Calmar
        assert np.isclose(result.loc["Calmar Ratio", "AAA"], calmar_ratio(series_a))

    def test_risk_free_rate_propagation(self):
        """Verify non-zero risk-free rate affects Sharpe and Sortino."""
        prices_a = _make_price_df(seed=77, periods=504, name="AAA")
        prices_b = _make_price_df(seed=78, periods=504, name="BBB")

        result_rf0 = _compute_comparison(
            prices_a, prices_b, "AAA", "BBB", risk_free_rate=0.0
        )
        result_rf5 = _compute_comparison(
            prices_a, prices_b, "AAA", "BBB", risk_free_rate=0.05
        )

        # Sharpe with positive risk-free should be lower than with zero.
        sharpe_rf0 = float(result_rf0.loc["Sharpe Ratio", "AAA"])
        sharpe_rf5 = float(result_rf5.loc["Sharpe Ratio", "AAA"])
        assert sharpe_rf5 < sharpe_rf0

        sortino_rf0 = float(result_rf0.loc["Sortino Ratio", "AAA"])
        sortino_rf5 = float(result_rf5.loc["Sortino Ratio", "AAA"])
        assert sortino_rf5 < sortino_rf0

    def test_zero_volatility_raises(self):
        """Constant price → zero returns → zero vol → Sharpe/Sortino should raise."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        flat = pd.DataFrame({"AAA": 100.0}, index=dates)
        with pytest.raises(ValueError):
            _compute_comparison(flat, flat, "AAA", "BBB")


# ---------------------------------------------------------------------------
# _fetch_and_align: download/collate orchestration with an injected fetcher
# ---------------------------------------------------------------------------


class _FakeCollationService:
    """Stand-in for ``CollationService`` that records calls and returns a frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.downloads: list[tuple[tuple[str, ...], dict[str, object]]] = []
        self.collate_calls: list[tuple[str, tuple[str, ...]]] = []
        self.fetchers: list[object] = []

    def download_portfolio_prices(self, tickers, **kwargs):
        self.downloads.append((tuple(tickers), kwargs))
        return {}

    def collate_portfolio(self, name, tickers):
        self.collate_calls.append((name, tuple(tickers)))
        return self.frame


def _install_fake_collation(monkeypatch, frame: pd.DataFrame) -> _FakeCollationService:
    """Patch ``CollationService`` where ``_fetch_and_align`` imports it."""

    service = _FakeCollationService(frame)

    def _factory(fetcher):
        service.fetchers.append(fetcher)
        return service

    monkeypatch.setattr("pysharpe.data.collation.CollationService", _factory)
    return service


def _two_column_frame(periods: int = 30, offset_b: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    return pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 110.0, periods),
            "BBB": np.linspace(50.0, 55.0, periods) + offset_b,
        },
        index=dates,
    )


class TestFetchAndAlign:
    def test_downloads_and_collates_each_ticker(self, monkeypatch):
        service = _install_fake_collation(monkeypatch, _two_column_frame())
        fetcher = object()

        prices_a, prices_b = _fetch_and_align(
            "AAA",
            "BBB",
            "2020-01-01",
            "2024-01-01",
            "CAD",
            fetcher=fetcher,
        )

        assert list(prices_a.columns) == ["AAA"]
        assert list(prices_b.columns) == ["BBB"]
        assert prices_a.index.equals(prices_b.index)
        assert len(prices_a) == 30

        assert [call[0] for call in service.downloads] == [("AAA",), ("BBB",)]
        for _, kwargs in service.downloads:
            assert kwargs["period"] == "max"
            assert kwargs["interval"] == "1d"
            assert kwargs["start"] == "2020-01-01"
            assert kwargs["end"] == "2024-01-01"

        assert service.collate_calls == [("_head_to_head_AAA_BBB", ("AAA", "BBB"))]
        assert service.fetchers == [fetcher]

    def test_default_fetcher_is_auto_adjusted_yfinance(self, monkeypatch):
        from pysharpe.data.fetcher import YFinancePriceFetcher

        service = _install_fake_collation(monkeypatch, _two_column_frame())

        _fetch_and_align("AAA", "BBB", None, None, "CAD")

        assert len(service.fetchers) == 1
        assert isinstance(service.fetchers[0], YFinancePriceFetcher)
        assert service.fetchers[0]._history_overrides == {"auto_adjust": True}
        # Absent dates are forwarded as None untouched.
        assert service.downloads[0][1]["start"] is None
        assert service.downloads[0][1]["end"] is None

    def test_aligns_on_intersection_of_non_null_dates(self, monkeypatch):
        frame = _two_column_frame(periods=10)
        # AAA has prices only on the first 6 dates, BBB only on the last 6.
        frame.loc[frame.index[6:], "AAA"] = np.nan
        frame.loc[frame.index[:4], "BBB"] = np.nan
        _install_fake_collation(monkeypatch, frame)

        prices_a, prices_b = _fetch_and_align("AAA", "BBB", None, None, "CAD")

        expected_index = frame.index[4:6]
        assert prices_a.index.equals(expected_index)
        assert prices_b.index.equals(expected_index)
        assert prices_a["AAA"].notna().all()
        assert prices_b["BBB"].notna().all()

    def test_empty_collated_raises(self, monkeypatch):
        _install_fake_collation(monkeypatch, pd.DataFrame())

        with pytest.raises(
            ValueError, match="Could not retrieve overlapping price data"
        ):
            _fetch_and_align("AAA", "BBB", None, None, "CAD")

    def test_missing_ticker_column_raises(self, monkeypatch):
        frame = _two_column_frame().drop(columns=["BBB"])
        _install_fake_collation(monkeypatch, frame)

        with pytest.raises(
            ValueError, match="Could not retrieve overlapping price data"
        ):
            _fetch_and_align("AAA", "BBB", None, None, "CAD")

    def test_single_common_date_raises(self, monkeypatch):
        frame = _two_column_frame(periods=6)
        frame.loc[frame.index[1:], "AAA"] = np.nan
        frame.loc[frame.index[:-1], "BBB"] = np.nan
        _install_fake_collation(monkeypatch, frame)

        with pytest.raises(ValueError, match="Insufficient overlapping data"):
            _fetch_and_align("AAA", "BBB", None, None, "CAD")

    def test_no_common_dates_raises(self, monkeypatch):
        frame = _two_column_frame(periods=4)
        frame.loc[frame.index[:2], "AAA"] = np.nan
        frame.loc[frame.index[2:], "BBB"] = np.nan
        _install_fake_collation(monkeypatch, frame)

        with pytest.raises(ValueError, match="only 0 common dates"):
            _fetch_and_align("AAA", "BBB", None, None, "CAD")


# ---------------------------------------------------------------------------
# compare_two_funds: fetch → compute delegation
# ---------------------------------------------------------------------------


class TestCompareTwoFunds:
    def test_delegates_to_fetch_and_compute(self, monkeypatch):
        prices_a = _make_price_df(seed=11, periods=300, name="AAA")
        prices_b = _make_price_df(seed=12, periods=300, name="BBB")
        captured: dict[str, object] = {}

        def _fake_fetch(
            ticker_a, ticker_b, start_date, end_date, base_currency, fetcher
        ):
            captured.update(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                start_date=start_date,
                end_date=end_date,
                base_currency=base_currency,
                fetcher=fetcher,
            )
            return prices_a, prices_b

        monkeypatch.setattr(
            "pysharpe.analysis.comparison._fetch_and_align", _fake_fetch
        )

        sentinel = object()
        result = compare_two_funds(
            "AAA",
            "BBB",
            "2021-01-01",
            "2024-01-01",
            "USD",
            risk_free_rate=0.03,
            fetcher=sentinel,
        )

        assert captured == {
            "ticker_a": "AAA",
            "ticker_b": "BBB",
            "start_date": "2021-01-01",
            "end_date": "2024-01-01",
            "base_currency": "USD",
            "fetcher": sentinel,
        }

        expected = _compute_comparison(
            prices_a, prices_b, "AAA", "BBB", risk_free_rate=0.03
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_end_to_end_with_fake_collation(self, monkeypatch):
        dates = pd.bdate_range("2022-01-03", periods=600)
        rng = np.random.default_rng(4242)
        a = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.010, dates.size))
        b = 50.0 * np.cumprod(1 + rng.normal(0.0003, 0.011, dates.size))
        frame = pd.DataFrame({"AAA": a, "BBB": b}, index=dates)
        _install_fake_collation(monkeypatch, frame)

        result = compare_two_funds("AAA", "BBB")

        assert list(result.columns) == ["AAA", "BBB"]
        assert "CAGR" in result.index
        assert result.notna().all().all()
