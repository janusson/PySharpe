"""Tests for the equity-curve and correlation visualization helpers.

.. note::

    **Canadian ETF Context** — These helpers render CAD-denominated
    broad-market ETF comparisons: equity curves against a buy-and-hold
    baseline, cumulative returns from a common inception date, and pairwise
    return correlations.  All price data is synthetic with fixed values — no
    live network calls.  Rendering uses the headless ``Agg`` backend so the
    suite never requires a display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe.data.fetcher import PriceFetcher, PriceHistoryError
from pysharpe.exceptions import PySharpeError
from pysharpe.visualization import (
    plot_comparative_returns,
    plot_correlation_heatmap,
    plot_equity_curves,
    plot_holdings_history,
)

# matplotlib is optional in the same way the modules under test treat it.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # deterministic, headless rendering


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures opened by a test so they cannot leak between tests."""

    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture()
def sns():
    """Ensure seaborn is importable for the heatmap tests."""

    return pytest.importorskip("seaborn")


class _StubFetcher(PriceFetcher):
    """Deterministic in-memory fetcher: ticker → DataFrame or Exception."""

    def __init__(self, responses: dict[str, pd.DataFrame | Exception]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def fetch_history(self, ticker, *, period, interval, start=None, end=None):
        self.calls.append(ticker)
        response = self.responses.get(ticker, pd.DataFrame())
        if isinstance(response, Exception):
            raise response
        return response


def _frame(start: str, values: list[float], column: str = "Close") -> pd.DataFrame:
    """Build a single-close price frame on a daily index."""

    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({column: values}, index=dates)


# ---------------------------------------------------------------------------
# plot_equity_curves
# ---------------------------------------------------------------------------


def test_plot_equity_curves_defaults_labels_and_values():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    optimized = pd.Series([100.0, 105.0, 103.0, 110.0, 115.0], index=dates)
    baseline = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)

    ax = plot_equity_curves(optimized, baseline)

    assert ax.get_title() == "Equity Curve Comparison: Optimized vs. Baseline"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Portfolio Value ($)"
    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {
        "Sharpe-Optimized Portfolio",
        "Buy-and-Hold Baseline",
    }
    # Curves are plotted in order with the values passed straight through.
    np.testing.assert_allclose(ax.lines[0].get_ydata(), optimized.to_numpy())
    np.testing.assert_allclose(ax.lines[1].get_ydata(), baseline.to_numpy())


def test_plot_equity_curves_custom_title_and_existing_axes():
    import matplotlib.pyplot as plt

    _, existing = plt.subplots()
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    optimized = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
    baseline = pd.Series([1.0, 1.5, 2.0, 2.5], index=dates)

    returned = plot_equity_curves(optimized, baseline, ax=existing, title="My Curve")

    assert returned is existing
    assert existing.get_title() == "My Curve"


def test_plot_equity_curves_show_triggers_display(monkeypatch):
    import matplotlib.pyplot as plt

    calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(True))
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([1.0, 2.0, 3.0], index=dates)

    plot_equity_curves(series, series, show=True)

    assert calls == [True]


# ---------------------------------------------------------------------------
# plot_comparative_returns
# ---------------------------------------------------------------------------


def test_plot_comparative_returns_aligns_to_common_inception():
    fetcher = _StubFetcher(
        {
            "AAA": _frame("2023-01-01", [90.0, 95.0, 100.0, 110.0, 120.0]),
            "BBB": _frame("2023-01-03", [50.0, 60.0, 66.0, 72.0, 78.0]),
        }
    )

    ax = plot_comparative_returns(["AAA", "BBB"], fetcher)

    # Only the overlapping window (2023-01-03 → 2023-01-05) is plotted and the
    # default title reports the youngest asset's inception date.
    assert ax.get_title() == "Cumulative Returns from 2023-01-03"
    assert ax.get_ylabel() == "Cumulative Return (%)"
    # Two series plus the 0 % reference line drawn by axhline.
    lines = {
        line.get_label(): line
        for line in ax.lines
        if line.get_label() in {"AAA", "BBB"}
    }
    assert set(lines) == {"AAA", "BBB"}
    # Each series starts at 0 % and is normalized to its value on the common date.
    np.testing.assert_allclose(lines["AAA"].get_ydata(), [0.0, 10.0, 20.0])
    np.testing.assert_allclose(lines["BBB"].get_ydata(), [0.0, 20.0, 32.0])


def test_plot_comparative_returns_prefers_adjusted_close():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0], "Adj Close": [10.0, 20.0, 40.0]}, index=dates
    )
    fetcher = _StubFetcher({"AAA": frame})

    ax = plot_comparative_returns(["AAA"], fetcher)

    # Normalized against Adj Close → 0 %, +100 %, +300 % (Close would give +200 %).
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.0, 100.0, 300.0])


def test_plot_comparative_returns_skips_failed_and_empty_tickers():
    fetcher = _StubFetcher(
        {
            "AAA": _frame("2023-01-01", [100.0, 110.0, 121.0]),
            "BBB": PriceHistoryError("no data"),
            "CCC": pd.DataFrame(),
        }
    )

    ax = plot_comparative_returns(["AAA", "BBB", "CCC"], fetcher)

    assert fetcher.calls == ["AAA", "BBB", "CCC"]
    series_lines = [line for line in ax.lines if line.get_label() == "AAA"]
    assert len(series_lines) == 1


def test_plot_comparative_returns_raises_when_no_valid_data():
    fetcher = _StubFetcher({"AAA": PriceHistoryError("boom"), "BBB": pd.DataFrame()})

    with pytest.raises(ValueError, match="No valid price data"):
        plot_comparative_returns(["AAA", "BBB"], fetcher)


def test_plot_comparative_returns_raises_without_overlap():
    fetcher = _StubFetcher(
        {
            "AAA": _frame("2023-01-01", [100.0, 101.0, 102.0]),
            "BBB": _frame("2023-06-01", [50.0, 51.0, 52.0]),
        }
    )

    with pytest.raises(ValueError, match="No overlapping date range"):
        plot_comparative_returns(["AAA", "BBB"], fetcher)


def test_plot_comparative_returns_custom_title_and_existing_axes():
    import matplotlib.pyplot as plt

    _, existing = plt.subplots()
    fetcher = _StubFetcher({"AAA": _frame("2023-01-01", [100.0, 110.0])})

    returned = plot_comparative_returns(
        ["AAA"], fetcher, ax=existing, title="Comparison"
    )

    assert returned is existing
    assert existing.get_title() == "Comparison"


# ---------------------------------------------------------------------------
# plot_holdings_history
# ---------------------------------------------------------------------------


def _holdings_frame(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    base = np.linspace(100.0, 130.0, periods)
    return pd.DataFrame(
        {"AAA": base, "BBB": base * 0.5, "CCC": base * 1.2}, index=dates
    )


def test_plot_holdings_history_plots_every_column():
    frame = _holdings_frame(periods=30)

    ax = plot_holdings_history(frame)

    # One line per holding, plus the 0 % reference line added by axhline.
    series_lines = [
        line for line in ax.lines if line.get_label() in {"AAA", "BBB", "CCC"}
    ]
    assert len(series_lines) == 3
    assert {line.get_label() for line in series_lines} == {"AAA", "BBB", "CCC"}
    assert ax.get_ylabel() == "Cumulative Return (%)"
    assert ax.get_title() == (
        "Holdings History — Cumulative Returns from 2024-01-01 "
        "to 2024-01-30 (3 tickers)"
    )
    # Every series is normalized to 0 % on the common inception date.
    for line in series_lines:
        assert line.get_ydata()[0] == pytest.approx(0.0)


def test_plot_holdings_history_empty_frame_raises():
    with pytest.raises(PySharpeError, match="empty"):
        plot_holdings_history(pd.DataFrame())


def test_plot_holdings_history_no_overlap_lists_ticker_ranges():
    dates_a = pd.date_range("2024-01-01", periods=5, freq="D")
    dates_b = pd.date_range("2024-06-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "AAA": pd.Series(np.arange(5.0) + 100, index=dates_a),
            "BBB": pd.Series(np.arange(5.0) + 50, index=dates_b),
        }
    )

    with pytest.raises(PySharpeError, match="No overlapping date range") as excinfo:
        plot_holdings_history(frame)

    # The diagnostic message names each ticker and its available range.
    assert "AAA: 2024-01-01 to 2024-01-05" in str(excinfo.value)
    assert "BBB: 2024-06-01 to 2024-06-05" in str(excinfo.value)


def test_plot_holdings_history_raises_when_overlap_below_minimum():
    dates_long = pd.date_range("2024-01-01", periods=25, freq="D")
    dates_late = pd.date_range("2024-01-21", periods=21, freq="D")
    frame = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100.0, 120.0, 25), index=dates_long),
            "BBB": pd.Series(np.linspace(50.0, 60.0, 21), index=dates_late),
        }
    )

    # 2024-01-21 → 2024-01-25 is a 5-day intersection, below the 20-day default.
    with pytest.raises(PySharpeError, match="only 5 trading days") as excinfo:
        plot_holdings_history(frame)

    assert "minimum 20 required" in str(excinfo.value)


def test_plot_holdings_history_respects_min_trading_days():
    dates_long = pd.date_range("2024-01-01", periods=25, freq="D")
    dates_late = pd.date_range("2024-01-21", periods=21, freq="D")
    frame = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100.0, 120.0, 25), index=dates_long),
            "BBB": pd.Series(np.linspace(50.0, 60.0, 21), index=dates_late),
        }
    )

    ax = plot_holdings_history(frame, min_trading_days=3)

    series_lines = [line for line in ax.lines if line.get_label() in {"AAA", "BBB"}]
    assert len(series_lines) == 2
    assert ax.get_title() == (
        "Holdings History — Cumulative Returns from 2024-01-21 "
        "to 2024-01-25 (2 tickers)"
    )


def test_plot_holdings_history_show_triggers_display(monkeypatch):
    import matplotlib.pyplot as plt

    calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(True))

    plot_holdings_history(_holdings_frame(), show=True)

    assert calls == [True]


# ---------------------------------------------------------------------------
# plot_correlation_heatmap
# ---------------------------------------------------------------------------


def test_plot_correlation_heatmap_annotates_daily_return_matrix(sns):
    import matplotlib.pyplot as plt

    steps = np.arange(30, dtype=float)
    prices = pd.DataFrame(
        {"AAA": steps + 100, "BBB": (steps + 100) * 2, "CCC": 200 - steps}
    )
    _, existing = plt.subplots()

    ax = plot_correlation_heatmap(prices, ax=existing)

    assert ax is existing
    assert ax.get_title() == "Portfolio Asset Correlation Heatmap (Daily Returns)"
    # One annotated cell per (row, column) pair in the 3 × 3 matrix.
    assert len(ax.texts) == 9
    # The heatmap mesh itself is drawn onto the provided axes.
    assert len(ax.collections) >= 1


def test_plot_correlation_heatmap_custom_title_creates_axes(sns):
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [3.0, 2.0, 1.0]})

    ax = plot_correlation_heatmap(prices, title="Custom Correlations")

    assert ax.get_title() == "Custom Correlations"
    assert len(ax.texts) == 4


def test_plot_correlation_heatmap_tolerates_partial_overlap(sns):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 110.0, 10),
            "BBB": [np.nan] * 3 + list(np.linspace(50.0, 57.0, 7)),
        },
        index=dates,
    )

    ax = plot_correlation_heatmap(frame)

    # Pairwise correlation keeps the full 2 × 2 grid despite the leading NaNs,
    # and every annotated cell is a valid correlation value.
    assert len(ax.texts) == 4
    annotations = sorted(float(text.get_text()) for text in ax.texts)
    # Both series rise linearly over their overlapping window → correlation 1.
    assert annotations == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_plot_correlation_heatmap_show_triggers_display(sns, monkeypatch):
    import matplotlib.pyplot as plt

    calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(True))
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [3.0, 2.0, 1.0]})

    plot_correlation_heatmap(prices, show=True)

    assert calls == [True]
