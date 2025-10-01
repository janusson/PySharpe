"""Tests for the dollar-cost averaging utilities."""

from __future__ import annotations

import numpy as np

from pysharpe.visualization import DCAProjection, plot_dca_projection, simulate_dca


def test_simulate_dca_returns_expected_series():
    projection = simulate_dca(
        months=3,
        initial_investment=1000.0,
        monthly_contribution=100.0,
        annual_return_rate=0.12,
    )

    assert isinstance(projection, DCAProjection)
    assert len(projection.months) == 3
    np.testing.assert_array_equal(projection.contributions, np.array([1000.0, 1100.0, 1200.0]))

    monthly_rate = (1 + 0.12) ** (1 / 12) - 1
    expected_balance_month_1 = (1000.0 + 100.0) * (1 + monthly_rate)
    expected_balance_month_2 = (expected_balance_month_1 + 100.0) * (1 + monthly_rate)
    np.testing.assert_allclose(
        projection.balances,
        np.array([1000.0, expected_balance_month_1, expected_balance_month_2]),
    )


def test_plot_dca_projection_returns_axes(monkeypatch):
    class _StubLine:
        def __init__(self, label: str) -> None:
            self._label = label

        def get_label(self) -> str:
            return self._label

    class _StubAxes:
        def __init__(self) -> None:
            self.lines: list[_StubLine] = []

        def plot(self, *_args, label: str, **_kwargs):
            line = _StubLine(label)
            self.lines.append(line)
            return [line]

        def set_xlabel(self, *_args, **_kwargs):
            pass

        def set_ylabel(self, *_args, **_kwargs):
            pass

        def set_title(self, *_args, **_kwargs):
            pass

        def legend(self):
            pass

        def grid(self, *_args, **_kwargs):
            pass

    class _StubMatplotlib:
        def __init__(self) -> None:
            self.axes = _StubAxes()

        def subplots(self):
            return (None, self.axes)

        def show(self):
            pass

    stub = _StubMatplotlib()
    monkeypatch.setattr("pysharpe.visualization.dca._require_matplotlib", lambda: stub)

    projection = simulate_dca(
        months=2,
        initial_investment=500.0,
        monthly_contribution=50.0,
        annual_return_rate=0.05,
    )

    ax = plot_dca_projection(projection, show=False)
    assert len(ax.lines) == 2
    labels = {line.get_label() for line in ax.lines}
    assert labels == {"Total Balance", "Total Contributions"}


def test_simulate_dca_validates_input():
    try:
        simulate_dca(
            months=0,
            initial_investment=100.0,
            monthly_contribution=10.0,
            annual_return_rate=0.05,
        )
    except ValueError:
        assert True
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected ValueError for non-positive months")

    try:
        simulate_dca(
            months=12,
            initial_investment=100.0,
            monthly_contribution=10.0,
            annual_return_rate=-1.5,
        )
    except ValueError:
        assert True
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected ValueError for unrealistic annual return")
