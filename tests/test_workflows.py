"""Tests for high-level workflows.

.. note::

    **Canadian TFSA Context** — Download and optimisation workflows process
    CAD-denominated ETF portfolios.  MER values are decimal fractions.
    Base currency is CAD.  Optimisation results are written to the exports
    directory for use by the VA rebalancing engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysharpe import workflows
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # deterministic, headless rendering


class _SettingsStub:
    """Minimal stand-in for :class:`PySharpeSettings` (no filesystem side effects)."""

    def __init__(self, root) -> None:
        self.export_dir = root
        self.portfolio_dir = root
        self.price_history_dir = root


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures opened by a test so they cannot leak between tests."""

    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _result(name: str) -> OptimisationResult:
    return OptimisationResult(
        name=name,
        weights=PortfolioWeights({"AAA": 0.5, "BBB": 0.5}),
        performance=OptimisationPerformance(
            0.08, 0.15, 1.3, "2020-01-01", "2021-01-01"
        ),
    )


def _prices(start: str, columns: dict[str, list[float]]) -> pd.DataFrame:
    periods = len(next(iter(columns.values())))
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(columns, index=dates)


def _write_collated(directory, name: str, prices: pd.DataFrame) -> None:
    """Persist a collated price CSV in the layout ``plot_holdings_history`` reads."""

    directory.mkdir(parents=True, exist_ok=True)
    frame = prices.copy()
    frame.index.name = "Date"
    frame.to_csv(directory / f"{name}_collated.csv")


# ---------------------------------------------------------------------------
# download_portfolios
# ---------------------------------------------------------------------------


def test_download_portfolios_delegates(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _StubWorkflow:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def process_portfolios(self, **kwargs):  # noqa: D401 - signature mirrors workflow
            captured["process"] = kwargs
            return {"demo": pd.DataFrame({"AAA": [1.0]})}

    monkeypatch.setattr(workflows, "PortfolioDownloadWorkflow", _StubWorkflow)

    portfolio_dir = tmp_path / "portfolio"
    price_dir = tmp_path / "price_hist"
    export_dir = tmp_path / "exports"

    result = workflows.download_portfolios(
        portfolio_dir=portfolio_dir,
        price_history_dir=price_dir,
        export_dir=export_dir,
        period="1y",
        interval="1d",
        start="2020-01-01",
        end="2021-01-01",
    )

    assert set(result.keys()) == {"demo"}
    assert captured["init"]["portfolio_dir"] == portfolio_dir
    assert captured["process"]["period"] == "1y"
    assert captured["process"]["start"] == "2020-01-01"


def test_download_portfolios_handles_empty(monkeypatch, tmp_path):
    class _StubWorkflow:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_portfolios(self, **_kwargs):  # noqa: D401
            return {}

    monkeypatch.setattr(workflows, "PortfolioDownloadWorkflow", _StubWorkflow)

    result = workflows.download_portfolios(
        portfolio_dir=tmp_path / "portfolio",
        price_history_dir=tmp_path / "price_hist",
        export_dir=tmp_path / "exports",
    )

    assert result == {}


def test_download_portfolios_defaults_to_settings_dirs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _StubWorkflow:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def process_portfolios(self, **_kwargs):  # noqa: D401
            return {"demo": pd.DataFrame()}

    monkeypatch.setattr(workflows, "PortfolioDownloadWorkflow", _StubWorkflow)
    monkeypatch.setattr(workflows, "get_settings", lambda: _SettingsStub(tmp_path))

    workflows.download_portfolios()

    # Every directory falls back to the settings-provided location.
    assert captured["init"]["portfolio_dir"] == tmp_path
    assert captured["init"]["price_history_dir"] == tmp_path
    assert captured["init"]["export_dir"] == tmp_path


# ---------------------------------------------------------------------------
# optimise_portfolios
# ---------------------------------------------------------------------------


def test_optimise_portfolios_skips_failed_runs(monkeypatch, tmp_path):
    collated_dir = tmp_path / "exports"
    collated_dir.mkdir()
    (collated_dir / "alpha_collated.csv").write_text("Date,AAA\n", encoding="utf-8")

    def _raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    def _should_not_be_called(**_kwargs):
        raise AssertionError("optimise_all_portfolios should not be invoked")

    monkeypatch.setattr(workflows, "optimise_portfolio", _raise_missing)
    monkeypatch.setattr(workflows, "optimise_all_portfolios", _should_not_be_called)

    result = workflows.optimise_portfolios(
        collated_dir=collated_dir,
        output_dir=collated_dir,
        make_plot=False,
    )

    assert result == {}


def test_optimise_portfolios_with_names(monkeypatch, tmp_path):
    collated_dir = tmp_path / "exports"
    collated_dir.mkdir()

    def _fake_optimize(name: str, **kwargs):  # noqa: D401
        return _result(name)

    monkeypatch.setattr(workflows, "optimise_portfolio", _fake_optimize)

    result = workflows.optimise_portfolios(
        portfolio_names=["demo"],
        collated_dir=collated_dir,
        output_dir=collated_dir,
        make_plot=False,
    )

    assert set(result.keys()) == {"demo"}
    assert isinstance(result["demo"], OptimisationResult)


def test_optimise_portfolios_returns_empty_without_collated_files(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(workflows, "get_settings", lambda: _SettingsStub(tmp_path))
    calls: list[object] = []
    monkeypatch.setattr(
        workflows, "optimise_portfolio", lambda *a, **k: calls.append(a)
    )

    result = workflows.optimise_portfolios(collated_dir=tmp_path)

    assert result == {}
    assert calls == []


def test_optimise_portfolios_forwards_constraints(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "get_settings", lambda: _SettingsStub(tmp_path))
    captured: dict[str, dict[str, object]] = {}

    def _fake_optimize(name: str, **kwargs):  # noqa: D401
        captured["kwargs"] = kwargs
        return _result(name)

    monkeypatch.setattr(workflows, "optimise_portfolio", _fake_optimize)

    workflows.optimise_portfolios(
        portfolio_names=["demo"],
        collated_dir=tmp_path,
        output_dir=tmp_path,
        time_constraint="2020-01-01",
        mer_mapping={"AAA": 0.0017},
        max_portfolio_mer=0.0025,
        geo_mapping={"AAA": "US"},
        geo_lower_bounds={"US": 0.1},
        geo_upper_bounds={"US": 0.9},
        make_plot=False,
        category_map={"AAA": "Equity"},
        include_unmapped_categories=False,
        return_model="mean",
        base_currency="USD",
        max_weight=0.6,
        shrinkage_floor=0.5,
    )

    kwargs = captured["kwargs"]
    assert kwargs["return_model"] == "mean"
    assert kwargs["max_weight"] == 0.6
    assert kwargs["shrinkage_floor"] == 0.5
    assert kwargs["time_constraint"] == "2020-01-01"
    # MER values stay decimal fractions — never divided or re-scaled here.
    assert kwargs["mer_mapping"] == {"AAA": 0.0017}
    assert kwargs["max_portfolio_mer"] == 0.0025
    assert kwargs["geo_mapping"] == {"AAA": "US"}
    assert kwargs["geo_lower_bounds"] == {"US": 0.1}
    assert kwargs["geo_upper_bounds"] == {"US": 0.9}
    assert kwargs["category_map"] == {"AAA": "Equity"}
    assert kwargs["include_unmapped_categories"] is False
    assert kwargs["base_currency"] == "USD"
    assert kwargs["make_plot"] is False
    assert kwargs["execution_config"] is None
    assert kwargs["proxy_map"] is None


# ---------------------------------------------------------------------------
# plot_holdings_history
# ---------------------------------------------------------------------------


class TestPlotHoldingsHistory:
    """Coverage for the collated-CSV → cumulative-return plotting workflow."""

    # AAA compounds at +10 %/day; BBB is flat.
    AAA = [100.0, 110.0, 121.0, 133.1]
    FLAT = [100.0, 100.0, 100.0, 100.0]

    def _setup(self, tmp_path, monkeypatch, **frames: pd.DataFrame) -> None:
        for name, prices in frames.items():
            _write_collated(tmp_path, name, prices)
        monkeypatch.setattr(workflows, "get_settings", lambda: _SettingsStub(tmp_path))
        # Identity FX so no network call is attempted on the default path.
        monkeypatch.setattr(workflows, "apply_fx_conversion", lambda df, **k: df)

    @staticmethod
    def _line(ax, label: str):
        matches = [line for line in ax.lines if line.get_label() == label]
        assert len(matches) == 1, f"expected exactly one '{label}' line"
        return matches[0]

    def test_equal_weight_normalises_and_titles(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA, "BBB": self.FLAT}),
        )

        ax = workflows.plot_holdings_history(
            portfolio_names=["demo"], collated_dir=tmp_path
        )

        assert ax.get_ylabel() == "Cumulative Return (normalised)"
        # The title reports the first *return* date (pct_change drops the
        # first price row), i.e. one day after the price history begins.
        assert ax.get_title() == "Cumulative Returns from 2024-01-02"
        # Equal weight: (+10 % + 0 %)/2 = +5 % per day, normalised to 1.0.
        np.testing.assert_allclose(
            self._line(ax, "demo").get_ydata(), [1.0, 1.05, 1.1025]
        )

    def test_weighted_allocation_uses_supplied_weights(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA, "BBB": self.FLAT}),
        )

        ax = workflows.plot_holdings_history(
            portfolio_names=["demo"],
            collated_dir=tmp_path,
            weights={"demo": {"AAA": 0.75, "BBB": 0.25}},
        )

        # 0.75 * 10 % + 0.25 * 0 % = 7.5 % per day.
        np.testing.assert_allclose(
            self._line(ax, "demo").get_ydata(), [1.0, 1.075, 1.155625]
        )

    def test_weights_filter_to_matching_tickers(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA, "BBB": self.FLAT}),
        )

        ax = workflows.plot_holdings_history(
            portfolio_names=["demo"],
            collated_dir=tmp_path,
            weights={"demo": {"AAA": 1.0}},  # BBB not mentioned → dropped
        )

        np.testing.assert_allclose(self._line(ax, "demo").get_ydata(), [1.0, 1.1, 1.21])

    def test_weights_with_no_matching_tickers_is_skipped(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA}),
        )

        with pytest.raises(ValueError, match="No valid portfolio return series"):
            workflows.plot_holdings_history(
                portfolio_names=["demo"],
                collated_dir=tmp_path,
                weights={"demo": {"ZZZ": 1.0}},
            )

    def test_discovers_collated_files_and_uses_latest_inception(
        self, tmp_path, monkeypatch
    ):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA}),
            core=_prices("2024-01-03", {"CCC": self.FLAT}),
        )

        # No collated_dir → falls back to settings.export_dir; no names → glob.
        ax = workflows.plot_holdings_history()

        labels = {line.get_label() for line in ax.lines}
        assert {"demo", "core"} <= labels
        # Title reports the youngest portfolio's first return date (2024-01-04
        # is the day after 'core' starts).
        assert ax.get_title() == "Cumulative Returns from 2024-01-04"

    def test_missing_collated_files_raise(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)

        with pytest.raises(FileNotFoundError, match="No collated portfolios found"):
            workflows.plot_holdings_history(collated_dir=tmp_path)

    def test_named_portfolio_without_csv_is_skipped(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="No valid portfolio return series"):
            workflows.plot_holdings_history(
                portfolio_names=["ghost"], collated_dir=tmp_path
            )

    def test_time_constraint_filters_history(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA, "BBB": self.FLAT}),
        )

        ax = workflows.plot_holdings_history(
            portfolio_names=["demo"],
            collated_dir=tmp_path,
            time_constraint="2024-01-03",
        )

        # The constraint keeps 2024-01-03 and 2024-01-04, so the return series
        # starts on 2024-01-04 (and holds a single normalized point).
        assert ax.get_title() == "Cumulative Returns from 2024-01-04"
        # Only 2024-01-03 and 2024-01-04 remain → a single normalized point.
        assert len(self._line(ax, "demo").get_ydata()) == 1

    def test_time_constraint_beyond_history_is_skipped(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path,
            monkeypatch,
            demo=_prices("2024-01-01", {"AAA": self.AAA}),
        )

        with pytest.raises(ValueError, match="No valid portfolio return series"):
            workflows.plot_holdings_history(
                portfolio_names=["demo"],
                collated_dir=tmp_path,
                time_constraint="2025-01-01",
            )

    def test_missing_values_are_forward_filled(self, tmp_path, monkeypatch):
        prices = _prices("2024-01-01", {"AAA": self.AAA, "BBB": self.FLAT})
        prices.iloc[2, 1] = np.nan
        self._setup(tmp_path, monkeypatch, demo=prices)

        ax = workflows.plot_holdings_history(
            portfolio_names=["demo"], collated_dir=tmp_path
        )

        # The gap is forward-filled, so a full curve is still produced.
        assert len(self._line(ax, "demo").get_ydata()) == 3

    def test_apply_fx_true_invokes_conversion_with_base_currency(
        self, tmp_path, monkeypatch
    ):
        self._setup(
            tmp_path, monkeypatch, demo=_prices("2024-01-01", {"AAA": self.AAA})
        )
        calls: list[str] = []
        monkeypatch.setattr(
            workflows,
            "apply_fx_conversion",
            lambda df, base_currency=None, **k: (calls.append(base_currency), df)[1],
        )

        workflows.plot_holdings_history(
            portfolio_names=["demo"],
            collated_dir=tmp_path,
            apply_fx=True,
            base_currency="USD",
        )

        assert calls == ["USD"]

    def test_apply_fx_false_skips_conversion(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path, monkeypatch, demo=_prices("2024-01-01", {"AAA": self.AAA})
        )
        calls: list[int] = []
        monkeypatch.setattr(
            workflows, "apply_fx_conversion", lambda *a, **k: calls.append(1)
        )

        workflows.plot_holdings_history(
            portfolio_names=["demo"], collated_dir=tmp_path, apply_fx=False
        )

        assert calls == []

    def test_custom_title_and_show(self, tmp_path, monkeypatch):
        import matplotlib.pyplot as plt

        self._setup(
            tmp_path, monkeypatch, demo=_prices("2024-01-01", {"AAA": self.AAA})
        )
        shown: list[bool] = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))

        ax = workflows.plot_holdings_history(
            portfolio_names=["demo"],
            collated_dir=tmp_path,
            title="Hedged vs Unhedged",
            show=True,
        )

        assert ax.get_title() == "Hedged vs Unhedged"
        assert shown == [True]

    def test_all_missing_column_is_skipped(self, tmp_path, monkeypatch):
        # BBB is entirely absent: ffill cannot repair it, so dropna removes
        # every row and the portfolio is skipped.
        prices = _prices("2024-01-01", {"AAA": self.AAA, "BBB": [np.nan] * 4})
        self._setup(tmp_path, monkeypatch, demo=prices)

        with pytest.raises(ValueError, match="No valid portfolio return series"):
            workflows.plot_holdings_history(
                portfolio_names=["demo"], collated_dir=tmp_path, apply_fx=False
            )

    def test_output_dir_persists_png(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path, monkeypatch, demo=_prices("2024-01-01", {"AAA": self.AAA})
        )
        output_dir = tmp_path / "plots"

        workflows.plot_holdings_history(
            portfolio_names=["demo"], collated_dir=tmp_path, output_dir=output_dir
        )

        png = output_dir / "demo_holdings_history.png"
        assert png.exists()
        assert png.stat().st_size > 0

    def test_no_output_dir_writes_nothing(self, tmp_path, monkeypatch):
        self._setup(
            tmp_path, monkeypatch, demo=_prices("2024-01-01", {"AAA": self.AAA})
        )

        workflows.plot_holdings_history(portfolio_names=["demo"], collated_dir=tmp_path)

        assert list(tmp_path.glob("*.png")) == []
