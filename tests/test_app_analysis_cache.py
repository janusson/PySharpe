"""Regression tests for dashboard analysis input identity."""

from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest

import app
from pysharpe.optimization.tax_location import ReturnBasis


@pytest.fixture
def analysis_stubs(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(
        app,
        "st",
        SimpleNamespace(session_state={}, spinner=lambda *_args: nullcontext()),
    )
    monkeypatch.setattr(
        app, "_build_metadata_from_upload", lambda _tickers: pd.DataFrame()
    )
    monkeypatch.setattr(app, "compute_metrics", lambda prices: float(prices.iloc[1, 0]))

    def fake_analysis(prices, _weights, *, return_basis):
        assert return_basis == ReturnBasis.NET_OF_EXPENSES
        calls.append(float(prices.iloc[1, 0]))
        return {"synthetic_result": calls[-1]}

    monkeypatch.setattr(app, "run_full_analysis", fake_analysis)
    return calls


def _controls(csv: str) -> dict[str, object]:
    ticker_sig, state = app._build_upload_state(StringIO(csv))
    return {
        "ticker_sig": ticker_sig,
        "start": "2024-01-01",
        "end": "2024-01-03",
        "custom_weights": {"VFV.TO": 0.5, "VCN.TO": 0.5},
        "return_basis": ReturnBasis.NET_OF_EXPENSES,
        "price_data": state["full_prices"],
    }


def test_replacing_uploaded_prices_invalidates_analysis(analysis_stubs: list[float]):
    original = _controls(
        "Date,VFV.TO,VCN.TO\n2024-01-01,100,50\n2024-01-02,101,51\n2024-01-03,102,52\n"
    )
    revised = _controls(
        "Date,VFV.TO,VCN.TO\n2024-01-01,100,50\n2024-01-02,110,51\n2024-01-03,102,52\n"
    )

    first = app._ensure_analysis(original)
    assert first is not None and first["synthetic_result"] == 101.0
    original_copy = {**original, "price_data": original["price_data"].copy(deep=True)}
    assert app._ensure_analysis(original_copy) is first
    assert analysis_stubs == [101.0]

    assert app._get_cached_analysis(revised) is None
    second = app._ensure_analysis(revised)
    assert second is not None and second["synthetic_result"] == 110.0
    assert second["metrics_result"] == 110.0
    assert analysis_stubs == [101.0, 110.0]
    assert app._ensure_analysis(revised) is second
    assert analysis_stubs == [101.0, 110.0]


def test_mutating_price_frame_invalidates_analysis(analysis_stubs: list[float]):
    controls = _controls(
        "Date,VFV.TO,VCN.TO\n2024-01-01,100,50\n2024-01-02,101,51\n2024-01-03,102,52\n"
    )
    assert app._ensure_analysis(controls) is not None
    prices = controls["price_data"]
    assert isinstance(prices, pd.DataFrame)
    prices.iloc[1, 0] = 110

    assert app._get_cached_analysis(controls) is None
    result = app._ensure_analysis(controls)
    assert result is not None and result["synthetic_result"] == 110.0
    assert analysis_stubs == [101.0, 110.0]


@pytest.mark.parametrize("change", ["index", "column_order", "dtype", "missing"])
def test_changed_frame_structure_invalidates_analysis(
    analysis_stubs: list[float], change: str
):
    controls = _controls(
        "Date,VFV.TO,VCN.TO\n2024-01-01,100,50\n2024-01-02,101,51\n2024-01-03,102,52\n"
    )
    assert app._ensure_analysis(controls) is not None
    prices = controls["price_data"]
    assert isinstance(prices, pd.DataFrame)
    changed = prices.copy(deep=True)
    if change == "index":
        changed.index = changed.index + pd.Timedelta(days=1)
    elif change == "column_order":
        changed = changed.iloc[:, ::-1]
    elif change == "dtype":
        changed = changed.astype(float)
    else:
        changed.iloc[0, 0] = float("nan")

    assert app._get_cached_analysis({**controls, "price_data": changed}) is None
    assert analysis_stubs == [101.0]
