"""Offline example: known synthetic data and real Streamlit navigation."""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf
from streamlit.testing.v1 import AppTest

from pysharpe.app.offline_example import _example_prices, _price_multipliers


def test_example_chart_matches_hand_computed_price_multipliers():
    prices = _example_prices()
    multipliers = _price_multipliers(prices)

    assert multipliers.iloc[0].tolist() == [1.0, 1.0]
    assert multipliers.iloc[1].tolist() == pytest.approx([1.02, 0.99])
    assert multipliers.iloc[-1].tolist() == pytest.approx([1.10, 1.05])
    pd.testing.assert_frame_equal(
        _price_multipliers(prices.iloc[:4]), multipliers.iloc[:4]
    )
    assert prices.index.min() == pd.Timestamp("2024-01-02")
    assert prices.index.max() == pd.Timestamp("2024-08-01")


def test_first_run_example_loads_without_network_or_portfolio(monkeypatch):
    attempted = []

    def deny_network(*args, **kwargs):
        attempted.append((args, kwargs))
        raise AssertionError("offline example attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(yf, "download", deny_network)
    monkeypatch.setattr(yf, "Ticker", deny_network)

    # Confirm that the guard fails closed rather than merely recording calls.
    with pytest.raises(AssertionError, match="attempted network access"):
        yf.download("FAKE")
    attempted.clear()

    at = AppTest.from_file("demo.py", default_timeout=60).run()
    assert not at.exception
    assert at.button[0].label == "Load example dataset"
    assert not at.metric
    assert not attempted

    at.button[0].click().run()
    assert not at.exception
    assert not attempted
    assert [(item.label, item.value) for item in at.metric] == [
        ("Fictional ETF A · price change", "+10.0%"),
        ("Fictional ETF B · price change", "+5.0%"),
    ]
    assert "not market history, a recommendation, or a prediction" in at.info[0].value
    assert "CAD-denominated" in " ".join(item.value for item in at.markdown)
    assert "January 2 to August 1, 2024" in " ".join(item.value for item in at.markdown)
    assert at.dataframe[0].value.equals(_example_prices())
    assert at.get("vega_lite_chart")
    assert at.get("download_button")
    assert at.code[0].value == "uv run streamlit run app.py"
    assert "request market data immediately" in " ".join(
        item.value for item in at.markdown
    )
    assert not at.tabs

    # An unrelated rerun keeps the example visible without a provider call.
    at.run()
    assert not at.exception
    assert len(at.metric) == 2
    assert not attempted


def test_fresh_process_does_not_initialize_research_data(tmp_path):
    script = """\
import socket
import sys

def deny_network(*args, **kwargs):
    raise AssertionError('unexpected network access')

socket.socket.connect = deny_network
socket.create_connection = deny_network
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('demo.py', default_timeout=60).run()
at.button[0].click().run()
assert not at.exception, at.exception
assert len(at.metric) == 2
assert 'pysharpe.app.data' not in sys.modules
"""
    env = os.environ.copy()
    data_dir = tmp_path / "research-data"
    env["PYSHARPE_DATA_DIR"] = str(data_dir)
    env["PYSHARPE_LOG_DIR"] = str(tmp_path / "research-logs")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not data_dir.exists()
    assert not (tmp_path / "research-logs").exists()


def test_example_does_not_replace_research_session_state():
    at = AppTest.from_file("demo.py", default_timeout=60)
    at.session_state["data_state"] = {"source": "upload", "marker": "keep"}
    at.session_state["analysis_state"] = {"marker": "keep"}
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["data_state"] == {"source": "upload", "marker": "keep"}
    assert at.session_state["analysis_state"] == {"marker": "keep"}
