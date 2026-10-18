"""Tests for portfolio categorisation helpers.

.. note::

    **Canadian ETF Context** — Category mapping groups correlated ETFs
    (e.g., US Equity, Fixed Income) before optimisation.  Geographic
    constraints (US, CA, INT) are applied at the optimizer level.  Test
    data uses synthetic ticker labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysharpe.analysis import apply_category_mapping, load_category_map


def test_apply_category_mapping_groups_and_retains():
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0],
            "BBB": [50.0, 51.0, 52.0],
            "CCC": [200.0, 199.0, 198.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    mapping = {"AAA": "US Equity", "BBB": "US Equity"}
    aggregation = apply_category_mapping(prices, mapping)

    assert list(aggregation.prices.columns) == ["US Equity", "CCC"]
    assert aggregation.prices.iloc[0].tolist() == [1.0, 1.0]
    assert set(aggregation.groups["US Equity"]) == {"AAA", "BBB"}
    assert aggregation.dropped == ()


def test_apply_category_mapping_drop_unmapped():
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 99.0],
            "BBB": [30.0, 30.5, 31.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    mapping = {"AAA": "Core"}

    aggregation = apply_category_mapping(prices, mapping, include_unmapped=False)

    assert list(aggregation.prices.columns) == ["Core"]
    assert aggregation.dropped == ("BBB",)


def test_load_category_map_from_path(tmp_path: Path):
    mapping_path = tmp_path / "categories.json"
    mapping_path.write_text(
        '{"AAA": "Equity", "BBB": "Fixed Income"}', encoding="utf-8"
    )

    mapping = load_category_map(mapping_path)
    assert mapping == {"AAA": "Equity", "BBB": "Fixed Income"}


def test_apply_category_mapping_requires_remaining_assets():
    prices = pd.DataFrame(
        {"AAA": [1.0, 1.1, 1.2]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    with pytest.raises(ValueError):
        apply_category_mapping(prices, {}, include_unmapped=False)


def test_no_bfill_lookahead_bias(monkeypatch):
    """
    Ensure that apply_category_mapping does not use .bfill() on the time-series dataframe,
    preventing any potential lookahead bias.
    """
    import pandas as pd

    from pysharpe.analysis.categorization import apply_category_mapping

    # Mock pd.DataFrame.bfill to raise an error if called
    def mocked_bfill(*args, **kwargs):
        raise RuntimeError("bfill() was called, which introduces lookahead bias!")

    monkeypatch.setattr(pd.DataFrame, "bfill", mocked_bfill)

    prices = pd.DataFrame(
        {"AAPL": [pd.NA, 150.0, 155.0], "MSFT": [250.0, 255.0, 260.0]},
        index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
    )
    mapping = {"AAPL": "Tech", "MSFT": "Tech"}

    # This should succeed without calling bfill
    result = apply_category_mapping(prices, mapping)

    # Verify the results match expectations
    assert list(result.prices.columns) == ["Tech"]
    assert len(result.prices) == 3

    # The first row for AAPL is missing, but MSFT is 1.0 (250/250)
    # The first category average should be based on available data
    # Wait, 1st row: AAPL is NA, MSFT is 1.0 -> Mean of Tech = 1.0
    # 2nd row: AAPL is 1.0 (150/150), MSFT is 1.02 (255/250) -> Mean = 1.01
    # 3rd row: AAPL is 1.0333 (155/150), MSFT is 1.04 (260/250) -> Mean = 1.03666

    assert pd.notna(result.prices.iloc[0, 0])
