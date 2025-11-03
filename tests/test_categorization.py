"""Tests for portfolio categorisation helpers."""

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
    mapping_path.write_text('{"AAA": "Equity", "BBB": "Fixed Income"}', encoding="utf-8")

    mapping = load_category_map(mapping_path)
    assert mapping == {"AAA": "Equity", "BBB": "Fixed Income"}


def test_apply_category_mapping_requires_remaining_assets():
    prices = pd.DataFrame(
        {"AAA": [1.0, 1.1, 1.2]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    with pytest.raises(ValueError):
        apply_category_mapping(prices, {}, include_unmapped=False)
