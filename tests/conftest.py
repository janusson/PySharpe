"""Shared pytest fixtures for deterministic portfolio data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_price_series() -> pd.Series:
    """Return a synthetic price series with mild volatility."""

    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    values = pd.Series([100, 102, 101, 103, 104, 107], index=dates, name="TEST")
    return values.astype(float)


@pytest.fixture()
def sample_price_frame() -> pd.DataFrame:
    """Return a seeded price frame for multi-asset scenarios."""

    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    shocks = rng.normal(loc=0.001, scale=0.01, size=(len(dates), 3))
    cumulative = 1 + shocks
    prices = 100 * cumulative.cumprod(axis=0)
    frame = pd.DataFrame(prices, index=dates, columns=["AAA", "BBB", "CCC"])
    return frame.astype(float)


@pytest.fixture()
def ensure_directory(tmp_path: Path):
    """Factory fixture to create nested directories under tmp_path."""

    created: list[Path] = []

    def _factory(parts: Iterable[str] | None = None) -> Path:
        target = tmp_path
        for part in parts or []:
            target = target / part
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
        return target

    yield _factory

    for path in reversed(created):
        if path.exists():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
            path.rmdir()

