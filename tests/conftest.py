"""Shared pytest fixtures for deterministic portfolio data.

.. note::

    **Canadian TFSA Context** — All fixtures use synthetic data with fixed
    seeds (np.random.default_rng).  No live network calls, no real ticker
    data.  Price series simulate CAD-denominated broad-market ETF behavior
    with mild volatility.

    PyMC/PyTensor sampling tests are skipped when a C-compiler toolchain
    is unavailable (common on macOS without Xcode CLI tools).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# PyMC / PyTensor compile‑mode guard
# ---------------------------------------------------------------------------
try:
    import pytensor

    _PYTENSOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYTENSOR_AVAILABLE = False


def pymc_sampling_works() -> bool:
    """Return ``True`` when the environment can actually compile & sample.

    PyTensor's C-compilation backend requires a working C++ toolchain
    (g++ or clang++) with the correct linker flags.  On macOS with
    Python 3.13 this is frequently unavailable unless Xcode CLI tools are
    installed.  Tests that exercise the real sampler are skipped when
    the toolchain is missing.
    """
    if not _PYTENSOR_AVAILABLE:
        return False
    try:
        import pytensor.tensor as pt

        x = pt.scalar("x")
        f = pytensor.function([x], x + 1, mode=pytensor.compile.mode.FAST_RUN)
        f(0)
        return True
    except Exception:
        return False


_SAMPLING_SKIP_REASON = (
    "PyMC sampling requires a working C-compiler toolchain; "
    "skipping real-sampler integration test."
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
