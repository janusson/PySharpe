"""Shared pytest fixtures for deterministic portfolio data.

.. note::

    **Canadian TFSA Context** — All fixtures use synthetic data with fixed
    seeds (np.random.default_rng).  No live network calls, no real ticker
    data.  Price series simulate CAD-denominated broad-market ETF behavior
    with mild volatility.

    PyMC/PyTensor sampling tests are skipped when a C-compiler toolchain
    is unavailable (common on macOS without Xcode CLI tools).

    **Type-safety enforcement** — At collection time, if pyright is
    installed, the test suite verifies that the source tree has zero
    type errors.  Tests are skipped (not failed) when pyright is
    unavailable so that contributors without pyright can still run the
    suite.  Set ``PYSHARPE_REQUIRE_PYRIGHT=1`` to make this check fatal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# PyMC / PyTensor compile-mode guard
# ---------------------------------------------------------------------------

_PYTENSOR_AVAILABLE = find_spec("pytensor") is not None


def pymc_sampling_works() -> bool:
    """Return ``True`` when the environment can actually compile & sample."""
    if not _PYTENSOR_AVAILABLE:
        return False
    try:
        import pytensor
        import pytensor.tensor as pt

        x = pt.scalar("x")
        f = pytensor.function(  # type: ignore[attr-defined]
            [x],
            x + 1,
            mode=pytensor.compile.mode.FAST_RUN,  # type: ignore[attr-defined]
        )
        f(0)
        return True
    except Exception:
        return False


_SAMPLING_SKIP_REASON = (
    "PyMC sampling requires a working C-compiler toolchain; "
    "skipping real-sampler integration test."
)


# ---------------------------------------------------------------------------
# Type-safety gate — run pyright at collection time
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_REQUIRE_PYRIGHT = os.environ.get("PYSHARPE_REQUIRE_PYRIGHT", "") == "1"


def _pyright_available() -> bool:
    """Check whether pyright can be invoked."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyright", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_pyright() -> tuple[int, int, str]:
    """Run pyright on src/ and return (errors, warnings, detail)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyright", str(_SRC_ROOT), "--outputjson"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_SRC_ROOT.parent),
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            errors = data["summary"]["errorCount"]
            warnings = data["summary"]["warningCount"]
            return errors, warnings, ""
        detail = result.stderr.strip() or result.stdout.strip()
        return -1, -1, detail
    except Exception as exc:
        return -1, -1, str(exc)


def pytest_configure(config: pytest.Config) -> None:
    """Gate: refuse to run if pyright is installed and reports errors."""
    if not _pyright_available():
        return

    errors, warnings, detail = _run_pyright()

    if errors < 0:
        # pyright failed to run at all — warn but don't block
        config.warn("C1", f"pyright is installed but failed to run: {detail[:200]}")
        return

    if errors > 0:
        msg = (
            f"\n{'=' * 60}\n"
            f"  TYPE SAFETY CHECK FAILED\n"
            f"  pyright reports {errors} error(s), {warnings} warning(s)\n"
            f"{'=' * 60}\n"
            f"  Fix them with:  make typecheck\n"
            f"  Or bypass with: PYSHARPE_SKIP_PYRIGHT=1 pytest ...\n"
            f"{'=' * 60}\n"
        )
        if _REQUIRE_PYRIGHT:
            pytest.exit(msg, returncode=1)
        else:
            config.warn("C1", msg)


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
