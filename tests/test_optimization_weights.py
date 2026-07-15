"""Tests for weight normalisation helpers.

.. note::

    **Canadian Investment Constraint** — Portfolio weights are always
    non-negative (no short-selling in TFSA accounts).  Weights must sum
    to 1.0.  These constraints are enforced at the optimizer and VA
    allocator layers.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import numpy as np
import pytest

from pysharpe.optimization import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
    normalize_weights,
)
from pysharpe.optimization.weights import ATOL, RTOL

RNG = np.random.default_rng(seed=42)


def _random_weights(count: int) -> Mapping[str, float]:
    values = RNG.random(count)
    return {f"T{i}": float(value) for i, value in enumerate(values)}


def test_normalize_weights_non_negative_output():
    raw = _random_weights(10)
    normalised = normalize_weights(raw)
    assert all(weight >= 0.0 for weight in normalised.values())


def test_normalize_weights_scale_invariant():
    raw = _random_weights(5)
    factor = RNG.uniform(0.5, 10.0)
    scaled = {key: value * factor for key, value in raw.items()}
    one = normalize_weights(raw)
    two = normalize_weights(scaled)
    np.testing.assert_allclose(
        np.fromiter(one.values(), float),
        np.fromiter(two.values(), float),
        rtol=RTOL,
        atol=ATOL,
    )


def test_normalize_weights_sum_is_one():
    raw = _random_weights(12)
    normalised = normalize_weights(raw)
    total = math.fsum(normalised.values())
    assert math.isclose(total, 1.0, rel_tol=RTOL, abs_tol=ATOL)


def test_normalize_weights_clips_negatives():
    raw = {"A": 1.0, "B": -0.5}
    normalised = normalize_weights(raw)
    assert normalised["B"] == 0.0
    assert math.isclose(
        normalised["A"],
        1.0,
        rel_tol=RTOL,
        abs_tol=ATOL,
    )


def test_normalize_weights_requires_positive_weight():
    with pytest.raises(ValueError):
        normalize_weights({"A": 0.0, "B": 0.0})


def test_normalize_weights_rejects_nan():
    with pytest.raises(ValueError):
        normalize_weights({"A": math.nan})


def test_portfolio_weights_non_zero_filters_negatives():
    weights = PortfolioWeights({"AAA": 0.6, "BBB": 0.0, "CCC": -0.1})
    assert weights.non_zero() == {"AAA": 0.6}


def test_portfolio_weights_preserve_raw_allocations():
    raw = {"AAA": 0.6, "BBB": -0.1}
    weights = PortfolioWeights(raw.copy())
    assert weights.allocations == raw


def test_portfolio_weights_normalized_helper_matches_function():
    raw = {"AAA": 2.0, "BBB": 1.0}
    weights = PortfolioWeights(raw)
    assert weights.normalized() == normalize_weights(raw)


def test_optimisation_result_summary_formats_percentages():
    performance = OptimisationPerformance(0.08, 0.15, 1.2, "2020-01-01", "2021-01-01")
    result = OptimisationResult(
        name="demo",
        weights=PortfolioWeights({"AAA": 0.6}),
        performance=performance,
    )
    summary = result.summary
    assert "expected 8.00%" in summary
    assert "volatility 15.00%" in summary
    assert "sharpe 1.20" in summary


@pytest.mark.benchmark
def test_normalize_weights_micro_benchmark():
    raw = {f"T{i}": float(i + 1) for i in range(32)}
    iterations = 1_000

    start = time.perf_counter()
    for _ in range(iterations):
        normalize_weights(raw)
    duration = time.perf_counter() - start

    assert duration < 1.0, f"normalize_weights too slow: {duration:.4f}s"
