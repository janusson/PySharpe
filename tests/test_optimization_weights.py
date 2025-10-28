"""Tests for weight normalisation helpers."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pytest

from pysharpe.optimization import normalize_weights
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
