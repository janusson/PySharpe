"""Micro-benchmark for weight normalisation helpers."""

from __future__ import annotations

import time

import pytest

from pysharpe.optimization import normalize_weights


@pytest.mark.benchmark
def test_normalize_weights_micro_benchmark():
    raw = {f"T{i}": float(i + 1) for i in range(32)}
    iterations = 1_000

    start = time.perf_counter()
    for _ in range(iterations):
        normalize_weights(raw)
    duration = time.perf_counter() - start

    assert duration < 1.0, f"normalize_weights too slow: {duration:.4f}s"
