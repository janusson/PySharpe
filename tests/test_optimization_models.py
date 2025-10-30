"""Tests for optimisation dataclasses."""

from __future__ import annotations

from pysharpe.optimization import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
    normalize_weights,
)


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
    performance = OptimisationPerformance(0.08, 0.15, 1.2)
    result = OptimisationResult(
        name="demo",
        weights=PortfolioWeights({"AAA": 0.6}),
        performance=performance,
    )

    summary = result.summary
    assert "expected 8.00%" in summary
    assert "volatility 15.00%" in summary
    assert "sharpe 1.20" in summary
