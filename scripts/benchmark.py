#!/usr/bin/env python3
"""Runtime benchmarks for PySharpe primitives.

This script generates deterministic synthetic data and measures the runtime of
key operations across growing portfolio sizes. It focuses on:

* Metrics helpers (returns, expected returns, volatility, Sharpe ratio)
* Portfolio optimisation
* Dollar-cost averaging simulation

Usage examples::

    python scripts/benchmark.py
    python scripts/benchmark.py --repeats 5 --output benchmarks.json
    python scripts/benchmark.py --baseline old.json --output new.json

If ``--baseline`` is supplied the script prints a comparison table showing the
speed-up relative to the stored timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from pysharpe import metrics
from pysharpe.portfolio_optimization import optimise_portfolio
from pysharpe.visualization.dca import simulate_dca


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    size: int
    metric: str
    seconds: float


def _generate_prices(assets: int, days: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + assets)
    shocks = rng.normal(loc=0.0005, scale=0.01, size=(days, assets))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    columns = [f"Asset_{idx:03d}" for idx in range(assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)


def _timeit(
    func: Callable,
    *args,
    repeats: int = 3,
    warmup: int = 1,
    **kwargs,
) -> tuple[float, object]:
    for _ in range(warmup):
        func(*args, **kwargs)
    timings: list[float] = []
    last_result: object = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = func(*args, **kwargs)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), last_result


def _simulate_dca_reference(
    *,
    months: int,
    initial_investment: float,
    monthly_contribution: float,
    annual_return_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    monthly_rate = (1 + annual_return_rate) ** (1 / 12) - 1
    balances = np.zeros(months, dtype=float)
    contributions = np.zeros(months, dtype=float)
    balances[0] = initial_investment
    contributions[0] = initial_investment
    for month in range(1, months):
        contributions[month] = contributions[month - 1] + monthly_contribution
        balances[month] = (balances[month - 1] + monthly_contribution) * (1 + monthly_rate)
    months_index = np.arange(months, dtype=int)
    return months_index, contributions, balances


def benchmark_metrics(portfolio_sizes: Iterable[int], repeats: int) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for size in portfolio_sizes:
        prices = _generate_prices(size)
        duration, returns = _timeit(metrics.compute_returns, prices, repeats=repeats)
        manual_returns = prices.pct_change().dropna(how="all")
        if not returns.equals(manual_returns):
            raise AssertionError("compute_returns mismatch against manual calculation")
        results.append(BenchmarkResult("metrics", size, "compute_returns", duration))

        duration, expected = _timeit(metrics.expected_return, returns, repeats=repeats)
        manual_expected = returns.mean() * 252
        pd.testing.assert_series_equal(
            expected.sort_index(),
            manual_expected.sort_index(),
            check_names=False,
            atol=1e-12,
            rtol=0,
        )
        results.append(BenchmarkResult("metrics", size, "expected_return", duration))

        duration, volatility = _timeit(
            metrics.annualize_volatility,
            returns,
            repeats=repeats,
        )
        manual_volatility = returns.std(ddof=1) * np.sqrt(252)
        pd.testing.assert_series_equal(
            volatility.sort_index(),
            manual_volatility.sort_index(),
            check_names=False,
            atol=1e-12,
            rtol=0,
        )
        results.append(BenchmarkResult("metrics", size, "annualize_volatility", duration))

        duration, sharpe = _timeit(metrics.sharpe_ratio, returns, repeats=repeats)
        manual_sharpe = (expected - 0.0) / volatility
        pd.testing.assert_series_equal(
            sharpe.sort_index(),
            manual_sharpe.sort_index(),
            check_names=False,
            atol=1e-12,
            rtol=0,
        )
        results.append(BenchmarkResult("metrics", size, "sharpe_ratio", duration))
    return results


def benchmark_dca(month_counts: Iterable[int], repeats: int) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for months in month_counts:
        args = {
            "months": months,
            "initial_investment": 1_000.0,
            "monthly_contribution": 250.0,
            "annual_return_rate": 0.08,
        }
        duration, projection = _timeit(simulate_dca, repeats=repeats, **args)
        ref_months, ref_contrib, ref_balances = _simulate_dca_reference(**args)
        np.testing.assert_allclose(projection.months, ref_months, rtol=0, atol=0)
        np.testing.assert_allclose(projection.contributions, ref_contrib, rtol=0, atol=1e-12)
        np.testing.assert_allclose(projection.balances, ref_balances, rtol=0, atol=1e-12)
        results.append(BenchmarkResult("dca", months, "simulate_dca", duration))
    return results


def benchmark_optimisation(portfolio_sizes: Iterable[int], repeats: int) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for size in portfolio_sizes:
        prices = _generate_prices(size)
        prices.index.name = "Date"
        name = f"synthetic_{size}"
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            collated = tmp_path / f"{name}_collated.csv"
            prices.reset_index().to_csv(collated, index=False)

            duration, first = _timeit(
                optimise_portfolio,
                name,
                collated_dir=tmp_path,
                output_dir=tmp_path,
                make_plot=False,
                repeats=repeats,
            )
            duration_again, second = _timeit(
                optimise_portfolio,
                name,
                collated_dir=tmp_path,
                output_dir=tmp_path,
                make_plot=False,
                repeats=repeats,
            )
            np.testing.assert_allclose(
                first.performance.expected_return,
                second.performance.expected_return,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                first.performance.volatility,
                second.performance.volatility,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                first.performance.sharpe_ratio,
                second.performance.sharpe_ratio,
                rtol=0,
                atol=1e-12,
            )
            # Average the two timings for stability
            results.append(
                BenchmarkResult(
                    "optimisation",
                    size,
                    "optimise_portfolio",
                    statistics.mean([duration, duration_again]),
                )
            )
    return results


def _collect_results(args: argparse.Namespace) -> list[BenchmarkResult]:
    metric_sizes = [5, 25, 100]
    dca_months = [120, 360, 720]
    optimisation_sizes = [5, 25, 100]

    results = []
    results.extend(benchmark_metrics(metric_sizes, repeats=args.repeats))
    results.extend(benchmark_dca(dca_months, repeats=args.repeats))
    results.extend(benchmark_optimisation(optimisation_sizes, repeats=args.repeats))
    return results


def _results_to_dict(results: Iterable[BenchmarkResult]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for item in results:
        key = f"{item.scenario}:{item.size}:{item.metric}"
        table[key] = {
            "scenario": item.scenario,
            "size": item.size,
            "metric": item.metric,
            "seconds": item.seconds,
        }
    return table


def _print_table(results: Iterable[BenchmarkResult], baseline: dict[str, dict[str, float]] | None) -> None:
    headers = ["Scenario", "Size", "Metric", "Seconds", "Δ vs baseline", "Speed-up"]
    rows: list[list[str]] = []
    for record in sorted(results, key=lambda r: (r.scenario, r.size, r.metric)):
        key = f"{record.scenario}:{record.size}:{record.metric}"
        delta = "-"
        speedup = "-"
        if baseline and key in baseline:
            previous = baseline[key]["seconds"]
            change = record.seconds - previous
            delta = f"{change:+.4f}"
            speed = "∞" if previous == 0 else f"{previous / record.seconds:.2f}x"
            speedup = speed
        rows.append(
            [
                record.scenario,
                str(record.size),
                record.metric,
                f"{record.seconds:.6f}",
                delta,
                speedup,
            ]
        )

    widths = [max(len(row[col]) for row in [headers, *rows]) for col in range(len(headers))]

    def _format(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    print(_format(headers))
    print(" | ".join("-" * width for width in widths))
    for row in rows:
        print(_format(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PySharpe primitives.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repetitions per scenario (default: 3).")
    parser.add_argument("--baseline", type=Path, help="JSON file containing baseline timings to compare against.")
    parser.add_argument("--output", type=Path, help="Optional path to write benchmark results as JSON.")
    args = parser.parse_args()

    baseline_data: dict[str, dict[str, float]] | None = None
    if args.baseline and args.baseline.exists():
        baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))

    results = _collect_results(args)
    _print_table(results, baseline_data)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(_results_to_dict(results), args.output.open("w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
