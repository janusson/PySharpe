"""Portfolio optimisation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

from pysharpe.config import get_settings
from pysharpe.optimization.models import (
    OptimisationPerformance,
    OptimisationResult,
    PortfolioWeights,
)

_SETTINGS = get_settings()
EXPORT_DIR = Path(_SETTINGS.export_dir)

logger = logging.getLogger(__name__)


def _load_collated_prices(
    portfolio_name: str,
    collated_dir: Path,
    *,
    time_constraint: Optional[str] = None,
) -> pd.DataFrame:
    csv_path = Path(collated_dir) / f"{portfolio_name}_collated.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Collated prices not found for {portfolio_name}: {csv_path}")

    frame = pd.read_csv(csv_path, parse_dates=True, index_col="Date")
    if time_constraint:
        frame = frame.sort_index()
        frame = frame.loc[time_constraint:]

    if frame.empty:
        raise ValueError(f"No data available for {portfolio_name} after applying constraint")

    if frame.isnull().values.any():
        frame = frame.ffill()

    return frame


def _require_matplotlib():  # pragma: no cover - optional dependency
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - raised when missing
        raise RuntimeError(
            "matplotlib must be installed to generate allocation plots."
        ) from exc
    return plt


def _prepare_output_dir(path: Path) -> Path:
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _plot_allocation(result: OptimisationResult, output_dir: Path) -> Path:
    weights = result.weights.non_zero()
    if not weights:
        raise ValueError("No positive weights to plot")

    plt = _require_matplotlib()
    target_dir = _prepare_output_dir(output_dir)
    output_path = target_dir / f"{result.name}_allocation.png"

    fig, ax = plt.subplots()
    ax.pie(weights.values(), labels=weights.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title(f"{result.name} Allocation")
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_weights(result: OptimisationResult, output_dir: Path) -> Path:
    target_dir = _prepare_output_dir(output_dir)
    output_path = target_dir / f"{result.name}_weights.txt"

    lines = ["ticker,weight"]
    for ticker, weight in result.weights.allocations.items():
        lines.append(f"{ticker},{weight:.8f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _save_performance(result: OptimisationResult, output_dir: Path) -> Path:
    target_dir = _prepare_output_dir(output_dir)
    output_path = target_dir / f"{result.name}_performance.txt"

    perf = result.performance
    lines = [
        f"Expected annual return: {perf.expected_return * 100:.2f}%",
        f"Annual volatility: {perf.volatility * 100:.2f}%",
        f"Sharpe Ratio: {perf.sharpe_ratio:.2f}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def optimise_portfolio(
    portfolio_name: str,
    *,
    collated_dir: Path = EXPORT_DIR,
    output_dir: Path = EXPORT_DIR,
    time_constraint: Optional[str] = None,
    asset_constraints: Optional[Dict[str, float]] = None,
    geo_exposure: Optional[Iterable] = None,  # placeholder for future logic
    make_plot: bool = True,
) -> OptimisationResult:
    """Optimise a portfolio using the PyPortfolioOpt max Sharpe workflow."""

    prices = _load_collated_prices(portfolio_name, collated_dir, time_constraint=time_constraint)

    mu = mean_historical_return(prices)
    try:
        cov = CovarianceShrinkage(prices).ledoit_wolf()
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - sklearn optional
        cov = prices.pct_change().dropna().cov()
    ef = EfficientFrontier(mu, cov)

    if asset_constraints:
        if "min_weight" in asset_constraints:
            ef.add_constraint(lambda w: w >= asset_constraints["min_weight"])
        if "max_weight" in asset_constraints:
            ef.add_constraint(lambda w: w <= asset_constraints["max_weight"])

    if geo_exposure:  # pragma: no cover - placeholder for parity with original script
        for _ in geo_exposure:
            pass

    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    expected, volatility, sharpe = ef.portfolio_performance(verbose=False)

    result = OptimisationResult(
        name=portfolio_name,
        weights=PortfolioWeights(cleaned_weights),
        performance=OptimisationPerformance(expected, volatility, sharpe),
    )

    _save_weights(result, output_dir)
    _save_performance(result, output_dir)

    if make_plot:
        try:
            _plot_allocation(result, output_dir)
        except RuntimeError as exc:
            logger.warning("Skipping allocation plot for %s: %s", portfolio_name, exc)

    return result


def optimise_all_portfolios(
    collated_dir: Path = EXPORT_DIR,
    *,
    output_dir: Path = EXPORT_DIR,
    time_constraint: Optional[str] = None,
) -> Dict[str, OptimisationResult]:
    """Optimise every collated portfolio in *collated_dir*."""

    results: Dict[str, OptimisationResult] = {}

    for path in Path(collated_dir).glob("*_collated.csv"):
        name = path.stem.replace("_collated", "")
        results[name] = optimise_portfolio(
            name,
            collated_dir=collated_dir,
            output_dir=output_dir,
            time_constraint=time_constraint,
        )

    return results


def main() -> None:  # pragma: no cover - interactive legacy flow
    collated_dir = EXPORT_DIR
    output_dir = EXPORT_DIR

    portfolio_files = [path for path in Path(_SETTINGS.portfolio_dir).glob("*.csv")]
    print(f"Found {len(portfolio_files)} portfolios:")
    for csv_path in portfolio_files:
        try:
            df = pd.read_csv(csv_path)
            print(f"- {csv_path.stem}: {len(df.index)} equities")
        except Exception as exc:
            print(f"  Unable to read {csv_path}: {exc}")

    proceed = input("Do you want to optimize all portfolios found? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("Optimization aborted.")
        return

    for csv_path in portfolio_files:
        name = csv_path.stem
        try:
            optimise_portfolio(name, collated_dir=collated_dir, output_dir=output_dir, time_constraint="1980-01-01")
            print(f"Optimised portfolio: {name}")
        except Exception as exc:
            print(f"Error optimising {name}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
