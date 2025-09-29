"""Portfolio optimisation helpers mirroring the original script."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

try:  # Allow running as a script without installing the package.
    from .data_collector import EXPORT_DIR
except ImportError:
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(PACKAGE_ROOT.parent))
    from pysharpe.data_collector import EXPORT_DIR  # type: ignore  # noqa: E402


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


def _plot_allocation(weights: Dict[str, float], portfolio_name: str, output_dir: Path) -> Path:
    filtered = {asset: weight for asset, weight in weights.items() if weight > 0}
    if not filtered:
        raise ValueError("No positive weights to plot")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{portfolio_name}_allocation.png"

    fig, ax = plt.subplots()
    ax.pie(filtered.values(), labels=filtered.keys(), autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    plt.title(f"{portfolio_name} Allocation")
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_weights(weights: Dict[str, float], portfolio_name: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{portfolio_name}_weights.txt"

    lines = ["ticker,weight"]
    for ticker, weight in weights.items():
        lines.append(f"{ticker},{weight:.8f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _save_performance(perf: Tuple[float, float, float], portfolio_name: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{portfolio_name}_performance.txt"

    expected, volatility, sharpe = perf
    lines = [
        f"Expected annual return: {expected * 100:.2f}%",
        f"Annual volatility: {volatility * 100:.2f}%",
        f"Sharpe Ratio: {sharpe:.2f}",
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
) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
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
    performance = ef.portfolio_performance(verbose=False)

    _save_weights(cleaned_weights, portfolio_name, output_dir)
    _save_performance(performance, portfolio_name, output_dir)

    if make_plot:
        _plot_allocation(cleaned_weights, portfolio_name, output_dir)

    return cleaned_weights, performance


def optimise_all_portfolios(
    collated_dir: Path = EXPORT_DIR,
    *,
    output_dir: Path = EXPORT_DIR,
    time_constraint: Optional[str] = None,
) -> Dict[str, Tuple[Dict[str, float], Tuple[float, float, float]]]:
    """Optimise every collated portfolio in *collated_dir*."""

    results: Dict[str, Tuple[Dict[str, float], Tuple[float, float, float]]] = {}

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

    portfolio_files = [path for path in Path("data/portfolio").glob("*.csv")]
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
