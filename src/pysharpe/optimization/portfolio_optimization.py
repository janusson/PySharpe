"""High-level helpers for running portfolio optimizations from stored prices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

from ..models import PortfolioAllocation, PortfolioPerformance
from .optimizer import OptimizationResult, PortfolioOptimizer
from ..data.data_collector import DEFAULT_EXPORT_DIR


class PortfolioOptimizationError(RuntimeError):
    """Raised when a portfolio cannot be optimised."""


def read_price_history(path: Path, *, index_col: str = "Date") -> pd.DataFrame:
    """Return price history stored in *path*.

    The CSV file is expected to contain one column per asset with a timestamp
    index column (default ``Date``). Missing values are forward filled before
    being returned. Any completely empty columns are dropped.
    """

    csv_path = Path(path)
    try:
        frame = pd.read_csv(csv_path, index_col=index_col, parse_dates=True)
    except FileNotFoundError as exc:  # pragma: no cover - guardrail for callers
        raise PortfolioOptimizationError(
            f"Price history file not found: {path}") from exc
    except ValueError as exc:
        raise PortfolioOptimizationError(
            f"Price history file is missing expected column '{index_col}': {path}"
        ) from exc

    if frame.empty:
        raise PortfolioOptimizationError(
            f"Price history file is empty: {path}")

    frame = frame.ffill().dropna(axis=1, how="all").sort_index()
    if frame.empty:
        raise PortfolioOptimizationError(
            f"Price history for {path} only contained missing data after cleaning."
        )

    if frame.isnull().all(axis=None):
        raise PortfolioOptimizationError(
            f"Price history for {path} contains only NaN values after forward fill."
        )

    return frame


def window_prices(
    prices: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Return *prices* restricted to the requested date window."""

    if start is None and end is None:
        return prices.copy()

    windowed = prices.loc[start:end]
    windowed = windowed.dropna(axis=1, how="all")
    if windowed.empty:
        raise PortfolioOptimizationError(
            "No price data available in the requested window.")
    return windowed


def optimize_prices(
    prices: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> OptimizationResult:
    """Run a maximum Sharpe ratio optimisation for *prices*."""

    sliced = window_prices(prices, start=start,
                           end=end) if start or end else prices
    optimizer = PortfolioOptimizer(sliced)
    return optimizer.max_sharpe(weight_bounds=weight_bounds)


def optimize_price_file(
    name: str,
    path: Path,
    *,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> OptimizationResult:
    """Convenience wrapper that reads *path* and optimises the contained prices."""

    prices = read_price_history(path)
    try:
        return optimize_prices(
            prices,
            start=start,
            end=end,
            weight_bounds=weight_bounds,
        )
    except ValueError as exc:
        raise PortfolioOptimizationError(
            f"Failed to optimise portfolio '{name}'.") from exc


def batch_optimize(
    price_files: Mapping[str, Path],
    *,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> Dict[str, OptimizationResult]:
    """Optimise several portfolios defined by *price_files*.

    ``price_files`` is a mapping of portfolio name to CSV file path. A
    ``PortfolioOptimizationError`` is raised if any portfolio fails so callers can
    decide how to handle partial success scenarios.
    """

    results: Dict[str, OptimizationResult] = {}
    for name, path in price_files.items():
        results[name] = optimize_price_file(
            name,
            Path(path),
            start=start,
            end=end,
            weight_bounds=weight_bounds,
        )
    return results


def optimize_portfolio(
    name: str,
    *,
    collated_dir: Path | str = DEFAULT_EXPORT_DIR,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    output_dir: Optional[Path | str] = DEFAULT_EXPORT_DIR,
    export_plot: bool = False,
) -> OptimizationResult:
    """Optimise a single portfolio using the legacy collated CSV naming convention."""

    collated_path = Path(collated_dir) / f"{name}_collated.csv"
    result = optimize_price_file(
        name,
        collated_path,
        start=start,
        end=end,
        weight_bounds=weight_bounds,
    )

    if output_dir is not None:
        export_result(
            name,
            result,
            Path(output_dir),
            export_plot=export_plot,
        )

    return result


def optimize_all_portfolios(
    collated_dir: Path | str = DEFAULT_EXPORT_DIR,
    *,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    output_dir: Optional[Path | str] = DEFAULT_EXPORT_DIR,
    export_plot: bool = False,
) -> Dict[str, OptimizationResult]:
    """Optimise every collated portfolio found in *collated_dir*."""

    directory = Path(collated_dir)
    price_files: Dict[str, Path] = {}
    for csv_path in directory.glob("*_collated.csv"):
        portfolio_name = csv_path.stem.replace("_collated", "")
        price_files[portfolio_name] = csv_path

    if not price_files:
        raise PortfolioOptimizationError(
            f"No collated price files found in {directory} with pattern '*_collated.csv'."
        )

    results = batch_optimize(
        price_files,
        start=start,
        end=end,
        weight_bounds=weight_bounds,
    )

    if output_dir is not None:
        for name, result in results.items():
            export_result(
                name,
                result,
                Path(output_dir),
                export_plot=export_plot,
            )

    return results


def allocation_pie(
    allocation: PortfolioAllocation,
    *,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":  # pragma: no cover - plotting code
    """Return a matplotlib pie chart for the provided *allocation*."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - defensive, depends on optional extra
        raise RuntimeError(
            "matplotlib is required to plot allocations") from exc

    weights = allocation.as_series()
    non_zero = weights[weights > 0]
    if non_zero.empty:
        raise ValueError(
            "Cannot build allocation plot without positive weights.")

    fig, ax = plt.subplots()
    ax.pie(non_zero.values, labels=non_zero.index,
           autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title(title or "Portfolio Allocation")
    return fig


def save_allocation_plot(
    allocation: PortfolioAllocation,
    destination: Path,
    *,
    title: Optional[str] = None,
    dpi: int = 96,
) -> Path:  # pragma: no cover - plotting code
    """Persist an allocation pie chart to *destination* and return the file path."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig = allocation_pie(allocation, title=title)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    fig.clf()
    return destination


def save_weights(allocation: PortfolioAllocation, destination: Path) -> Path:
    """Write portfolio weights to *destination* in the original text layout."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    weights = allocation.as_series()
    lines = ["ticker,weight"]
    lines.extend(f"{ticker},{weight:.8f}" for ticker,
                 weight in weights.items())
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def save_performance(performance: PortfolioPerformance, destination: Path) -> Path:
    """Write portfolio performance metrics to *destination* matching the legacy format."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Expected annual return: {performance.expected_annual_return * 100:.2f}%",
        f"Annual volatility: {performance.annual_volatility * 100:.2f}%",
        f"Sharpe Ratio: {performance.sharpe_ratio:.2f}",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


@dataclass
class ExportArtifacts:
    """Represents the files created when exporting optimisation results."""

    weights_file: Optional[Path] = None
    performance_file: Optional[Path] = None
    allocation_plot: Optional[Path] = None


def export_result(
    name: str,
    result: OptimizationResult,
    output_dir: Path,
    *,
    export_weights: bool = True,
    export_performance: bool = True,
    export_plot: bool = False,
    plot_dpi: int = 96,
) -> ExportArtifacts:
    """Persist optimisation *result* under *output_dir* and return created files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path: Optional[Path] = None
    performance_path: Optional[Path] = None
    plot_path: Optional[Path] = None

    if export_weights:
        weights_path = save_weights(
            result.allocation,
            output_dir / f"{name}_weights.txt",
        )

    if export_performance:
        performance_path = save_performance(
            result.performance,
            output_dir / f"{name}_performance.txt",
        )

    if export_plot:
        plot_path = save_allocation_plot(
            result.allocation,
            output_dir / f"{name}_allocation.png",
            title=f"{name} Allocation",
            dpi=plot_dpi,
        )

    return ExportArtifacts(
        weights_file=weights_path,
        performance_file=performance_path,
        allocation_plot=plot_path,
    )


def _create_parser():  # pragma: no cover - CLI plumbing
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimise one or more portfolios from collated price histories.",
    )
    parser.add_argument(
        "price_files",
        metavar="PORTFOLIO=PATH",
        nargs="+",
        help="Mapping of portfolio name to CSV path, e.g. growth=./data/growth.csv",
    )
    parser.add_argument("--start", help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Optional end date (YYYY-MM-DD).")
    parser.add_argument(
        "--weight-bounds",
        metavar="LOW,HIGH",
        help="Optional weight bounds applied to all assets (default 0.0,1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory to write results. When omitted, no files are created.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create allocation plots when an output directory is provided.",
    )
    return parser


def _parse_weight_bounds(bounds_spec: Optional[str]) -> tuple[float, float]:  # pragma: no cover
    if not bounds_spec:
        return 0.0, 1.0
    try:
        low_str, high_str = bounds_spec.split(",", 1)
        return float(low_str), float(high_str)
    except Exception as exc:  # broad except to surface user input issues cleanly
        raise PortfolioOptimizationError(
            "--weight-bounds must be provided as 'LOW,HIGH'"
        ) from exc


def _parse_price_arguments(arguments: Iterable[str]) -> Dict[str, Path]:  # pragma: no cover
    mapping: Dict[str, Path] = {}
    for item in arguments:
        if "=" not in item:
            raise PortfolioOptimizationError(
                "Each price file argument must be in the form 'NAME=PATH'."
            )
        name, path = item.split("=", 1)
        mapping[name.strip()] = Path(path).expanduser()
    return mapping


def main(argv: Optional[Iterable[str]] = None) -> int:  # pragma: no cover - CLI plumbing
    parser = _create_parser()
    args = parser.parse_args(argv)

    try:
        weight_bounds = _parse_weight_bounds(args.weight_bounds)
        price_files = _parse_price_arguments(args.price_files)
    except PortfolioOptimizationError as exc:
        parser.error(str(exc))

    results = batch_optimize(
        price_files,
        start=args.start,
        end=args.end,
        weight_bounds=weight_bounds,
    )

    for name, result in results.items():
        print(f"Portfolio: {name}")
        print(result.performance.as_dict())

    if args.output is not None:
        for name, result in results.items():
            export_result(
                name,
                result,
                args.output,
                export_plot=args.plot,
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
