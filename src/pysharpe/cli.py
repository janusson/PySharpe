"""Simplified command line interface for PySharpe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from pysharpe import data_collector, workflows
    from pysharpe.config import get_settings
    from pysharpe.data import PortfolioDefinition, PortfolioRepository
    from pysharpe.optimization.models import OptimisationResult
except ImportError:  # pragma: no cover - support running as a script
    import sys

    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from pysharpe import data_collector, workflows  # type: ignore
    from pysharpe.config import get_settings  # type: ignore
    from pysharpe.data import PortfolioDefinition, PortfolioRepository  # type: ignore
    from pysharpe.optimization.models import OptimisationResult  # type: ignore


DEFAULT_PERIOD = "max"
DEFAULT_INTERVAL = "1d"


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _print_optimisation_results(results: Iterable[OptimisationResult]) -> None:
    for result in sorted(results, key=lambda item: item.name):
        perf = result.performance
        print(
            f"Optimised {result.name}: expected return {perf.expected_return:.2%}, "
            f"volatility {perf.volatility:.2%}, sharpe {perf.sharpe_ratio:.2f}"
        )
        weights = result.weights.non_zero()
        if weights:
            allocations = ", ".join(
                f"{ticker}={weight:.2%}" for ticker, weight in sorted(weights.items())
            )
            print(f"  Weights: {allocations}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysharpe",
        description="Download price history and optimise portfolios in one step.",
    )
    parser.add_argument(
        "--portfolio",
        dest="portfolios",
        nargs="*",
        help="Optional portfolio names to target (defaults to all discovered portfolios).",
    )
    parser.add_argument(
        "--portfolio-dir",
        help=(
            "Directory containing portfolio CSVs (default:"
            f" {data_collector.PORTFOLIO_DIR})"
        ),
    )
    parser.add_argument(
        "--price-dir",
        help=(
            "Directory for individual price history CSV files (default:"
            f" {data_collector.PRICE_HISTORY_DIR})"
        ),
    )
    parser.add_argument(
        "--export-dir",
        help=(
            "Directory for collated price files and optimisation artefacts (default:"
            f" {data_collector.EXPORT_DIR})"
        ),
    )
    parser.add_argument(
        "--log-dir",
        help="Optional directory for log files; enables file logging if provided.",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help=f"yfinance period passed to history() when no explicit dates are supplied (default: {DEFAULT_PERIOD}).",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help=f"Sampling interval passed to yfinance history() (default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--start",
        help="Optional ISO date marking the beginning of the download window.",
    )
    parser.add_argument(
        "--end",
        help="Optional ISO date marking the end of the download window.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume price history has already been downloaded and collated.",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip the optimisation step and stop after downloading data.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating allocation pie charts during optimisation.",
    )
    return parser


def _select_portfolios(
    repo: PortfolioRepository, names: Sequence[str] | None
) -> List[PortfolioDefinition]:
    available = {definition.name: definition for definition in repo.list_portfolios()}
    if not available:
        return []

    if not names:
        return list(available.values())

    selections = []
    missing: list[str] = []
    for name in names:
        definition = available.get(name)
        if definition is None:
            missing.append(name)
        else:
            selections.append(definition)

    if missing:
        print("The following portfolio names were not found:")
        for name in missing:
            print(f"  - {name}")

    return selections


def _ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()

    portfolio_dir = _resolve_path(args.portfolio_dir) or Path(data_collector.PORTFOLIO_DIR)
    price_dir = _resolve_path(args.price_dir) or Path(data_collector.PRICE_HISTORY_DIR)
    export_dir = _resolve_path(args.export_dir) or Path(data_collector.EXPORT_DIR)
    log_dir = _resolve_path(args.log_dir)

    _ensure_directories(portfolio_dir, price_dir, export_dir)

    if log_dir is not None:
        data_collector.setup_logging(log_dir)

    repo = PortfolioRepository(settings, directory=portfolio_dir)
    available = repo.list_portfolios()
    print(f"Portfolio definitions directory: {portfolio_dir}")

    if not available:
        print("No portfolio CSV files found. Add one ticker per line to create a portfolio.")
        return 1

    print("Available portfolios:")
    for definition in available:
        print(f"  - {definition.name} ({len(definition.tickers)} tickers)")

    selected = _select_portfolios(repo, args.portfolios)
    if not selected:
        print("No valid portfolios selected.")
        return 1

    target_names = [definition.name for definition in selected]

    if args.skip_download:
        print("Skipping download step (per --skip-download).")
    else:
        print(
            f"Downloading and collating price history to {export_dir}"
            f" (raw data in {price_dir})."
        )
        processed = workflows.download_portfolios(
            portfolio_names=target_names,
            portfolio_dir=portfolio_dir,
            price_history_dir=price_dir,
            export_dir=export_dir,
            period=args.period,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )
        if not processed:
            print("No portfolios processed during download.")
        else:
            for name in sorted(processed):
                print(f"  Collated data written to {export_dir / f'{name}_collated.csv'}")

    if args.skip_optimize:
        print("Skipping optimisation step (per --skip-optimize).")
        return 0

    print(f"Optimising portfolios using data in {export_dir}.")
    results = workflows.optimise_portfolios(
        portfolio_names=target_names,
        collated_dir=export_dir,
        output_dir=export_dir,
        time_constraint=None,
        make_plot=not args.no_plot,
    )
    if not results:
        print("No portfolios optimised.")
        return 1

    _print_optimisation_results(results.values())
    print(f"Optimisation artefacts saved to {export_dir}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
