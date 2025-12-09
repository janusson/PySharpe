"""Command line interface for PySharpe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from pysharpe import data_collector, workflows
from pysharpe.analysis import load_category_map
from pysharpe.config import get_settings
from pysharpe.data import PortfolioRepository
from pysharpe.optimization.models import OptimisationResult
from pysharpe.visualization import plot_dca_projection, simulate_dca
from pysharpe.visualization import utils as viz_utils


def _resolve(path: Path | str | None, default: Path) -> Path:
    """Expand user paths and fall back to a default when *path* is ``None``."""

    if path is None:
        return default
    return Path(path).expanduser().resolve()


def _summarise_results(results: Iterable[OptimisationResult]) -> None:
    """Print optimisation summaries in a stable order."""

    for outcome in sorted(results, key=lambda item: item.name):
        perf = outcome.performance
        print(
            f"{outcome.name}: return {perf.expected_return:.2%}, "
            f"volatility {perf.volatility:.2%}, sharpe {perf.sharpe_ratio:.2f}"
        )


def _handle_optimise(args: argparse.Namespace) -> int:
    settings = get_settings()
    portfolio_dir = _resolve(args.portfolio_dir, settings.portfolio_dir)
    price_dir = _resolve(args.price_dir, settings.price_history_dir)
    export_dir = _resolve(args.export_dir, settings.export_dir)

    portfolio_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        data_collector.setup_logging(Path(args.log_dir))

    repo = PortfolioRepository(settings, directory=portfolio_dir)
    available = {definition.name: definition for definition in repo.list_portfolios()}
    if not available:
        print("No portfolios discovered. Add CSV files to the portfolio directory.")
        return 1

    if args.portfolios:
        missing = [name for name in args.portfolios if name not in available]
        if missing:
            print("Unknown portfolio names:")
            for item in missing:
                print(f"  - {item}")
        targets = [available[name] for name in args.portfolios if name in available]
    else:
        targets = list(available.values())

    if not targets:
        print("No valid portfolios selected.")
        return 1

    target_names = [definition.name for definition in targets]

    if not args.skip_download:
        workflows.download_portfolios(
            portfolio_names=target_names,
            portfolio_dir=portfolio_dir,
            price_history_dir=price_dir,
            export_dir=export_dir,
            period=args.period,
            interval=args.interval,
            start=args.start,
            end=args.end,
        )

    include_unmapped = not args.drop_unmapped_categories
    category_map: dict[str, str] | None = None
    if args.category_map:
        try:
            category_map = load_category_map(args.category_map)
        except ValueError as exc:
            print(f"Unable to load category map: {exc}")
            return 1
    elif args.use_default_categories:
        category_map = load_category_map()

    results = workflows.optimise_portfolios(
        portfolio_names=target_names,
        collated_dir=export_dir,
        output_dir=export_dir,
        time_constraint=args.time_constraint,
        make_plot=not args.no_plot,
        category_map=category_map,
        include_unmapped_categories=include_unmapped,
    )

    if not results:
        print("No optimisation results produced. Ensure collated data exists.")
        return 1

    _summarise_results(results.values())
    print(f"Artefacts written to {export_dir}")
    return 0


def _handle_simulate_dca(args: argparse.Namespace) -> int:
    projection = simulate_dca(
        months=args.months,
        initial_investment=args.initial,
        monthly_contribution=args.monthly,
        annual_return_rate=args.rate,
    )

    total_contribution = projection.final_contribution()
    final_balance = projection.final_balance()
    print(f"Final contribution: ${total_contribution:,.2f}")
    print(f"Final balance:      ${final_balance:,.2f}")

    if args.plot or args.output:
        ax = plot_dca_projection(projection)
        if args.output:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(output_path)
            print(f"Plot saved to {output_path}")
        if args.plot:
            viz_utils.require_matplotlib().show()
        else:
            ax.figure.clf()

    return 0


def _handle_plot(args: argparse.Namespace) -> int:
    path = Path(args.input).expanduser()
    if not path.exists():
        print(f"Input file not found: {path}")
        return 1

    frame = pd.read_csv(path)
    if args.date_column and args.date_column in frame.columns:
        frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
        frame.set_index(args.date_column, inplace=True)

    columns = args.columns or frame.select_dtypes("number").columns.tolist()
    if not columns:
        print("No numeric columns available to plot.")
        return 1

    ax = frame[columns].plot(figsize=(8, 4))
    ax.set_title(args.title or path.stem)
    ax.set_ylabel(args.ylabel)
    ax.grid(True)

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output)
        print(f"Plot saved to {output}")

    if args.show:
        viz_utils.require_matplotlib().show()
    else:
        ax.figure.clf()

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysharpe",
        description="Portfolio analytics helpers for optimisation and simulations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # optimise
    optimise = subparsers.add_parser(
        "optimise",
        help="Download (optional) data and run the optimisation pipeline.",
    )
    optimise.add_argument("--portfolio", dest="portfolios", nargs="*", help="Portfolio names to target.")
    optimise.add_argument("--portfolio-dir", type=Path, help="Directory containing portfolio CSV files.")
    optimise.add_argument("--price-dir", type=Path, help="Directory for price history CSVs.")
    optimise.add_argument("--export-dir", type=Path, help="Directory for collated data and outputs.")
    optimise.add_argument("--log-dir", type=Path, help="Optional logging directory.")
    optimise.add_argument("--period", default="max", help="History period requested when start/end are omitted.")
    optimise.add_argument("--interval", default="1d", help="Sampling interval for downloads (default: 1d).")
    optimise.add_argument("--start", help="ISO start date for downloads.")
    optimise.add_argument("--end", help="ISO end date for downloads.")
    optimise.add_argument("--time-constraint", dest="time_constraint", help="ISO start date applied to collated data.")
    optimise.add_argument("--skip-download", action="store_true", help="Reuse previously downloaded data.")
    optimise.add_argument("--no-plot", action="store_true", help="Skip allocation pie charts.")
    optimise.add_argument(
        "--category-map",
        type=Path,
        help="Optional path to a JSON mapping of ticker -> category.",
    )
    optimise.add_argument(
        "--use-default-categories",
        action="store_true",
        help="Load the default category mapping from the PySharpe info directory.",
    )
    optimise.add_argument(
        "--drop-unmapped-categories",
        action="store_true",
        help="Discard tickers missing a category instead of treating them as standalone categories.",
    )

    # simulate-dca
    simulate = subparsers.add_parser(
        "simulate-dca",
        help="Generate a dollar-cost averaging projection.",
    )
    simulate.add_argument("--months", type=int, default=36, help="Number of months to simulate (default: 36).")
    simulate.add_argument("--initial", type=float, default=1000.0, help="Initial lump sum investment.")
    simulate.add_argument("--monthly", type=float, default=250.0, help="Recurring contribution amount.")
    simulate.add_argument("--rate", type=float, default=0.08, help="Annual return rate as a decimal (default: 0.08).")
    simulate.add_argument("--output", help="Optional path to save the generated plot.")
    simulate.add_argument("--plot", action="store_true", help="Display the plot interactively.")

    # plot
    plot = subparsers.add_parser(
        "plot",
        help="Plot numeric series from a CSV file (collated data or performance logs).",
    )
    plot.add_argument("--input", required=True, help="Path to the CSV file to visualise.")
    plot.add_argument("--columns", nargs="*", help="Specific column names to plot (defaults to numeric columns).")
    plot.add_argument("--date-column", help="Name of a column to use as the datetime index.")
    plot.add_argument("--title", help="Custom plot title.")
    plot.add_argument("--ylabel", default="Value", help="Y axis label (default: Value).")
    plot.add_argument("--output", help="Optional file path for saving the plot.")
    plot.add_argument("--show", action="store_true", help="Display the plot interactively.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``python -m pysharpe.cli`` and console scripts."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "optimise":
        return _handle_optimise(args)
    if args.command == "simulate-dca":
        return _handle_simulate_dca(args)
    if args.command == "plot":
        return _handle_plot(args)

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
