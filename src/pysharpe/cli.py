"""Command line interface for PySharpe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

try:
    from pysharpe import data_collector, portfolio_optimization
except ImportError:  # pragma: no cover - support running as a script
    import sys
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from pysharpe import data_collector, portfolio_optimization  # type: ignore



def _with_default_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _handle_download(args: argparse.Namespace) -> int:
    portfolio_dir = _with_default_path(args.portfolio_dir)
    price_dir = _with_default_path(args.price_dir)
    export_dir = _with_default_path(args.export_dir)

    if args.log_dir:
        data_collector.setup_logging(_with_default_path(args.log_dir))

    target_portfolios: Sequence[str] | None = args.portfolios
    period: str = args.period
    interval: str = args.interval
    start: str | None = args.start
    end: str | None = args.end

    if target_portfolios:
        processed = {}
        for name in target_portfolios:
            portfolio_file = _resolve_portfolio_file(name, portfolio_dir)
            frame = data_collector.process_portfolio(
                portfolio_file,
                price_history_dir=price_dir,
                export_dir=export_dir,
                period=period,
                interval=interval,
                start=start,
                end=end,
            )
            if not frame.empty:
                processed[portfolio_file.stem] = frame
        portfolio_names = sorted(processed.keys())
    else:
        processed = data_collector.process_all_portfolios(
            portfolio_dir,
            price_history_dir=price_dir,
            export_dir=export_dir,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        portfolio_names = sorted(processed.keys())

    if not portfolio_names:
        print("No portfolios processed")
        return 1

    for name in portfolio_names:
        print(f"Collated prices exported for: {name}")

    return 0


def _handle_optimize(args: argparse.Namespace) -> int:
    collated_dir = _with_default_path(args.collated_dir)
    output_dir = _with_default_path(args.output_dir)
    time_constraint: str | None = args.start
    make_plot = not args.no_plot

    target_portfolios: Iterable[str]
    if args.portfolios:
        target_portfolios = [name for name in args.portfolios]
    else:
        target_portfolios = _discover_collated_names(collated_dir)

    processed_any = False
    for name in target_portfolios:
        try:
            weights, performance = portfolio_optimization.optimise_portfolio(
                name,
                collated_dir=collated_dir,
                output_dir=output_dir,
                time_constraint=time_constraint,
                make_plot=make_plot,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")
            continue
        except ValueError as exc:
            print(f"Skipping {name}: {exc}")
            continue

        expected, volatility, sharpe = performance
        processed_any = True
        print(
            f"Optimised {name}: expected return {expected:.2%}, "
            f"volatility {volatility:.2%}, sharpe {sharpe:.2f}"
        )
        non_zero = {ticker: weight for ticker, weight in weights.items() if weight > 0}
        if non_zero:
            allocations = ", ".join(
                f"{ticker}={weight:.2%}" for ticker, weight in sorted(non_zero.items())
            )
            print(f"  Weights: {allocations}")

    if not processed_any:
        print("No portfolios optimised")
        return 1

    return 0


def _resolve_portfolio_file(name: str, directory: Path) -> Path:
    candidate = Path(name)
    if candidate.exists() and candidate.is_file():
        return candidate

    if not name.endswith(".csv"):
        candidate = directory / f"{name}.csv"
    else:
        candidate = directory / name

    if not candidate.exists():
        raise FileNotFoundError(f"Portfolio file not found: {candidate}")

    return candidate


def _discover_collated_names(collated_dir: Path) -> list[str]:
    return sorted(
        path.stem.replace("_collated", "")
        for path in collated_dir.glob("*_collated.csv")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pysharpe", description="PySharpe command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download",
        help="Download price histories and collate portfolio data",
    )
    download.add_argument(
        "--portfolio-dir",
        default=data_collector.PORTFOLIO_DIR,
        help="Directory containing portfolio CSV definitions",
    )
    download.add_argument(
        "--price-dir",
        default=data_collector.PRICE_HISTORY_DIR,
        help="Directory for individual price history CSV files",
    )
    download.add_argument(
        "--export-dir",
        default=data_collector.EXPORT_DIR,
        help="Directory for collated price CSV files",
    )
    download.add_argument(
        "--period",
        default="max",
        help="Time period passed to yfinance history()",
    )
    download.add_argument(
        "--interval",
        default="1d",
        help="Sampling interval passed to yfinance history()",
    )
    download.add_argument(
        "--start",
        help="Optional ISO date to begin the download window",
    )
    download.add_argument(
        "--end",
        help="Optional ISO date to end the download window",
    )
    download.add_argument(
        "--log-dir",
        help="Optional directory for log files; enables file logging if provided",
    )
    download.add_argument(
        "portfolios",
        nargs="*",
        help="Optional portfolio names or CSV files to process (defaults to all)",
    )
    download.set_defaults(func=_handle_download)

    optimise = subparsers.add_parser(
        "optimize",
        help="Optimise collated portfolios to maximise Sharpe ratio",
    )
    optimise.add_argument(
        "--collated-dir",
        default=data_collector.EXPORT_DIR,
        help="Directory containing collated price CSV files",
    )
    optimise.add_argument(
        "--output-dir",
        default=data_collector.EXPORT_DIR,
        help="Directory where optimisation artefacts are written",
    )
    optimise.add_argument(
        "--start",
        "--time-constraint",
        dest="start",
        help="Filter collated data to rows on/after the provided ISO date",
    )
    optimise.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the allocation pie chart",
    )
    optimise.add_argument(
        "portfolios",
        nargs="*",
        help="Optional portfolio names to optimise (defaults to all collated files)",
    )
    optimise.set_defaults(func=_handle_optimize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
