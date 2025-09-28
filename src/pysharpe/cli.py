"""Command line interface for PySharpe."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .data.data_collector import (
    DEFAULT_PORTFOLIO_DIR,
    DEFAULT_PRICE_HISTORY_DIR,
    PortfolioTickerReader,
    collate_prices,
    get_csv_file_paths,
    process_portfolio,
)
from .models import PortfolioPerformance
from .optimization import portfolio_optimization as p_opt


logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected ISO format YYYY-MM-DD."
        ) from exc


def _parse_weight_bounds(spec: Optional[str]) -> tuple[float, float]:
    if not spec:
        return 0.0, 1.0
    try:
        low_str, high_str = spec.split(",", 1)
        return float(low_str), float(high_str)
    except Exception as exc:  # pragma: no cover - user input validation
        raise argparse.ArgumentTypeError("Weight bounds must be provided as LOW,HIGH.") from exc


def _handle_download(args: argparse.Namespace) -> int:
    portfolio_dir: Path = args.portfolio_dir
    price_dir: Path = args.price_dir

    csv_files = get_csv_file_paths(portfolio_dir)
    if not csv_files:
        logger.warning("No portfolio definition files found in %s", portfolio_dir)
        return 1

    logger.info("Downloading price history for %d portfolio(s).", len(csv_files))
    total_symbols = 0
    for csv_path in csv_files:
        tickers = process_portfolio(
            csv_path,
            price_history_dir=price_dir,
            start=args.start,
            end=args.end,
            interval=args.interval,
        )
        if not tickers:
            logger.warning("No tickers processed for %s", csv_path.name)
            continue
        total_symbols += len(tickers)
        logger.info("%s: downloaded %d ticker(s)", csv_path.stem, len(tickers))

    logger.info("Completed downloads for %d ticker(s).", total_symbols)
    return 0 if total_symbols else 1


def _collate_portfolio_prices(
    portfolio_name: str,
    tickers: Iterable[str],
    price_dir: Path,
) -> pd.DataFrame:
    csv_files = [price_dir / f"{ticker}_hist.csv" for ticker in tickers]
    return collate_prices(portfolio_name, csv_files, tickers)


def _save_collated_prices(prices: pd.DataFrame, output_dir: Optional[Path], name: str) -> Optional[Path]:
    if output_dir is None or prices.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.csv"
    prices.to_csv(output_path, index=True)
    return output_path


def _print_performance(name: str, performance: PortfolioPerformance) -> None:
    stats = performance.as_dict()
    logger.info(
        "%s -> return: %.2f%% | volatility: %.2f%% | sharpe: %.2f",
        name,
        stats["expected_annual_return"] * 100,
        stats["annual_volatility"] * 100,
        stats["sharpe_ratio"],
    )


def _handle_optimize(args: argparse.Namespace) -> int:
    portfolio_dir: Path = args.portfolio_dir
    price_dir: Path = args.price_dir
    collated_dir: Optional[Path] = args.collated_dir

    reader = PortfolioTickerReader(portfolio_dir)
    if not reader.portfolio_tickers:
        logger.error("No portfolios loaded from %s", portfolio_dir)
        return 1

    weight_bounds = args.weight_bounds
    results: dict[str, tuple[PortfolioAllocation, PortfolioPerformance]] = {}

    for name, tickers in reader.portfolio_tickers.items():
        prices = _collate_portfolio_prices(name, tickers, price_dir)
        if prices.empty:
            logger.warning("Skipping %s: no price data found.", name)
            continue

        _save_collated_prices(prices, collated_dir, name)

        try:
            optimization = p_opt.optimize_prices(
                prices,
                start=args.start,
                end=args.end,
                weight_bounds=weight_bounds,
            )
        except Exception as exc:  # pragma: no cover - surfaces optimisation issues nicely
            logger.error("Failed to optimise %s: %s", name, exc)
            continue

        _print_performance(name, optimization.performance)

        if args.output:
            p_opt.export_result(
                name,
                optimization,
                args.output,
                export_plot=args.plot,
            )

        results[name] = (optimization.allocation, optimization.performance)

    if not results:
        return 1

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PySharpe command line interface")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download price histories")
    download_parser.add_argument(
        "--portfolio-dir",
        type=Path,
        default=DEFAULT_PORTFOLIO_DIR,
        help="Directory containing portfolio definition CSV files.",
    )
    download_parser.add_argument(
        "--price-dir",
        type=Path,
        default=DEFAULT_PRICE_HISTORY_DIR,
        help="Directory to write per-ticker price history CSV files.",
    )
    download_parser.add_argument("--start", type=_parse_date, help="Optional start date (YYYY-MM-DD).")
    download_parser.add_argument("--end", type=_parse_date, help="Optional end date (YYYY-MM-DD).")
    download_parser.add_argument(
        "--interval",
        default="1d",
        help="Yahoo Finance data interval (default: 1d).",
    )
    download_parser.set_defaults(func=_handle_download)

    optimize_parser = subparsers.add_parser("optimize", help="Optimise portfolios")
    optimize_parser.add_argument(
        "--portfolio-dir",
        type=Path,
        default=DEFAULT_PORTFOLIO_DIR,
        help="Directory containing portfolio definition CSV files.",
    )
    optimize_parser.add_argument(
        "--price-dir",
        type=Path,
        default=DEFAULT_PRICE_HISTORY_DIR,
        help="Directory containing per-ticker price history CSV files.",
    )
    optimize_parser.add_argument(
        "--collated-dir",
        type=Path,
        help="Optional directory to write collated price histories for each portfolio.",
    )
    optimize_parser.add_argument(
        "--output",
        type=Path,
        help="Directory to write optimisation outputs (weights, performance, plots).",
    )
    optimize_parser.add_argument(
        "--plot",
        action="store_true",
        help="Write allocation pie charts when --output is specified.",
    )
    optimize_parser.add_argument("--start", type=_parse_date, help="Optional start date (YYYY-MM-DD).")
    optimize_parser.add_argument("--end", type=_parse_date, help="Optional end date (YYYY-MM-DD).")
    optimize_parser.add_argument(
        "--weight-bounds",
        type=_parse_weight_bounds,
        default=(0.0, 1.0),
        help="Global weight bounds as LOW,HIGH (default 0.0,1.0).",
    )
    optimize_parser.set_defaults(func=_handle_optimize)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    # argparse converts to datetime; pass ISO strings onwards for index filtering
    args.start = args.start.date().isoformat() if isinstance(args.start, datetime) else args.start
    args.end = args.end.date().isoformat() if isinstance(args.end, datetime) else args.end

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - entry point for python -m pysharpe.cli
    raise SystemExit(main())
