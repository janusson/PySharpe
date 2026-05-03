"""Command line interface for PySharpe."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from pysharpe import data_collector, workflows
from pysharpe.analysis import load_category_map
from pysharpe.config import get_settings
from pysharpe.data import PortfolioRepository
from pysharpe.execution.allocator import (
    AllocationConfig,
    allocate_contribution,
    score_opportunities,
)
from pysharpe.execution.rebalance import build_rebalance_plan, format_rebalance_plan
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

    # Optimization configuration
    mer_mapping: dict[str, float] | None = None
    geo_mapping: dict[str, str] | None = None
    max_portfolio_mer: float | None = None
    geo_upper_bounds: dict[str, float] | None = None
    geo_lower_bounds: dict[str, float] | None = None

    config_path = None
    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            return 1
    else:
        default_config = Path("portfolio_config.json")
        if default_config.exists():
            config_path = default_config

    if config_path:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)

            mer_mapping = config_data.get("mer_mapping")
            geo_mapping = config_data.get("geo_mapping")

            constraints = config_data.get("constraints", {})
            max_portfolio_mer = constraints.get("max_portfolio_mer")
            geo_upper_bounds = constraints.get("geo_upper_bounds")
            geo_lower_bounds = constraints.get("geo_lower_bounds")

        except json.JSONDecodeError as exc:
            print(f"Error parsing configuration file: {exc}")
            return 1
        except Exception as exc:
            print(f"Unexpected error loading configuration: {exc}")
            return 1
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
        mer_mapping=mer_mapping,
        max_portfolio_mer=max_portfolio_mer,
        geo_mapping=geo_mapping,
        geo_lower_bounds=geo_lower_bounds,
        geo_upper_bounds=geo_upper_bounds,
        make_plot=not args.no_plot,
        category_map=category_map,
        include_unmapped_categories=include_unmapped,
        return_model=args.return_model,
    )

    if not results:
        print("No optimisation results produced. Ensure collated data exists.")
        return 1

    _summarise_results(results.values())
    print(f"Artefacts written to {export_dir}")
    return 0


def _handle_allocate(args: argparse.Namespace) -> int:
    import numpy as np

    current_state_path = Path(args.portfolio).expanduser()
    if not current_state_path.exists():
        print(f"Error: Could not find portfolio state file at {current_state_path}")
        print("Please provide a CSV with columns: ticker, current_value, target_weight")
        return 1

    try:
        df = pd.read_csv(current_state_path)
    except Exception as exc:
        print(f"Error reading portfolio state: {exc}")
        return 1

    required_cols = {"ticker", "current_value", "target_weight"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Error: Missing required columns in {current_state_path}: {missing}")
        return 1

    config = None
    if args.config:
        config_path = Path(args.config).expanduser()
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)

                alloc_cfg_kwargs = {}
                if "allocation_weights" in config_data:
                    weights = config_data["allocation_weights"]
                    alloc_cfg_kwargs.update(
                        {
                            k: v
                            for k, v in weights.items()
                            if hasattr(AllocationConfig, k)
                        }
                    )

                config = AllocationConfig(**alloc_cfg_kwargs)

                # Overlay fundamental data if present in config
                if "fundamentals" in config_data:
                    funds_df = pd.DataFrame.from_dict(
                        config_data["fundamentals"], orient="index"
                    )
                    funds_df.index.name = "ticker"
                    df = df.merge(funds_df, on="ticker", how="left")

            except Exception as exc:
                print(f"Warning: Failed to load config {config_path}: {exc}")
                print("Proceeding with default allocation config.")

    # Ensure required valuation columns exist even if empty
    for col in ["pe_ratio", "pb_ratio", "div_yield", "momentum_6m"]:
        if col not in df.columns:
            df[col] = np.nan

    scored = score_opportunities(df, config=config)
    result = allocate_contribution(
        scored, contribution_dollars=args.amount, config=config
    )

    cols = [
        "ticker",
        "current_value",
        "target_weight",
        "current_weight",
        "underweight",
        "valuation_score",
        "opportunity_score",
        "recommended_allocation",
        "recommended_weight_increase",
    ]

    pd.options.display.float_format = "{:,.4f}".format
    print(f"\nAllocation recommendation for ${args.amount:,.2f}:\n")
    print(result[cols].to_string(index=False))

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
        frame[args.date_column] = pd.to_datetime(
            frame[args.date_column], errors="coerce"
        )
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


def _parse_holdings_json(raw: str) -> dict[str, float]:
    """Parse inline JSON or a JSON file path into a holdings mapping.

    Parameters
    ----------
    raw : str
        Either a JSON object string or a path to a JSON file.

    Returns
    -------
    dict[str, float]
        Mapping of ticker symbol to current value or share count.

    Raises
    ------
    ValueError
        If the payload is not a valid non-empty ticker-to-number mapping.
    """

    candidate = Path(raw).expanduser()
    payload = candidate.read_text(encoding="utf-8") if candidate.exists() else raw
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Holdings JSON must be an object mapping ticker to value.")

    parsed: dict[str, float] = {}
    for ticker, value in data.items():
        clean_ticker = str(ticker).strip()
        if not clean_ticker:
            continue

        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            raise ValueError(f"Invalid holdings value for ticker {clean_ticker}.")
        if numeric < 0:
            raise ValueError(
                f"Holdings value cannot be negative for ticker {clean_ticker}."
            )
        parsed[clean_ticker] = float(numeric)

    if not parsed:
        raise ValueError("Holdings JSON did not contain any valid ticker entries.")
    return parsed


def _handle_rebalance(args: argparse.Namespace) -> int:
    """Execute the user-facing rebalance CLI workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments for the ``rebalance`` subcommand.

    Returns
    -------
    int
        Process exit code. Returns ``0`` on success and ``1`` for validation,
        parsing, or file errors.
    """

    holdings_mapping = None
    if args.holdings_json:
        try:
            holdings_mapping = _parse_holdings_json(args.holdings_json)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1

    try:
        plan = build_rebalance_plan(
            args.portfolio,
            new_cash=args.new_cash,
            holdings_csv=args.holdings_csv,
            holdings_mapping=holdings_mapping,
            holdings_kind=args.holdings_kind,
            export_dir=args.export_dir,
            config_path=args.config,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(format_rebalance_plan(plan, include_zero_buys=args.include_zero_buys))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysharpe",
        description="Portfolio analytics helpers for optimisation and simulations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # allocate
    allocate = subparsers.add_parser(
        "allocate",
        help="Calculate optimal cash deployment based on drift and valuation.",
    )
    allocate.add_argument(
        "--portfolio",
        required=True,
        help="Path to CSV containing current portfolio state (ticker, current_value, target_weight).",
    )
    allocate.add_argument(
        "--amount",
        type=float,
        required=True,
        help="Dollar amount to contribute to the portfolio.",
    )
    allocate.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON configuration file defining fundamental valuation mappings.",
    )

    rebalance = subparsers.add_parser(
        "rebalance",
        help="Build buy orders from saved optimisation outputs and current holdings.",
    )
    rebalance.add_argument(
        "--portfolio",
        required=True,
        help=(
            "Portfolio name whose <portfolio>_weights.txt and "
            "<portfolio>_collated.csv will be used."
        ),
    )
    rebalance_holdings = rebalance.add_mutually_exclusive_group(required=True)
    rebalance_holdings.add_argument(
        "--holdings-csv",
        type=Path,
        help=(
            "CSV containing a ticker column and either "
            "current_value/total_value or shares."
        ),
    )
    rebalance_holdings.add_argument(
        "--holdings-json",
        help=(
            "Inline JSON object or path to a JSON file mapping "
            "ticker to value/share count."
        ),
    )
    rebalance.add_argument(
        "--holdings-kind",
        choices=("value", "shares"),
        help="Interpret holdings JSON values, or override CSV auto-detection.",
    )
    rebalance.add_argument(
        "--new-cash",
        type=float,
        required=True,
        help="Dollar amount of new capital to allocate.",
    )
    rebalance.add_argument(
        "--export-dir",
        type=Path,
        help=(
            "Directory containing optimisation artefacts. "
            "Defaults to the configured export directory."
        ),
    )
    rebalance.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config with allocation_weights and fundamentals.",
    )
    rebalance.add_argument(
        "--include-zero-buys",
        action="store_true",
        help=(
            "Show the full merged portfolio state instead of only "
            "positive buy recommendations."
        ),
    )

    # optimise
    optimise = subparsers.add_parser(
        "optimise",
        help="Download (optional) data and run the optimisation pipeline.",
    )
    optimise.add_argument(
        "--portfolio", dest="portfolios", nargs="*", help="Portfolio names to target."
    )
    optimise.add_argument(
        "--portfolio-dir", type=Path, help="Directory containing portfolio CSV files."
    )
    optimise.add_argument(
        "--price-dir", type=Path, help="Directory for price history CSVs."
    )
    optimise.add_argument(
        "--export-dir", type=Path, help="Directory for collated data and outputs."
    )
    optimise.add_argument("--log-dir", type=Path, help="Optional logging directory.")
    optimise.add_argument(
        "--period",
        default="max",
        help="History period requested when start/end are omitted.",
    )
    optimise.add_argument(
        "--interval",
        default="1d",
        help="Sampling interval for downloads (default: 1d).",
    )
    optimise.add_argument("--start", help="ISO start date for downloads.")
    optimise.add_argument("--end", help="ISO end date for downloads.")
    optimise.add_argument(
        "--time-constraint",
        dest="time_constraint",
        help="ISO start date applied to collated data.",
    )
    optimise.add_argument(
        "--skip-download", action="store_true", help="Reuse previously downloaded data."
    )
    optimise.add_argument(
        "--no-plot", action="store_true", help="Skip allocation pie charts."
    )
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
    optimise.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON configuration file defining MER, geography, and constraints.",
    )
    optimise.add_argument(
        "--return-model",
        choices=["ema", "mean"],
        default="ema",
        help="Expected return calculation method (default: ema).",
    )

    # simulate-dca
    simulate = subparsers.add_parser(
        "simulate-dca",
        help="Generate a dollar-cost averaging projection.",
    )
    simulate.add_argument(
        "--months",
        type=int,
        default=36,
        help="Number of months to simulate (default: 36).",
    )
    simulate.add_argument(
        "--initial", type=float, default=1000.0, help="Initial lump sum investment."
    )
    simulate.add_argument(
        "--monthly", type=float, default=250.0, help="Recurring contribution amount."
    )
    simulate.add_argument(
        "--rate",
        type=float,
        default=0.08,
        help="Annual return rate as a decimal (default: 0.08).",
    )
    simulate.add_argument("--output", help="Optional path to save the generated plot.")
    simulate.add_argument(
        "--plot", action="store_true", help="Display the plot interactively."
    )

    # plot
    plot = subparsers.add_parser(
        "plot",
        help="Plot numeric series from a CSV file (collated data or performance logs).",
    )
    plot.add_argument(
        "--input", required=True, help="Path to the CSV file to visualise."
    )
    plot.add_argument(
        "--columns",
        nargs="*",
        help="Specific column names to plot (defaults to numeric columns).",
    )
    plot.add_argument(
        "--date-column", help="Name of a column to use as the datetime index."
    )
    plot.add_argument("--title", help="Custom plot title.")
    plot.add_argument(
        "--ylabel", default="Value", help="Y axis label (default: Value)."
    )
    plot.add_argument("--output", help="Optional file path for saving the plot.")
    plot.add_argument(
        "--show", action="store_true", help="Display the plot interactively."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``python -m pysharpe.cli`` and console scripts."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "optimise":
        return _handle_optimise(args)
    if args.command == "allocate":
        return _handle_allocate(args)
    if args.command == "rebalance":
        return _handle_rebalance(args)
    if args.command == "simulate-dca":
        return _handle_simulate_dca(args)
    if args.command == "plot":
        return _handle_plot(args)

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
