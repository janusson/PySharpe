#!/usr/bin/env python3
"""Unified execution script for scheduled portfolio allocation using DuckDB signals."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pysharpe.config import get_settings
from pysharpe.data.collation import CollationService
from pysharpe.data.fetcher import YFinancePriceFetcher
from pysharpe.data.linkage import DataLinker
from pysharpe.exceptions import DataIngestionError
from pysharpe.execution.allocator import (
    AllocationConfig,
    allocate_contribution,
    score_opportunities,
)
from pysharpe.execution.rebalance import (
    RebalancePlan,
    _load_holdings_frame,
    _load_target_weights,
    format_rebalance_plan,
)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def generate_signals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Use DuckDB to generate trend signals for each ticker in the prices DataFrame.

    Returns a DataFrame with columns ['ticker', 'ma_crossover_signal', 'volatility_ratio'] for the latest date.
    """
    linker = DataLinker()
    signals = []

    # prices_df has Date index and ticker columns
    for ticker in prices_df.columns:
        # Create a price frame for this ticker
        ticker_data = prices_df[ticker].dropna().to_frame(name="price").reset_index()
        ticker_data.columns = ["date", "price"]

        linker.register_data("market_data", ticker_data)

        # Calculate trend signals (MA crossover + volatility ratio)
        trend_df = linker.calculate_trend_signals(
            market_table="market_data",
            short_window=20,
            long_window=50,
        )

        if not trend_df.empty:
            latest = trend_df.iloc[-1]
            signal = float(latest["ma_crossover_signal"])
            vol_ratio = float(latest["volatility_ratio"])

            # Fall back to 1.0 if the ratio is NaN (e.g. not enough data)
            if pd.isna(vol_ratio):
                vol_ratio = 1.0

            signals.append(
                {
                    "ticker": ticker,
                    "ma_crossover_signal": signal,
                    "volatility_ratio": vol_ratio,
                }
            )

    linker.close()
    return pd.DataFrame(signals)


def main():
    parser = argparse.ArgumentParser(description="Scheduled PySharpe Allocation Runner")
    parser.add_argument(
        "--portfolio", required=True, help="Portfolio name (e.g., 'demo')"
    )
    parser.add_argument(
        "--holdings",
        type=Path,
        required=True,
        help="Path to holdings CSV (ticker, shares)",
    )
    parser.add_argument(
        "--cash", type=float, required=True, help="New cash contribution amount"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh price data from Yahoo Finance"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional CSV path to save the allocation plan"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("unified_allocator")

    settings = get_settings()
    export_dir = Path(settings.export_dir)

    try:
        # 1. Data Ingestion & Refresh
        collation_service = CollationService(fetcher=YFinancePriceFetcher())
        target_weights, _ = _load_target_weights(args.portfolio, export_dir)
        tickers = target_weights["ticker"].tolist()

        if args.refresh:
            logger.info("Refreshing price data for %d tickers...", len(tickers))
            prices_df = collation_service.process_portfolio(
                args.portfolio, tickers, period="2y"
            )
        else:
            logger.info("Loading cached price data for %s...", args.portfolio)
            csv_path = export_dir / f"{args.portfolio}_collated.csv"
            if not csv_path.exists():
                raise DataIngestionError(
                    f"Cached data not found at {csv_path}. Run with --refresh first."
                )
            prices_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # 2. Signal Generation (DuckDB)
        logger.info("Generating trend signals using DuckDB...")
        trend_signals = generate_signals(prices_df)

        # 3. Allocation logic
        logger.info("Calculating allocations...")
        latest_prices = (
            prices_df.iloc[-1].rename_axis("ticker").reset_index(name="latest_price")
        )

        holdings = _load_holdings_frame(
            holdings_csv=args.holdings,
            holdings_mapping=None,
            holdings_kind="shares",  # Assuming shares for this scheduled script
            latest_prices=latest_prices,
        )

        # Merge everything for scoring
        merged = target_weights.merge(latest_prices, on="ticker", how="outer")
        merged = merged.merge(holdings, on="ticker", how="outer")
        merged = merged.merge(trend_signals, on="ticker", how="left")

        # Fill defaults
        merged["target_weight"] = merged["target_weight"].fillna(0.0)
        merged["current_value"] = merged["current_value"].fillna(0.0)
        merged["ma_crossover_signal"] = merged["ma_crossover_signal"].fillna(
            0
        )  # Neutral if missing
        merged["volatility_ratio"] = merged["volatility_ratio"].fillna(1.0)

        config = AllocationConfig(
            weight_underweight=1.0,
            weight_valuation=0.0,
            trend_factors={"ma_crossover_signal": 0.2, "volatility_ratio": 0.1},
        )

        scored = score_opportunities(merged, config=config)
        allocations = allocate_contribution(
            scored, contribution_dollars=args.cash, config=config
        )

        # Prepare final report/plan
        valid_prices = allocations["latest_price"].where(
            allocations["latest_price"] > 0
        )
        allocations["recommended_shares"] = (
            allocations["recommended_allocation"] / valid_prices
        )

        plan = RebalancePlan(
            portfolio_name=args.portfolio,
            new_cash=args.cash,
            weights_path=export_dir / f"{args.portfolio}_weights.txt",
            prices_path=export_dir / f"{args.portfolio}_collated.csv",
            scored_state=scored,
            allocations=allocations,
        )

        print("\n" + "=" * 50)
        print(format_rebalance_plan(plan, include_zero_buys=False))
        print("=" * 50 + "\n")

        if args.output:
            allocations.to_csv(args.output, index=False)
            logger.info("Allocation plan saved to %s", args.output)

    except DataIngestionError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
