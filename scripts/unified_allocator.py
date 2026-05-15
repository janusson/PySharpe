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

    Returns a DataFrame with columns ['ticker', 'ma_crossover_signal', 'volatility_penalty'] for the latest date.
    """
    linker = DataLinker()
    signals = []

    # prices_df has Date index and ticker columns
    for ticker in prices_df.columns:
        # Create a price frame for this ticker
        ticker_data = prices_df[ticker].dropna().to_frame(name="price").reset_index()
        ticker_data.columns = ["date", "price"]

        linker.register_data("market_data", ticker_data)

        # Calculate signals (using default 20/50 windows)
        enhanced = linker.get_enhanced_market_data(short_window=20, long_window=50)

        if not enhanced.empty:
            linker.register_data("enhanced_data", enhanced)
            query = """
                WITH returns AS (
                    SELECT
                        date,
                        ma_crossover_signal,
                        LN(price / NULLIF(LAG(price, 1) OVER (ORDER BY date), 0)) AS log_ret
                    FROM enhanced_data
                ),
                volatility AS (
                    SELECT
                        date,
                        ma_crossover_signal,
                        STDDEV_SAMP(log_ret) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) * SQRT(252) AS vol_30d,
                        STDDEV_SAMP(log_ret) OVER () * SQRT(252) AS vol_hist
                    FROM returns
                )
                SELECT * FROM volatility ORDER BY date
            """
            vol_df = linker.execute_query(query)

            if not vol_df.empty:
                latest = vol_df.iloc[-1]
                signal = float(latest["ma_crossover_signal"])
                vol_30d = float(latest["vol_30d"])
                vol_hist = float(latest["vol_hist"])

                vol_ratio = (
                    (vol_30d / vol_hist) if pd.notna(vol_hist) and vol_hist > 0 else 1.0
                )

                # Proportional penalty based on how much current volatility exceeds historical
                penalty = max(0.0, min(1.0, vol_ratio - 1.0))

                # Reduce signal proportionally if it's positive to prevent buying falling knives
                if signal > 0:
                    signal *= 1.0 - penalty

                signals.append(
                    {
                        "ticker": ticker,
                        "ma_crossover_signal": signal,
                        "volatility_penalty": -penalty,  # Negative score for the allocation factor
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

    # 1. Data Ingestion & Refresh
    collation_service = CollationService(fetcher=YFinancePriceFetcher())
    target_weights, _ = _load_target_weights(args.portfolio, export_dir)
    tickers = target_weights["ticker"].tolist()

    if args.refresh:
        logger.info(f"Refreshing price data for {len(tickers)} tickers...")
        prices_df = collation_service.process_portfolio(
            args.portfolio, tickers, period="2y"
        )
    else:
        logger.info(f"Loading cached price data for {args.portfolio}...")
        csv_path = export_dir / f"{args.portfolio}_collated.csv"
        if not csv_path.exists():
            logger.error(
                f"Cached data not found at {csv_path}. Run with --refresh first."
            )
            return 1
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
    merged["volatility_penalty"] = merged["volatility_penalty"].fillna(0)

    config = AllocationConfig(
        trend_factors={"ma_crossover_signal": 0.2, "volatility_penalty": 0.1}
    )

    scored = score_opportunities(merged, config=config)
    allocations = allocate_contribution(
        scored, contribution_dollars=args.cash, config=config
    )

    # Prepare final report/plan
    valid_prices = allocations["latest_price"].where(allocations["latest_price"] > 0)
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
        logger.info(f"Allocation plan saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
