from pathlib import Path

import pandas as pd

from pysharpe.analysis.backtest_engine import HistoricalBacktester
from pysharpe.metrics import cagr, maximum_drawdown


def main():
    # 1. Load Collated Data
    # Assuming you have run 'pysharpe optimise' at least once to generate this
    collated_path = Path("data/exports/demo_collated.csv")
    
    if not collated_path.exists():
        print(f"Error: {collated_path} not found.")
        print("Please run 'pysharpe optimise --portfolio demo' first to fetch data.")
        return

    print(f"Loading data from {collated_path}...")
    prices = pd.read_csv(collated_path, parse_dates=True, index_col="Date")

    # 2. Define Target Allocations (must match columns in CSV)
    # Adjust these to match the tickers in your demo.csv
    # For this example, we'll try to detect them or use a generic split
    assets = prices.columns.tolist()
    n = len(assets)
    # Simple equal weight strategy for demo
    target_weights = {ticker: 1.0/n for ticker in assets}
    
    print(f"Backtesting with equal weights ({n} assets)...")

    # 3. Configure Backtester
    backtester = HistoricalBacktester(
        prices=prices,
        target_weights=target_weights,
        initial_capital=10000.0,
        rebalance_freq="Q",  # Quarterly rebalancing
        abs_band=0.05,       # 5% absolute drift tolerance
        rel_band=None        # No relative drift check
    )

    # 4. Run Simulation
    result = backtester.run()

    # 5. Analyze Results
    final_val = result.portfolio_value.iloc[-1]
    total_return = (final_val / 10000.0) - 1
    
    # Calculate metrics
    cagr_val = cagr(result.portfolio_value)
    max_dd = maximum_drawdown(result.portfolio_value)

    print("-" * 40)
    print("Initial Value:   $10,000.00")
    print(f"Final Value:     ${final_val:,.2f}")
    print(f"Total Return:    {total_return:.2%}")
    print(f"CAGR:            {cagr_val:.2%}")
    print(f"Max Drawdown:    {max_dd:.2%}")
    print(f"Rebalance Events: {len(result.rebalance_events)}")
    print("-" * 40)
    
    if len(result.rebalance_events) > 0:
        print("First 5 rebalance dates:")
        for date in result.rebalance_events[:5]:
            print(f" - {date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
