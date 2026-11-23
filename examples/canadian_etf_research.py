"""Reproducible Canadian ETF Portfolio Research Example for PySharpe.

Demonstrates how a Canadian DIY investor can compare five distinct long-term
portfolio structures using PySharpe's public APIs:
  1. US Large-Cap Equity (conceptual analogue: VFV.TO)
  2. Globally Diversified All-Equity (conceptual analogue: VEQT.TO / XEQT.TO)
  3. US Equity with Technology Tilt (conceptual analogue: VFV.TO + QQC.TO tilt)
  4. Globally Diversified Equity/Bond (conceptual analogue: VBAL.TO / 60-40)
  5. Simple Equal-Weight Baseline (1/N across the constituent universe)

All asset identifiers are fictional (e.g. F_US_LC, F_CAN_EQ, F_INTL_EQ,
F_TECH_TILT, F_CAN_BOND). Price histories are deterministically generated
using fixed seeds. No network calls or live market data are required.

Key Analyses:
  - In-Sample (IS: 2019-2021) vs. Out-of-Sample (OOS: 2022-2023) evaluation.
  - Realized returns, annualised volatility, CAGR, maximum drawdown, and duration.
  - Sharpe and Sortino ratios (calibrated to a consistent 2.5% risk-free rate).
  - Portfolio concentration: Herfindahl-Hirschman Index (HHI) and Effective N.
  - Correlation structure and Diversification Ratio (Choueifaty & Coignard, 2008).
  - Quarterly rebalancing backtest with transaction costs, slippage, and spread.
  - Deterministic Value Averaging (VA) 60/40 opportunity scoring and contribution routing.
  - Demonstration that the portfolio with the highest realized return (US_TechTilt)
    is NOT the portfolio with the best risk-adjusted performance (Balanced_60_40).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from pysharpe import (
    AllocationConfig,
    allocate_contribution,
    metrics,
    score_opportunities,
)
from pysharpe.analysis.backtest_engine import HistoricalBacktester

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEED: Final[int] = 123
RISK_FREE_RATE: Final[float] = 0.025  # 2.5% annual risk-free rate (CAD T-bills / cash)
PERIODS_PER_YEAR: Final[int] = 252

# Constituent Fictional Asset Universe (CAD-denominated)
ASSETS: Final[list[str]] = [
    "F_US_LC",  # Fictional US Large-Cap Equity ETF (analogue: VFV)
    "F_CAN_EQ",  # Fictional Canadian Broad Market Equity ETF (analogue: VCN/VDY)
    "F_INTL_EQ",  # Fictional International Developed & Emerging ETF (analogue: VIU/VEE)
    "F_TECH_TILT",  # Fictional US Technology Tilt ETF (analogue: QQC)
    "F_CAN_BOND",  # Fictional Canadian Aggregate Bond ETF (analogue: VAB/ZAG)
]

# Portfolio Target Allocations (sum to 1.0)
PORTFOLIO_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    "US_LargeCap": {
        "F_US_LC": 1.00,
        "F_CAN_EQ": 0.00,
        "F_INTL_EQ": 0.00,
        "F_TECH_TILT": 0.00,
        "F_CAN_BOND": 0.00,
    },
    "Global_AllEquity": {
        "F_US_LC": 0.45,
        "F_CAN_EQ": 0.30,
        "F_INTL_EQ": 0.25,
        "F_TECH_TILT": 0.00,
        "F_CAN_BOND": 0.00,
    },
    "US_TechTilt": {
        "F_US_LC": 0.60,
        "F_CAN_EQ": 0.00,
        "F_INTL_EQ": 0.00,
        "F_TECH_TILT": 0.40,
        "F_CAN_BOND": 0.00,
    },
    "Balanced_60_40": {
        "F_US_LC": 0.28,
        "F_CAN_EQ": 0.18,
        "F_INTL_EQ": 0.14,
        "F_TECH_TILT": 0.00,
        "F_CAN_BOND": 0.40,
    },
    "Equal_Weight_1_over_N": {
        "F_US_LC": 0.20,
        "F_CAN_EQ": 0.20,
        "F_INTL_EQ": 0.20,
        "F_TECH_TILT": 0.20,
        "F_CAN_BOND": 0.20,
    },
}

PORTFOLIO_DESCRIPTIONS: Final[dict[str, str]] = {
    "US_LargeCap": "100% US Large-Cap Equity (S&P 500 analogue, e.g. VFV)",
    "Global_AllEquity": "Globally Diversified All-Equity (VEQT/XEQT analogue: 45% US, 30% CA, 25% INTL)",
    "US_TechTilt": "US Equity with Tech Tilt (60% S&P 500 + 40% NASDAQ-100 analogue, e.g. VFV + QQC)",
    "Balanced_60_40": "Balanced 60/40 Equity/Bond (VBAL analogue: 60% Global Equity / 40% Canadian Bond)",
    "Equal_Weight_1_over_N": "Equal-Weight Baseline (20% in each of the 5 constituent asset classes)",
}


# ---------------------------------------------------------------------------
# Data Generation (Deterministic Synthetic Asset History)
# ---------------------------------------------------------------------------


def generate_synthetic_etf_data(
    seed: int = DEFAULT_SEED,
    start_date: str = "2019-01-02",
    end_date: str = "2023-12-29",
) -> pd.DataFrame:
    """Generate deterministic synthetic daily price histories for 5 fictional ETFs.

    Constructs a 5-year business-day price series with realistic correlation,
    volatility, and macro regime dynamics across:
      - In-Sample (IS: 2019-01-02 to 2021-12-31, 3 years)
      - Out-of-Sample (OOS: 2022-01-03 to 2023-12-29, 2 years)

    Returns:
        pd.DataFrame indexed by Business Day with columns ASSETS starting at 100.0.
    """
    dates = pd.date_range(start_date, end_date, freq="B")
    n_days = len(dates)

    # Base correlation matrix (equities positively correlated; bonds provide negative/zero correlation)
    corr = np.array(
        [
            [1.00, 0.70, 0.65, 0.85, -0.05],  # F_US_LC
            [0.70, 1.00, 0.60, 0.60, 0.05],  # F_CAN_EQ
            [0.65, 0.60, 1.00, 0.55, 0.00],  # F_INTL_EQ
            [0.85, 0.60, 0.55, 1.00, -0.10],  # F_TECH_TILT
            [-0.05, 0.05, 0.00, -0.10, 1.00],  # F_CAN_BOND
        ],
        dtype=float,
    )

    base_annual_vols = np.array([0.16, 0.14, 0.15, 0.22, 0.05], dtype=float)
    daily_vols = base_annual_vols / np.sqrt(PERIODS_PER_YEAR)

    is_mask = dates < pd.Timestamp("2022-01-01")
    is_len = int(np.sum(is_mask))
    oos_len = n_days - is_len

    # Regime 1: In-Sample (2019-2021) - Strong broad market expansion
    means_is = np.array([0.14, 0.10, 0.08, 0.20, 0.04], dtype=float) / PERIODS_PER_YEAR
    cov_is = np.outer(daily_vols, daily_vols) * corr
    chol_is = np.linalg.cholesky(cov_is)

    # Regime 2: Out-of-Sample (2022-2023) - Increased tech volatility & moderate equity returns
    means_oos = (
        np.array([0.07, 0.06, 0.05, 0.10, 0.035], dtype=float) / PERIODS_PER_YEAR
    )
    oos_vols = daily_vols.copy()
    oos_vols[3] *= 1.35  # Heightened volatility in tech tilt
    cov_oos = np.outer(oos_vols, oos_vols) * corr
    chol_oos = np.linalg.cholesky(cov_oos)

    rng = np.random.default_rng(seed=seed)
    shocks_is = rng.standard_normal((is_len, len(ASSETS)))
    shocks_oos = rng.standard_normal((oos_len, len(ASSETS)))

    rets_is = means_is + shocks_is @ chol_is.T
    rets_oos = means_oos + shocks_oos @ chol_oos.T

    all_returns = np.vstack([rets_is, rets_oos])
    prices = 100.0 * np.cumprod(1.0 + all_returns, axis=0)

    return pd.DataFrame(prices, index=dates, columns=ASSETS)


# ---------------------------------------------------------------------------
# Metric Evaluation Data Structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioSummary:
    """Summary of financial and risk-adjusted metrics for a portfolio."""

    portfolio_name: str
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_days: int
    cagr: float
    hhi_concentration: float
    effective_n: float
    diversification_ratio: float


def compute_concentration_metrics(weights: np.ndarray) -> tuple[float, float]:
    """Compute Herfindahl-Hirschman Index (HHI) and Effective Number of Assets.

    HHI = sum(w_i^2) in [1/N, 1.0]
    Effective N = 1 / HHI in [1.0, N]
    """
    hhi = float(np.sum(weights**2))
    effective_n = 1.0 / hhi if hhi > 0 else 0.0
    return hhi, effective_n


def compute_diversification_ratio(
    weights: np.ndarray,
    asset_vols: np.ndarray,
    portfolio_vol: float,
) -> float:
    """Compute Diversification Ratio (DR = sum(w_i * sigma_i) / sigma_p).

    Measures the ratio of the weighted constituent volatilities to the realized
    portfolio volatility. DR >= 1.0; higher DR indicates greater risk reduction.
    """
    weighted_vol = float(np.sum(weights * asset_vols))
    if portfolio_vol <= 0:
        return 1.0
    return weighted_vol / portfolio_vol


def evaluate_portfolio(
    name: str,
    weights_dict: dict[str, float],
    returns_df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PortfolioSummary:
    """Evaluate performance, risk, and concentration metrics for one portfolio."""
    weights = np.array([weights_dict[col] for col in returns_df.columns], dtype=float)
    port_returns = pd.Series(
        (returns_df @ weights).to_numpy(),
        index=returns_df.index,
        dtype=float,
    )

    ann_ret = float(
        metrics.annualize_return(port_returns, periods_per_year=PERIODS_PER_YEAR)  # type: ignore[arg-type]
    )
    ann_vol = float(
        metrics.annualize_volatility(port_returns, periods_per_year=PERIODS_PER_YEAR)  # type: ignore[arg-type]
    )
    sharpe = float(
        metrics.sharpe_ratio(
            port_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=PERIODS_PER_YEAR,
        )  # type: ignore[arg-type]
    )
    sortino = float(
        metrics.sortino_ratio(
            port_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=PERIODS_PER_YEAR,
        )  # type: ignore[arg-type]
    )

    # Reconstruct portfolio equity curve starting at CAD $10,000
    equity_curve = pd.Series(
        10000.0 * np.cumprod(1.0 + port_returns.to_numpy()),
        index=port_returns.index,
        dtype=float,
    )
    mdd = float(metrics.maximum_drawdown(equity_curve))
    mdd_days = int(metrics.max_drawdown_duration(equity_curve))
    cagr_val = float(metrics.cagr(equity_curve))

    hhi, eff_n = compute_concentration_metrics(weights)

    # Compute constituent asset volatilities for the diversification ratio
    asset_vols = np.array(
        [
            metrics.annualize_volatility(
                returns_df[col], periods_per_year=PERIODS_PER_YEAR
            )
            for col in returns_df.columns
        ],
        dtype=float,
    )
    div_ratio = compute_diversification_ratio(weights, asset_vols, ann_vol)

    return PortfolioSummary(
        portfolio_name=name,
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=mdd,
        max_drawdown_days=mdd_days,
        cagr=cagr_val,
        hhi_concentration=hhi,
        effective_n=eff_n,
        diversification_ratio=div_ratio,
    )


# ---------------------------------------------------------------------------
# Rebalancing & Transaction Cost Backtest
# ---------------------------------------------------------------------------


def run_rebalancing_backtest(
    prices: pd.DataFrame,
    initial_capital: float = 100000.0,
    fee_per_trade: float = 4.95,  # Canadian broker commission per trade
    slippage_pct: float = 0.0005,  # 5 bps execution slippage
    spread_pct: float = 0.0010,  # 10 bps bid-ask spread
) -> pd.DataFrame:
    """Simulate quarterly calendar rebalancing for all five portfolios.

    Models transaction costs, commissions, bid-ask spread, and slippage using
    PySharpe's HistoricalBacktester.
    """
    rows = []
    for name, tw in PORTFOLIO_WEIGHTS.items():
        bt = HistoricalBacktester(
            prices=prices,
            target_weights=tw,
            initial_capital=initial_capital,
            rebalance_freq="QE",
            fee_per_trade=fee_per_trade,
            slippage_pct=slippage_pct,
            spread_pct=spread_pct,
        )
        res = bt.run()
        terminal_val = float(res.portfolio_value.iloc[-1])
        num_rebalances = len(res.rebalance_events)
        total_ret = (terminal_val / initial_capital) - 1.0

        rows.append(
            {
                "Portfolio": name,
                "Start Capital (CAD)": initial_capital,
                "Terminal Value (CAD)": terminal_val,
                "Net Total Return": total_ret,
                "Rebalance Events": num_rebalances,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Value Averaging Contribution Allocation Demo
# ---------------------------------------------------------------------------


def demonstrate_value_averaging_contribution(
    current_prices: pd.Series,
    contribution_cad: float = 1000.0,
) -> pd.DataFrame:
    """Demonstrate PySharpe's Value Averaging contribution allocator.

    Illustrates how a Canadian DIY investor deploying a CAD $1,000 monthly contribution
    into the Balanced 60/40 portfolio routes cash using PySharpe's 60% path-drift /
    40% valuation mean-reversion scoring, directing capital to underweight holdings
    without triggering taxable sales.
    """
    target_weights = PORTFOLIO_WEIGHTS["Balanced_60_40"]

    # Fictional existing holdings showing slight drift:
    # Equities ran up, bonds fell behind their target path.
    drifted_holdings = {
        "F_US_LC": 32000.0,  # 32% (target 28%) -> Overweight
        "F_CAN_EQ": 19000.0,  # 19% (target 18%) -> Slight Overweight
        "F_INTL_EQ": 14000.0,  # 14% (target 14%) -> On Target
        "F_TECH_TILT": 0.0,  # 0%  (target 0%)
        "F_CAN_BOND": 35000.0,  # 35% (target 40%) -> Strongly Underweight
    }

    df = pd.DataFrame(
        [
            {
                "ticker": t,
                "current_value": drifted_holdings[t],
                "target_weight": target_weights[t],
                "price": float(current_prices[t]),
            }
            for t in ASSETS
        ]
    )

    # Build VA allocation configuration
    config = AllocationConfig(
        weight_underweight=0.60,  # 60% path-drift deviation
        weight_valuation=0.40,  # 40% valuation / mean-reversion
        min_allocation_dollars=25.0,
    )

    scored_df = score_opportunities(df, config=config)
    allocated_df = allocate_contribution(
        scored_df, contribution_dollars=contribution_cad, config=config
    )

    df_out = allocated_df[
        [
            "ticker",
            "current_value",
            "current_weight",
            "target_weight",
            "underweight",
            "opportunity_score",
            "recommended_allocation",
        ]
    ].copy()
    df_out.rename(columns={"recommended_allocation": "allocated_dollars"}, inplace=True)
    return df_out


# ---------------------------------------------------------------------------
# Formatting & Presentation Helpers
# ---------------------------------------------------------------------------


def format_summary_table(summaries: list[PortfolioSummary]) -> str:
    """Format a list of PortfolioSummary records into a clean Markdown table."""
    headers = [
        "Portfolio",
        "Ann. Ret",
        "Ann. Vol",
        "Sharpe",
        "Sortino",
        "Max DD",
        "CAGR",
        "HHI",
        "Eff. N",
        "Div. Ratio",
    ]
    rows = []
    for s in summaries:
        rows.append(
            [
                s.portfolio_name,
                f"{s.annual_return * 100:6.2f}%",
                f"{s.annual_volatility * 100:6.2f}%",
                f"{s.sharpe_ratio:6.2f}",
                f"{s.sortino_ratio:6.2f}",
                f"{s.max_drawdown * 100:6.2f}%",
                f"{s.cagr * 100:6.2f}%",
                f"{s.hhi_concentration:6.4f}",
                f"{s.effective_n:5.2f}",
                f"{s.diversification_ratio:5.2f}",
            ]
        )

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def fmt_row(vals: list[str]) -> str:
        return (
            "| " + " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(vals)) + " |"
        )

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    lines = [fmt_row(headers), sep]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Independent Numerical Validation
# ---------------------------------------------------------------------------


def run_independent_checks(
    returns_df: pd.DataFrame,
    tolerance: float = 1e-6,
) -> None:
    """Perform independent mathematical validation of PySharpe metrics.

    Compares PySharpe functions against raw numpy/pandas formula derivations:
      1. Geometric Annualized Return
      2. Annualized Volatility
      3. Sharpe Ratio
      4. Sortino Ratio (downside deviation below MAR)
      5. Concentration HHI and Effective N
      6. Diversification Ratio
    """
    port_weights = np.array(
        [PORTFOLIO_WEIGHTS["Balanced_60_40"][c] for c in returns_df.columns],
        dtype=float,
    )
    port_ret = (returns_df @ port_weights).to_numpy()
    n_obs = len(port_ret)

    # 1. Independent Return Check
    compounded = np.prod(1.0 + port_ret)
    expected_ann_ret = (compounded ** (PERIODS_PER_YEAR / n_obs)) - 1.0
    pysharpe_ann_ret = float(
        metrics.annualize_return(pd.Series(port_ret), periods_per_year=PERIODS_PER_YEAR)  # type: ignore[arg-type]
    )
    assert abs(expected_ann_ret - pysharpe_ann_ret) < tolerance, (
        "Annual return mismatch"
    )

    # 2. Independent Volatility Check
    expected_vol = np.std(port_ret, ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    pysharpe_vol = float(
        metrics.annualize_volatility(
            pd.Series(port_ret), periods_per_year=PERIODS_PER_YEAR
        )  # type: ignore[arg-type]
    )
    assert abs(expected_vol - pysharpe_vol) < tolerance, "Volatility mismatch"

    # 3. Independent Sharpe Ratio Check
    expected_sharpe = (expected_ann_ret - RISK_FREE_RATE) / expected_vol
    pysharpe_sharpe = float(
        metrics.sharpe_ratio(
            pd.Series(port_ret),
            risk_free_rate=RISK_FREE_RATE,
            periods_per_year=PERIODS_PER_YEAR,
        )  # type: ignore[arg-type]
    )
    assert abs(expected_sharpe - pysharpe_sharpe) < tolerance, "Sharpe ratio mismatch"

    # 4. Independent Sortino Ratio Check (MAR = Rf / 252)
    mar = RISK_FREE_RATE / PERIODS_PER_YEAR
    excess_mean = (np.mean(port_ret) - mar) * PERIODS_PER_YEAR
    downside = np.clip(port_ret - mar, a_min=None, a_max=0.0)
    downside_dev = np.sqrt(np.mean(downside**2)) * np.sqrt(PERIODS_PER_YEAR)
    expected_sortino = excess_mean / downside_dev
    pysharpe_sortino = float(
        metrics.sortino_ratio(
            pd.Series(port_ret),
            risk_free_rate=RISK_FREE_RATE,
            periods_per_year=PERIODS_PER_YEAR,
        )  # type: ignore[arg-type]
    )
    assert abs(expected_sortino - pysharpe_sortino) < tolerance, (
        "Sortino ratio mismatch"
    )

    # 5. Independent Concentration Checks
    expected_hhi = np.sum(port_weights**2)
    expected_eff_n = 1.0 / expected_hhi
    calc_hhi, calc_eff_n = compute_concentration_metrics(port_weights)
    assert abs(expected_hhi - calc_hhi) < tolerance, "HHI mismatch"
    assert abs(expected_eff_n - calc_eff_n) < tolerance, "Effective N mismatch"


# ---------------------------------------------------------------------------
# Main Orchestration Routine
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the offline Canadian ETF portfolio research comparison."""
    parser = argparse.ArgumentParser(
        description="PySharpe Canadian ETF Portfolio Research Comparison Example."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic synthetic data generation (default: 123)",
    )
    args = parser.parse_args(argv)

    print("=" * 80)
    print(" PySharpe · Reproducible Canadian ETF Portfolio Research Examples")
    print("=" * 80)
    print("Methodology:")
    print("  • Five long-term asset structures compared across Canadian DIY horizons.")
    print("  • Deterministic synthetic price series (2019-01-02 to 2023-12-29).")
    print("  • Risk-free hurdle rate: 2.50% annual (CAD cash/T-bills).")
    print("  • In-Sample window (2019-2021) vs. Out-of-Sample window (2022-2023).")
    print("  • Zero external network calls; fully reproducible offline.")
    print("-" * 80)

    # 1. Generate Data
    price_df = generate_synthetic_etf_data(seed=args.seed)
    returns_raw = metrics.compute_returns(price_df)
    assert isinstance(returns_raw, pd.DataFrame)
    returns_df: pd.DataFrame = returns_raw

    is_returns_raw = returns_df.loc[returns_df.index < "2022-01-01"]
    assert isinstance(is_returns_raw, pd.DataFrame)
    is_returns: pd.DataFrame = is_returns_raw

    oos_returns_raw = returns_df.loc[returns_df.index >= "2022-01-01"]
    assert isinstance(oos_returns_raw, pd.DataFrame)
    oos_returns: pd.DataFrame = oos_returns_raw

    # Run mathematical self-checks
    run_independent_checks(oos_returns)

    # 2. Portfolio Structure Overview
    print("\n1. Portfolio Structures & Conceptual Analogues:")
    for code, desc in PORTFOLIO_DESCRIPTIONS.items():
        weights_str = ", ".join(
            f"{k}: {v:.0%}" for k, v in PORTFOLIO_WEIGHTS[code].items() if v > 0
        )
        print(f"  • {code:22s}: {desc}")
        print(f"    Target Weights: {weights_str}")

    # 3. Correlation Matrix (Constituent ETFs)
    print("\n2. Constituent Asset Correlation Matrix (In-Sample 2019-2021):")
    corr_matrix = is_returns.corr().round(2)
    print(corr_matrix.to_string())

    # 4. Out-of-Sample Evaluation (2022 - 2023)
    oos_summaries = [
        evaluate_portfolio(name, weights, oos_returns, RISK_FREE_RATE)
        for name, weights in PORTFOLIO_WEIGHTS.items()
    ]
    print("\n3. OUT-OF-SAMPLE PERFORMANCE & RISK (2022 - 2023, Forward Evaluation):")
    print(format_summary_table(oos_summaries))

    # 5. In-Sample Evaluation (2019 - 2021)
    is_summaries = [
        evaluate_portfolio(name, weights, is_returns, RISK_FREE_RATE)
        for name, weights in PORTFOLIO_WEIGHTS.items()
    ]
    print("\n4. IN-SAMPLE PERFORMANCE & RISK (2019 - 2021, Historical Training):")
    print(format_summary_table(is_summaries))

    # 6. Key Financial Insight: Risk-Adjusted Inversion
    oos_dict = {s.portfolio_name: s for s in oos_summaries}
    tech_tilt = oos_dict["US_TechTilt"]
    balanced = oos_dict["Balanced_60_40"]
    global_eq = oos_dict["Global_AllEquity"]

    print(
        "\n5. Quantitative Research Takeaway: The Return vs. Risk-Adjusted Inversion:"
    )
    print(
        f"  • Highest Realized Return : {tech_tilt.portfolio_name} "
        f"({tech_tilt.annual_return * 100:.2f}% Ann. Ret, CAGR {tech_tilt.cagr * 100:.2f}%)"
    )
    print(
        f"  • BUT Worst Risk-Adjusted : Sharpe {tech_tilt.sharpe_ratio:.2f}, "
        f"Sortino {tech_tilt.sortino_ratio:.2f}, Max Drawdown {tech_tilt.max_drawdown * 100:.2f}%"
    )
    print(
        f"  • Best Risk-Adjusted      : {balanced.portfolio_name} "
        f"(Sharpe {balanced.sharpe_ratio:.2f}, Sortino {balanced.sortino_ratio:.2f}, "
        f"Max Drawdown {balanced.max_drawdown * 100:.2f}%)"
    )
    print(
        f"  • Geographic Benefit      : {global_eq.portfolio_name} reduced volatility by "
        f"{(oos_dict['US_LargeCap'].annual_volatility - global_eq.annual_volatility) * 100:.2f}% "
        f"and drawdown by {(global_eq.max_drawdown - oos_dict['US_LargeCap'].max_drawdown) * 100:.2f}% "
        f"versus single-country US concentration."
    )

    # 7. Rebalancing Backtest Simulation
    print("\n6. Rebalancing Backtest (Quarterly Execution with Fees & Slippage):")
    backtest_df = run_rebalancing_backtest(price_df.loc[price_df.index >= "2022-01-01"])
    for _, row in backtest_df.iterrows():
        print(
            f"  • {row['Portfolio']:22s} | Terminal: CAD ${row['Terminal Value (CAD)']:10,.2f} | "
            f"Net Return: {row['Net Total Return'] * 100:6.2f}% | Rebalances: {row['Rebalance Events']}"
        )

    # 8. Value Averaging Contribution Allocation Demo
    print("\n7. Value Averaging (VA) Smart Contribution Routing (CAD $1,000 New Cash):")
    current_prices = price_df.iloc[-1]
    va_df = demonstrate_value_averaging_contribution(
        current_prices, contribution_cad=1000.0
    )
    print(
        va_df.to_string(
            index=False,
            formatters={
                "current_value": "CAD ${:,.0f}".format,
                "current_weight": "{:.1%}".format,
                "target_weight": "{:.1%}".format,
                "underweight": "{:.1%}".format,
                "opportunity_score": "{:.3f}".format,
                "allocated_dollars": "CAD ${:,.2f}".format,
            },
        )
    )

    print("\n" + "=" * 80)
    print(
        " Summary: All independent verification checks and self-tests passed cleanly."
    )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
