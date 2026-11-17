"""Deterministic multi-account Canadian portfolio accounting scenario.

Demonstrates PySharpe's multi-currency ledger, explicit FX conversion,
cash conservation reconciliation, Non-Registered CRA ACB isolation vs. TFSA,
and handoff to cash-flow rebalancing using purely deterministic synthetic data
with explicit exchange rates and UTC timestamps (no live market data).

Run offline:
    PYTHONPATH=src python examples/canadian_accounting_scenario.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal

from pysharpe.execution.cash_flow_rebalance import (
    CashFlowRebalanceResult,
    allocate_contribution_cash_flow,
)
from pysharpe.execution.ledger import (
    Account,
    Currency,
    MarketSnapshot,
    PortfolioLedger,
    ReconciliationReport,
    Transaction,
    TransactionKind,
)
from pysharpe.optimization.tax_location import AccountType

UNRESOLVED_ACCOUNTING_POLICIES: tuple[str, ...] = (
    "1. Book cost reporting basis for USD securities: native USD cost vs. historical "
    "FX CAD cost (both are tracked separately without choosing one over the other).",
    "2. ITA s. 39(2) foreign-currency capital gains/losses on USD cash disposals "
    "(e.g., USD fees or spending USD cash converted at historical rates) are not "
    "computed automatically.",
    "3. Settlement-date (T+1) vs. trade-date spot FX translation for CRA Schedule 3 "
    "ACB reporting: each transaction records one explicit user-supplied FX rate.",
    "4. Cash rounding mode: ledger defaults explicitly to ROUND_HALF_EVEN and lets "
    "callers choose ROUND_HALF_UP or another supported Decimal mode.",
    "5. Rebalance integration currency: MarketSnapshot translates all positions to "
    "CAD before passing current_value to allocate_contribution_cash_flow; broker-side "
    "order sizing in USD cash vs. FX-converted CAD cash remains a caller decision.",
)


def build_canadian_scenario() -> tuple[PortfolioLedger, MarketSnapshot]:
    """Construct the deterministic TFSA + Non-Registered scenario ledger."""
    tfsa = Account(account_id="TFSA-Main", account_type=AccountType.TFSA)
    non_reg = Account(account_id="NonReg-Taxable", account_type=AccountType.NON_REG)

    transactions = [
        Transaction(
            timestamp=datetime(2025, 1, 6, 14, 30, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.CAD,
            amount=Decimal("10000.00"),
            note="Initial TFSA CAD contribution",
        ),
        Transaction(
            timestamp=datetime(2025, 1, 6, 14, 35, tzinfo=UTC),
            account=non_reg,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.CAD,
            amount=Decimal("8000.00"),
            note="Initial Non-Reg CAD deposit",
        ),
        Transaction(
            timestamp=datetime(2025, 1, 6, 15, 0, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.BUY,
            currency=Currency.CAD,
            ticker="VFV.TO",
            quantity=Decimal("40"),
            price=Decimal("140.00"),
            fee=Decimal("4.95"),
            note="Buy CAD-listed S&P 500 ETF in TFSA",
        ),
        Transaction(
            timestamp=datetime(2025, 1, 7, 15, 0, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.BUY,
            currency=Currency.CAD,
            ticker="VCN.TO",
            quantity=Decimal("50"),
            price=Decimal("50.00"),
            fee=Decimal("0.00"),
            note="Buy CAD-listed Canada All-Cap ETF in TFSA",
        ),
        Transaction(
            timestamp=datetime(2025, 1, 10, 16, 0, tzinfo=UTC),
            account=non_reg,
            kind=TransactionKind.FX_CONVERSION,
            currency=Currency.CAD,
            to_currency=Currency.USD,
            amount=Decimal("4200.00"),
            fx_rate=Decimal("1.4000"),
            fee=Decimal("15.00"),
            note="Convert CAD to USD in Non-Reg with explicit FX fee",
        ),
        Transaction(
            timestamp=datetime(2025, 1, 13, 15, 30, tzinfo=UTC),
            account=non_reg,
            kind=TransactionKind.BUY,
            currency=Currency.USD,
            ticker="VTI",
            quantity=Decimal("10"),
            price=Decimal("280.00"),
            fee=Decimal("5.00"),
            fx_rate=Decimal("1.4100"),
            note="Buy US-listed VTI in Non-Reg USD cash",
        ),
        Transaction(
            timestamp=datetime(2025, 2, 3, 15, 0, tzinfo=UTC),
            account=non_reg,
            kind=TransactionKind.BUY,
            currency=Currency.CAD,
            ticker="VFV.TO",
            quantity=Decimal("10"),
            price=Decimal("145.00"),
            fee=Decimal("0.00"),
            note="Buy VFV.TO in Non-Reg (isolated from TFSA ACB)",
        ),
        Transaction(
            timestamp=datetime(2025, 2, 10, 18, 0, tzinfo=UTC),
            account=non_reg,
            kind=TransactionKind.FEE,
            currency=Currency.USD,
            amount=Decimal("10.00"),
            fx_rate=Decimal("1.4200"),
            note="Monthly USD custody/data fee",
        ),
        Transaction(
            timestamp=datetime(2025, 2, 14, 15, 0, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.WITHDRAWAL,
            currency=Currency.CAD,
            amount=Decimal("500.00"),
            note="Partial TFSA CAD cash withdrawal",
        ),
        Transaction(
            timestamp=datetime(2025, 3, 3, 16, 0, tzinfo=UTC),
            account=non_reg,
            kind=TransactionKind.SELL,
            currency=Currency.USD,
            ticker="VTI",
            quantity=Decimal("4"),
            price=Decimal("300.00"),
            fee=Decimal("5.00"),
            fx_rate=Decimal("1.4400"),
            note="Partial taxable sale of USD VTI lot",
        ),
    ]

    ledger = PortfolioLedger(allow_negative_cash=False, rounding=ROUND_HALF_EVEN)
    ledger.record_many(transactions)

    snapshot = MarketSnapshot(
        as_of=datetime(2025, 3, 31, 20, 0, tzinfo=UTC),
        cad_per_usd=Decimal("1.4250"),
        cad_prices={"VFV.TO": Decimal("150.00"), "VCN.TO": Decimal("52.00")},
        usd_prices={"VTI": Decimal("295.00")},
    )
    return ledger, snapshot


def run_canadian_scenario() -> tuple[ReconciliationReport, CashFlowRebalanceResult]:
    """Run ledger reconciliation and downstream cash-flow rebalance handoff."""
    ledger, snapshot = build_canadian_scenario()
    report = ledger.reconcile(snapshot)

    holdings_df = report.to_holdings_frame(by="account_type")
    rebalance_result = allocate_contribution_cash_flow(
        cash_amount=1500.00,
        current_balances=holdings_df,
        target_weights={"VFV.TO": 0.50, "VCN.TO": 0.30, "VTI": 0.20},
        account_types={
            "VFV.TO": AccountType.TFSA.value,
            "VCN.TO": AccountType.TFSA.value,
            "VTI": AccountType.NON_REG.value,
        },
        opportunity_scores={"VFV.TO": 0.60, "VCN.TO": 0.60, "VTI": 0.60},
    )
    return report, rebalance_result


def format_scenario_summary(
    report: ReconciliationReport,
    rebalance_result: CashFlowRebalanceResult,
) -> str:
    """Format a human-readable reconciliation and rebalance report."""
    lines: list[str] = [
        f"Canadian Portfolio Reconciliation as of {report.as_of.isoformat()} "
        f"(Spot USD/CAD = {report.cad_per_usd})",
        "=" * 78,
        f"Total Portfolio Value (CAD): ${report.total_portfolio_cad:,.2f}",
        f"  Cash (CAD):                ${report.total_cash_cad:,.2f}",
        f"  Cash (USD):                USD {report.total_cash_usd:,.2f}",
        f"  Cash Equivalent (CAD):     ${report.total_cash_equivalent_cad:,.2f}",
        f"  Holdings Market Value:     ${report.total_holdings_cad:,.2f}",
        f"  All Cash Conserved Exact:  {report.all_cash_conserved}",
        "",
        "Account Breakdown:",
    ]

    for account_id, acct in report.accounts.items():
        lines.append(
            f"  [{account_id} | {acct.account_type.value}] "
            f"CAD Cash=${acct.cash_cad:,.2f}, USD Cash=${acct.cash_usd:,.2f}, "
            f"Holdings=${acct.holdings_market_value_cad:,.2f}, "
            f"Total=${acct.total_value_cad:,.2f}"
        )
        for h in acct.holdings:
            lines.append(
                f"    - {h.ticker} ({h.currency.value}): {h.shares} sh @ "
                f"{h.market_price_native} = ${h.market_value_cad:,.2f} CAD "
                f"(Book CAD=${h.book_cost_cad:,.2f}, "
                f"Unrealized CAD=${h.unrealized_gain_cad:,.2f})"
            )

    lines.append("")
    lines.append("CRA Non-Registered Pooled Tax ACB (CAD, TFSA excluded):")
    for ticker, pos in sorted(report.non_reg_tax_acb.items()):
        lines.append(
            f"  - {ticker}: {pos.total_shares:g} sh, Total ACB=${pos.total_cost:,.2f}, "
            f"ACB/sh=${pos.acb_per_share:,.4f}"
        )
    lines.append(
        f"  Realized Capital Gains (Non-Reg CAD): ${report.non_reg_realized_gains_cad:,.2f}"
    )

    lines.append("")
    lines.append("Rebalance Buy Orders for $1,500.00 CAD New Contribution:")
    for _, row in rebalance_result.buy_orders.iterrows():
        lines.append(
            f"  - {row['ticker']} -> {row['account']}: ${row['buy_amount']:,.2f} CAD "
            f"(new weight {row['new_weight']:.2%})"
        )
    if rebalance_result.taxable_sell_triggered:
        lines.append("Taxable-Sell Guardrail Alerts (Non-Reg only; TFSA excluded):")
        for _, row in rebalance_result.sell_orders.iterrows():
            lines.append(
                f"  - {row['ticker']} ({row['account']}): suggested sell "
                f"${row['sell_amount']:,.2f} CAD ({row['drift_bps']:.1f} bps drift)"
            )

    lines.append("")
    lines.append("Explicit Unresolved Accounting-Policy Choices:")
    lines.extend(f"  {item}" for item in UNRESOLVED_ACCOUNTING_POLICIES)
    return "\n".join(lines)


if __name__ == "__main__":
    summary_report, summary_rebalance = run_canadian_scenario()
    print(format_scenario_summary(summary_report, summary_rebalance))
