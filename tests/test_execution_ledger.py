"""Invariant and scenario tests for ``pysharpe.execution.ledger``.

All tests are deterministic, offline, and use explicit timestamps and FX rates.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

import pysharpe
from pysharpe.exceptions import DataValidationError
from pysharpe.execution.cash_flow_rebalance import allocate_contribution_cash_flow
from pysharpe.execution.ledger import (
    Account,
    Currency,
    MarketSnapshot,
    PortfolioLedger,
    Transaction,
    TransactionKind,
)
from pysharpe.optimization.tax_location import AccountType

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_scenario_module():
    path = _REPO_ROOT / "examples" / "canadian_accounting_scenario.py"
    spec = importlib.util.spec_from_file_location("canadian_accounting_scenario", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Validation & construction hygiene
# ---------------------------------------------------------------------------


def test_public_top_level_lazy_exports_resolve() -> None:
    for name in (
        "Account",
        "AccountValuationSummary",
        "CashBalanceSummary",
        "Currency",
        "HoldingValuation",
        "MarketSnapshot",
        "PortfolioLedger",
        "PositionState",
        "ReconciliationReport",
        "Transaction",
        "TransactionKind",
    ):
        assert hasattr(pysharpe, name)
        assert name in pysharpe.__all__


def test_account_validation_rejects_blank_and_untyped() -> None:
    with pytest.raises(DataValidationError):
        Account(account_id="   ", account_type=AccountType.TFSA)
    with pytest.raises(DataValidationError):
        Account(account_id="TFSA", account_type="TFSA")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_amount", [True, float("nan"), float("inf"), "-10", "abc"])
def test_transaction_rejects_invalid_numeric_inputs(bad_amount: object) -> None:
    acct = Account("TFSA", AccountType.TFSA)
    with pytest.raises(DataValidationError):
        Transaction(
            timestamp=datetime(2025, 1, 2, tzinfo=UTC),
            account=acct,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.CAD,
            amount=bad_amount,  # type: ignore[arg-type]
        )


def test_usd_transaction_and_fx_conversion_validation() -> None:
    acct = Account("NonReg", AccountType.NON_REG)
    ts = datetime(2025, 1, 2, tzinfo=UTC)

    with pytest.raises(DataValidationError, match="fx_rate"):
        Transaction(
            timestamp=ts,
            account=acct,
            kind=TransactionKind.BUY,
            currency=Currency.USD,
            ticker="VTI",
            quantity=1,
            price=250,
        )

    with pytest.raises(DataValidationError, match="differ"):
        Transaction(
            timestamp=ts,
            account=acct,
            kind=TransactionKind.FX_CONVERSION,
            currency=Currency.CAD,
            to_currency=Currency.CAD,
            amount=1000,
            fx_rate="1.40",
        )

    with pytest.raises(DataValidationError, match="disagrees"):
        Transaction(
            timestamp=ts,
            account=acct,
            kind=TransactionKind.FX_CONVERSION,
            currency=Currency.CAD,
            to_currency=Currency.USD,
            amount=1400,
            fx_rate="1.40",
            to_amount="999.00",
        )


def test_unsupported_rounding_mode_rejected() -> None:
    with pytest.raises(DataValidationError, match="rounding"):
        PortfolioLedger(rounding="INVALID_MODE")


# ---------------------------------------------------------------------------
# Ordering, atomicity, isolation & rounding invariants
# ---------------------------------------------------------------------------


def test_out_of_order_insertion_replays_chronologically() -> None:
    tfsa = Account("TFSA", AccountType.TFSA)
    ledger = PortfolioLedger()

    buy_tx = Transaction(
        timestamp=datetime(2025, 1, 10, 15, 0, tzinfo=UTC),
        account=tfsa,
        kind=TransactionKind.BUY,
        currency=Currency.CAD,
        ticker="VFV.TO",
        quantity=10,
        price="100.00",
    )
    deposit_tx = Transaction(
        timestamp=datetime(2025, 1, 10, 14, 0, tzinfo=UTC),
        account=tfsa,
        kind=TransactionKind.DEPOSIT,
        currency=Currency.CAD,
        amount="1000.00",
    )

    # Record BUY and DEPOSIT out of chronological order in one atomic batch.
    ledger.record_many([buy_tx, deposit_tx])
    assert ledger.cash_balance("TFSA", Currency.CAD) == Decimal("0.00")
    pos = ledger.positions("TFSA")[0]
    assert pos.shares == Decimal("10")
    assert pos.book_cost_cad == Decimal("1000.00")


def test_chronological_overdraft_or_oversell_rolls_back_atomically() -> None:
    tfsa = Account("TFSA", AccountType.TFSA)
    ledger = PortfolioLedger()
    ledger.record(
        Transaction(
            timestamp=datetime(2025, 1, 5, 12, 0, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.CAD,
            amount="500.00",
        )
    )

    # Overdraft attempt
    with pytest.raises(DataValidationError, match="Insufficient CAD cash"):
        ledger.record(
            Transaction(
                timestamp=datetime(2025, 1, 6, 12, 0, tzinfo=UTC),
                account=tfsa,
                kind=TransactionKind.BUY,
                currency=Currency.CAD,
                ticker="VFV.TO",
                quantity=10,
                price="100.00",
            )
        )

    assert ledger.cash_balance("TFSA", Currency.CAD) == Decimal("500.00")
    assert ledger.positions("TFSA") == []

    # Oversell attempt
    ledger.record(
        Transaction(
            timestamp=datetime(2025, 1, 6, 12, 0, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.BUY,
            currency=Currency.CAD,
            ticker="VFV.TO",
            quantity=4,
            price="100.00",
        )
    )
    with pytest.raises(DataValidationError, match="Cannot sell"):
        ledger.record(
            Transaction(
                timestamp=datetime(2025, 1, 7, 12, 0, tzinfo=UTC),
                account=tfsa,
                kind=TransactionKind.SELL,
                currency=Currency.CAD,
                ticker="VFV.TO",
                quantity=5,
                price="110.00",
            )
        )

    assert ledger.cash_balance("TFSA", Currency.CAD) == Decimal("100.00")
    assert ledger.positions("TFSA")[0].shares == Decimal("4")


def test_mixed_timezones_and_conflicting_account_types_rejected() -> None:
    ledger = PortfolioLedger()
    tfsa = Account("Acct-1", AccountType.TFSA)
    non_reg_same_id = Account("Acct-1", AccountType.NON_REG)

    ledger.record(
        Transaction(
            timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            account=tfsa,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.CAD,
            amount="1000.00",
        )
    )

    with pytest.raises(DataValidationError, match="naive and timezone-aware"):
        ledger.record(
            Transaction(
                timestamp=datetime(2025, 1, 2, 10, 0),
                account=tfsa,
                kind=TransactionKind.DEPOSIT,
                currency=Currency.CAD,
                amount="100.00",
            )
        )

    with pytest.raises(DataValidationError, match="switch AccountType"):
        ledger.record(
            Transaction(
                timestamp=datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
                account=non_reg_same_id,
                kind=TransactionKind.DEPOSIT,
                currency=Currency.CAD,
                amount="100.00",
            )
        )


def test_explicit_rounding_mode_controls_fx_half_cent() -> None:
    # Exact half-cent product (1.00 USD * 1.405 CAD/USD = 1.405 CAD):
    # ROUND_HALF_EVEN -> 1.40; ROUND_HALF_UP -> 1.41.
    acct = Account("NonReg", AccountType.NON_REG)
    txs = [
        Transaction(
            timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            account=acct,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.USD,
            amount="10.00",
            fx_rate="1.405",
        ),
        Transaction(
            timestamp=datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
            account=acct,
            kind=TransactionKind.FX_CONVERSION,
            currency=Currency.USD,
            to_currency=Currency.CAD,
            amount="1.00",
            fx_rate="1.405",
        ),
    ]

    even_ledger = PortfolioLedger(rounding=ROUND_HALF_EVEN)
    even_ledger.record_many(txs)
    assert even_ledger.cash_balance("NonReg", Currency.CAD) == Decimal("1.40")

    up_ledger = PortfolioLedger(rounding=ROUND_HALF_UP)
    up_ledger.record_many(txs)
    assert up_ledger.cash_balance("NonReg", Currency.CAD) == Decimal("1.41")


def test_multi_account_usd_cash_ties_out_across_portfolio_summary() -> None:
    # Two accounts each hold 1.00 USD translated at 1.405 CAD/USD (1.40 per
    # account under ROUND_HALF_EVEN). Portfolio total_cash_equivalent_cad must
    # equal sum(acct.cash_total_cad), not aggregate round(2.00 * 1.405) = 2.81.
    tfsa = Account("TFSA", AccountType.TFSA)
    non_reg = Account("NonReg", AccountType.NON_REG)
    ledger = PortfolioLedger(rounding=ROUND_HALF_EVEN)
    ledger.record_many(
        [
            Transaction(
                timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
                account=tfsa,
                kind=TransactionKind.DEPOSIT,
                currency=Currency.USD,
                amount="1.00",
                fx_rate="1.405",
            ),
            Transaction(
                timestamp=datetime(2025, 1, 1, 10, 5, tzinfo=UTC),
                account=non_reg,
                kind=TransactionKind.DEPOSIT,
                currency=Currency.USD,
                amount="1.00",
                fx_rate="1.405",
            ),
        ]
    )

    report = ledger.reconcile(
        MarketSnapshot(
            as_of=datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
            cad_per_usd="1.405",
        )
    )
    assert report.total_cash_equivalent_cad == Decimal("2.80")
    assert report.total_portfolio_cad == sum(
        acct.total_value_cad for acct in report.accounts.values()
    )
    assert report.total_portfolio_cad == sum(report.currency_exposure_cad.values())


def test_full_sale_zeroes_position_and_conserves_cash() -> None:
    acct = Account("TFSA", AccountType.TFSA)
    ledger = PortfolioLedger()
    ledger.record_many(
        [
            Transaction(
                timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
                account=acct,
                kind=TransactionKind.DEPOSIT,
                currency=Currency.CAD,
                amount="1000.00",
            ),
            Transaction(
                timestamp=datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
                account=acct,
                kind=TransactionKind.BUY,
                currency=Currency.CAD,
                ticker="VCN.TO",
                quantity=10,
                price="50.00",
                fee="5.00",
            ),
            Transaction(
                timestamp=datetime(2025, 1, 3, 10, 0, tzinfo=UTC),
                account=acct,
                kind=TransactionKind.SELL,
                currency=Currency.CAD,
                ticker="VCN.TO",
                quantity=10,
                price="55.00",
                fee="5.00",
            ),
        ]
    )

    pos = ledger.positions("TFSA")[0]
    assert pos.shares == Decimal("0")
    assert pos.book_cost_native == Decimal("0.00")
    assert pos.book_cost_cad == Decimal("0.00")
    # Net proceeds (550 - 5 = 545) - cost (500 + 5 = 505) = +40.00
    assert pos.realized_gain_cad == Decimal("40.00")
    assert ledger.cash_balance("TFSA", Currency.CAD) == Decimal("1040.00")
    assert ledger.non_reg_tax_acb == {}

    report = ledger.reconcile(
        MarketSnapshot(
            as_of=datetime(2025, 1, 4, 10, 0, tzinfo=UTC),
            cad_per_usd="1.40",
        )
    )
    assert report.all_cash_conserved
    assert report.to_holdings_frame().empty


def test_snapshot_validation_catches_missing_quote_overlap_and_stale_timestamp() -> (
    None
):
    acct = Account("TFSA", AccountType.TFSA)
    ledger = PortfolioLedger()
    ledger.record_many(
        [
            Transaction(
                timestamp=datetime(2025, 2, 1, 12, 0, tzinfo=UTC),
                account=acct,
                kind=TransactionKind.DEPOSIT,
                currency=Currency.CAD,
                amount="1000.00",
            ),
            Transaction(
                timestamp=datetime(2025, 2, 2, 12, 0, tzinfo=UTC),
                account=acct,
                kind=TransactionKind.BUY,
                currency=Currency.CAD,
                ticker="VFV.TO",
                quantity=5,
                price="100.00",
            ),
        ]
    )

    with pytest.raises(DataValidationError, match="both cad_prices and usd_prices"):
        MarketSnapshot(
            as_of=datetime(2025, 2, 3, 12, 0, tzinfo=UTC),
            cad_per_usd="1.40",
            cad_prices={"VFV.TO": "105.00"},
            usd_prices={"VFV.TO": "75.00"},
        )

    with pytest.raises(DataValidationError, match="Missing CAD quote"):
        ledger.reconcile(
            MarketSnapshot(
                as_of=datetime(2025, 2, 3, 12, 0, tzinfo=UTC),
                cad_per_usd="1.40",
            )
        )

    with pytest.raises(DataValidationError, match="cannot precede"):
        ledger.reconcile(
            MarketSnapshot(
                as_of=datetime(2025, 1, 15, 12, 0, tzinfo=UTC),
                cad_per_usd="1.40",
                cad_prices={"VFV.TO": "105.00"},
            )
        )


def test_account_cash_isolation_no_implicit_usd_cad_sweep() -> None:
    acct = Account("NonReg", AccountType.NON_REG)
    ledger = PortfolioLedger()
    ledger.record(
        Transaction(
            timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            account=acct,
            kind=TransactionKind.DEPOSIT,
            currency=Currency.CAD,
            amount="5000.00",
        )
    )
    # Buying a USD asset without first converting CAD -> USD must fail.
    with pytest.raises(DataValidationError, match="Insufficient USD cash"):
        ledger.record(
            Transaction(
                timestamp=datetime(2025, 1, 2, 10, 0, tzinfo=UTC),
                account=acct,
                kind=TransactionKind.BUY,
                currency=Currency.USD,
                ticker="VTI",
                quantity=10,
                price="250.00",
                fx_rate="1.40",
            )
        )


# ---------------------------------------------------------------------------
# End-to-end deterministic Canadian scenario
# ---------------------------------------------------------------------------


def test_canadian_accounting_scenario_reconciliation_and_rebalance_invariants() -> None:
    scenario = _load_scenario_module()
    ledger, snapshot = scenario.build_canadian_scenario()
    report = ledger.reconcile(snapshot)

    # 1. Exact conservation of cash in every (account_id, currency) bucket
    assert report.all_cash_conserved is True
    assert report.cash_conservation_residuals == {
        ("NonReg-Taxable", Currency.CAD): Decimal("0.00"),
        ("NonReg-Taxable", Currency.USD): Decimal("0.00"),
        ("TFSA-Main", Currency.CAD): Decimal("0.00"),
    }

    # 2. Exact account cash & valuation reconciliation
    tfsa = report.accounts["TFSA-Main"]
    non_reg = report.accounts["NonReg-Taxable"]

    assert tfsa.cash_cad == Decimal("1395.05")
    assert tfsa.cash_usd == Decimal("0.00")
    assert tfsa.holdings_market_value_cad == Decimal("8600.00")
    assert tfsa.total_value_cad == Decimal("9995.05")

    assert non_reg.cash_cad == Decimal("2335.00")
    assert non_reg.cash_usd == Decimal("1380.00")
    assert non_reg.cash_total_cad == Decimal("4301.50")  # 2335.00 + 1380 * 1.425
    assert non_reg.holdings_market_value_cad == Decimal("4022.25")
    assert non_reg.total_value_cad == Decimal("8323.75")

    # 3. Portfolio-wide totals & currency exposure identity
    assert report.total_cash_cad == Decimal("3730.05")
    assert report.total_cash_usd == Decimal("1380.00")
    assert report.total_cash_equivalent_cad == Decimal("5696.55")
    assert report.total_holdings_cad == Decimal("12622.25")
    assert report.total_portfolio_cad == Decimal("18318.80")
    assert report.currency_exposure_cad == {
        Currency.CAD: Decimal("13830.05"),
        Currency.USD: Decimal("4488.75"),
    }
    assert sum(report.currency_exposure_cad.values()) == report.total_portfolio_cad

    # 4. CRA Non-Registered ACB isolates TFSA holdings & preserves historical FX cost
    assert set(report.non_reg_tax_acb.keys()) == {"VFV.TO", "VTI"}
    assert "VCN.TO" not in report.non_reg_tax_acb
    assert report.non_reg_tax_acb["VFV.TO"].total_shares == pytest.approx(10.0)
    assert report.non_reg_tax_acb["VFV.TO"].total_cost == pytest.approx(1450.00)
    assert report.non_reg_tax_acb["VTI"].total_shares == pytest.approx(6.0)
    assert report.non_reg_tax_acb["VTI"].total_cost == pytest.approx(2373.03)
    assert report.non_reg_realized_gains_cad == Decimal("138.78")

    # 5. Reconciled holdings frame integrates cleanly with cash-flow rebalancing
    holdings_by_type = report.to_holdings_frame(by="account_type")
    assert holdings_by_type["current_value"].sum() == pytest.approx(12622.25)

    _, rebalance_result = scenario.run_canadian_scenario()
    total_allocated = float(rebalance_result.buy_orders["buy_amount"].sum())
    assert total_allocated + rebalance_result.cash_remaining == pytest.approx(1500.00)
    assert set(rebalance_result.buy_orders["ticker"]) == {"VCN.TO", "VTI"}
    assert rebalance_result.taxable_sell_triggered is True
    assert list(rebalance_result.sell_orders["account"]) == ["NON_REG"]

    # 6. Downstream account_types handoff in example formats cleanly
    summary_text = scenario.format_scenario_summary(report, rebalance_result)
    assert "All Cash Conserved Exact:  True" in summary_text
    assert "Explicit Unresolved Accounting-Policy Choices" in summary_text

    # Sanity-check direct call with alternative target weights (no taxable sells)
    balanced_res = allocate_contribution_cash_flow(
        cash_amount=1500.00,
        current_balances=holdings_by_type,
        target_weights={"VFV.TO": 0.55, "VCN.TO": 0.25, "VTI": 0.20},
        account_types={
            "VFV.TO": AccountType.TFSA.value,
            "VCN.TO": AccountType.TFSA.value,
            "VTI": AccountType.NON_REG.value,
        },
        opportunity_scores={"VFV.TO": 0.60, "VCN.TO": 0.60, "VTI": 0.60},
    )
    assert balanced_res.buy_orders["buy_amount"].sum() == pytest.approx(1500.00)
    assert not balanced_res.taxable_sell_triggered
