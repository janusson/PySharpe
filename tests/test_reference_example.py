"""Integration tests for the Canadian portfolio reference example.

Every expected value below is written out from the scenario description in
``examples/canadian_portfolio_reference.py`` rather than computed with the
example's own helpers, so a broken ledger cannot silently agree with itself.
"""

from __future__ import annotations

import pytest

from examples.canadian_portfolio_reference import (
    CAD_PER_USD,
    CAD_SECURITY,
    PURCHASE_DATE,
    SALE_DATE,
    USD_SECURITY,
    InvestmentAccount,
    run_reference_scenario,
)

# Scenario inputs ------------------------------------------------------------
TFSA_DEPOSIT_CAD = 10_000.0
TFSA_CONVERSION_CAD = 6_750.0
TFSA_USD_RECEIVED = 5_000.0
USD_UNITS = 10.0
USD_PRICE = 50.0
USD_SECURITY_COST = USD_UNITS * USD_PRICE  # 500 USD

TAXABLE_A_DEPOSIT_CAD = 10_000.0
TAXABLE_A_SHARES = 100.0
TAXABLE_A_PRICE = 100.0
SALE_PRICE = 150.0

TAXABLE_B_DEPOSIT_CAD = 20_000.0
TAXABLE_B_SHARES = 100.0
TAXABLE_B_PRICE = 200.0

CAD_MARKET_PRICE = 150.0

# Independently derived expectations ----------------------------------------
TFSA_CAD_CASH = TFSA_DEPOSIT_CAD - TFSA_CONVERSION_CAD  # 3,250
TFSA_USD_CASH = TFSA_USD_RECEIVED - USD_SECURITY_COST  # 4,500
TFSA_BOOK_COST_CAD = USD_SECURITY_COST * CAD_PER_USD  # 675
TAXABLE_A_CASH = TAXABLE_A_SHARES * SALE_PRICE  # 15,000
TAXABLE_A_ACCOUNT_GAIN = TAXABLE_A_SHARES * (SALE_PRICE - TAXABLE_A_PRICE)  # 5,000
TAXABLE_B_BOOK_COST = TAXABLE_B_SHARES * TAXABLE_B_PRICE  # 20,000

POOLED_COST = TAXABLE_A_SHARES * TAXABLE_A_PRICE + TAXABLE_B_SHARES * TAXABLE_B_PRICE
POOLED_SHARES = TAXABLE_A_SHARES + TAXABLE_B_SHARES
POOLED_ACB_PER_SHARE = POOLED_COST / POOLED_SHARES  # 150
TAXABLE_GAIN = TAXABLE_A_SHARES * (SALE_PRICE - POOLED_ACB_PER_SHARE)  # 0
REMAINING_TAX_ACB = POOLED_COST - TAXABLE_A_SHARES * POOLED_ACB_PER_SHARE  # 15,000

TFSA_TOTAL = (
    TFSA_CAD_CASH + TFSA_USD_CASH * CAD_PER_USD + USD_UNITS * USD_PRICE * CAD_PER_USD
)  # 10,000
TAXABLE_A_TOTAL = TAXABLE_A_CASH  # 15,000
TAXABLE_B_TOTAL = TAXABLE_B_SHARES * CAD_MARKET_PRICE  # 15,000
PORTFOLIO_TOTAL = TFSA_TOTAL + TAXABLE_A_TOTAL + TAXABLE_B_TOTAL  # 40,000
TOTAL_CASH_CAD = TFSA_CAD_CASH + TAXABLE_A_CASH + TFSA_USD_CASH * CAD_PER_USD  # 24,325
TOTAL_SECURITIES_CAD = (
    USD_UNITS * USD_PRICE * CAD_PER_USD + TAXABLE_B_SHARES * CAD_MARKET_PRICE
)  # 15,675

CONTRIBUTIONS = {
    "TFSA-A": TFSA_DEPOSIT_CAD,
    "TAXABLE-A": TAXABLE_A_DEPOSIT_CAD,
    "TAXABLE-B": TAXABLE_B_DEPOSIT_CAD,
}


@pytest.fixture(scope="module")
def scenario():
    return run_reference_scenario()


# -- Holdings and cash -------------------------------------------------------


def test_account_cash_by_currency(scenario):
    accounts = scenario.accounts
    assert accounts["TFSA-A"].cash["CAD"] == TFSA_CAD_CASH
    assert accounts["TFSA-A"].cash["USD"] == TFSA_USD_CASH
    assert accounts["TAXABLE-A"].cash["CAD"] == TAXABLE_A_CASH
    assert accounts["TAXABLE-A"].cash["USD"] == 0.0
    assert accounts["TAXABLE-B"].cash["CAD"] == 0.0
    assert accounts["TAXABLE-B"].cash["USD"] == 0.0


def test_holdings_reconcile_by_account(scenario):
    accounts = scenario.accounts
    assert accounts["TFSA-A"].quantities == {USD_SECURITY: USD_UNITS}
    assert accounts["TAXABLE-A"].quantities == {}
    assert accounts["TAXABLE-B"].quantities == {CAD_SECURITY: TAXABLE_B_SHARES}

    for name, account in accounts.items():
        net: dict[str, float] = {}
        for entry in account.journal:
            if entry.kind == "BUY":
                net[entry.ticker] = net.get(entry.ticker, 0.0) + entry.quantity
            elif entry.kind == "SELL":
                net[entry.ticker] = net.get(entry.ticker, 0.0) - entry.quantity
        settled = {ticker: qty for ticker, qty in net.items() if qty != 0.0}
        assert settled == account.quantities, name


def test_account_book_cost(scenario):
    accounts = scenario.accounts
    assert accounts["TFSA-A"].book_cost_total_cad() == TFSA_BOOK_COST_CAD
    assert accounts["TAXABLE-A"].book_cost_total_cad() == 0.0
    assert accounts["TAXABLE-B"].book_cost_total_cad() == TAXABLE_B_BOOK_COST
    assert accounts["TAXABLE-B"].book_cost_cad[CAD_SECURITY] == TAXABLE_B_BOOK_COST


# -- FX ----------------------------------------------------------------------


def test_fx_conversion_consistency(scenario):
    tfsa = scenario.accounts["TFSA-A"]
    conversions = [e for e in tfsa.journal if e.kind == "FX_CONVERSION"]
    assert len(conversions) == 2

    cad_leg = next(e for e in conversions if e.currency == "CAD")
    usd_leg = next(e for e in conversions if e.currency == "USD")

    assert cad_leg.amount == -TFSA_CONVERSION_CAD
    assert usd_leg.amount == TFSA_USD_RECEIVED
    assert -cad_leg.amount / usd_leg.amount == pytest.approx(CAD_PER_USD)
    # Both legs carry the same CAD value, in opposite directions.
    assert cad_leg.cad_value == -TFSA_CONVERSION_CAD
    assert usd_leg.cad_value == TFSA_CONVERSION_CAD

    # The USD received funds the position and the remaining USD cash exactly.
    assert tfsa.cash["USD"] + USD_SECURITY_COST == TFSA_USD_RECEIVED


def test_usd_purchase_uses_acquisition_fx(scenario):
    tfsa = scenario.accounts["TFSA-A"]
    usd_cost = tfsa.book_cost_cad[USD_SECURITY]
    assert usd_cost == pytest.approx(USD_SECURITY_COST * CAD_PER_USD)
    assert usd_cost == TFSA_BOOK_COST_CAD


# -- Tax ACB and realized gains ---------------------------------------------


def test_taxpayer_tax_acb_pooling(scenario):
    pooled = scenario.pooled_before_sale
    assert pooled.total_cost == POOLED_COST
    assert pooled.total_shares == POOLED_SHARES
    assert pooled.acb_per_share == POOLED_ACB_PER_SHARE

    assert scenario.tax_acb.get_shares(CAD_SECURITY) == (
        POOLED_SHARES - TAXABLE_A_SHARES
    )
    assert scenario.tax_acb.get_total_acb(CAD_SECURITY) == REMAINING_TAX_ACB
    assert scenario.tax_acb.get_acb_per_share(CAD_SECURITY) == POOLED_ACB_PER_SHARE


def test_realized_gains(scenario):
    assert scenario.tax_realized_gain_cad == pytest.approx(TAXABLE_GAIN)
    assert scenario.tax_realized_gain_cad == 0.0
    assert scenario.accounts["TAXABLE-A"].realized_gain_cad == pytest.approx(
        TAXABLE_A_ACCOUNT_GAIN
    )
    assert scenario.accounts["TFSA-A"].realized_gain_cad == 0.0
    assert scenario.accounts["TAXABLE-B"].realized_gain_cad == 0.0

    sale = scenario.tax_acb.trades[-1]
    assert sale.action == "SELL"
    assert sale.shares == TAXABLE_A_SHARES
    assert sale.total_amount == TAXABLE_A_SHARES * SALE_PRICE
    assert sale.realized_gain_loss == 0.0


def test_book_cost_and_tax_acb_diverge_for_taxable_b(scenario):
    taxable_b = scenario.accounts["TAXABLE-B"]
    assert taxable_b.book_cost_total_cad() == TAXABLE_B_BOOK_COST
    assert scenario.tax_acb_remaining_cad() == REMAINING_TAX_ACB
    assert taxable_b.book_cost_total_cad() != scenario.tax_acb_remaining_cad()


def test_tfsa_is_excluded_from_tax_acb(scenario):
    assert {trade.ticker for trade in scenario.tax_acb.trades} == {CAD_SECURITY}
    assert all(trade.account_type == "NON_REG" for trade in scenario.tax_acb.trades)


# -- Valuation ---------------------------------------------------------------


def test_account_and_portfolio_valuation(scenario):
    accounts = scenario.account_table().set_index("account")

    assert accounts.loc["TFSA-A", "total_value_cad"] == pytest.approx(TFSA_TOTAL)
    assert accounts.loc["TAXABLE-A", "total_value_cad"] == pytest.approx(
        TAXABLE_A_TOTAL
    )
    assert accounts.loc["TAXABLE-B", "total_value_cad"] == pytest.approx(
        TAXABLE_B_TOTAL
    )

    assert scenario.cash_total_cad() == pytest.approx(TOTAL_CASH_CAD)
    assert scenario.security_total_cad() == pytest.approx(TOTAL_SECURITIES_CAD)
    assert scenario.total_value_cad() == pytest.approx(PORTFOLIO_TOTAL)
    assert scenario.total_value_cad() == pytest.approx(
        sum(
            account.total_value_cad(scenario.market)
            for account in scenario.accounts.values()
        )
    )


def test_currency_breakdown(scenario):
    table = scenario.currency_table().set_index("currency")
    assert table.loc["CAD", "cash_native"] == pytest.approx(
        TFSA_CAD_CASH + TAXABLE_A_CASH
    )
    assert table.loc["USD", "cash_native"] == pytest.approx(TFSA_USD_CASH)
    assert table.loc["USD", "cash_cad"] == pytest.approx(TFSA_USD_CASH * CAD_PER_USD)
    assert table.loc["USD", "securities_cad"] == pytest.approx(
        USD_UNITS * USD_PRICE * CAD_PER_USD
    )
    assert table["total_cad"].sum() == pytest.approx(PORTFOLIO_TOTAL)


# -- Conservation and contributions -----------------------------------------


def test_cash_conservation(scenario):
    residuals = scenario.conservation_table()
    assert list(residuals["residual_cad"]) == [0.0, 0.0, 0.0]

    for name, account in scenario.accounts.items():
        # Posted CAD value of every movement equals settled cash at the same FX.
        assert account.journal_cash_cad() == pytest.approx(
            account.cash_value_cad(scenario.market)
        ), name


def test_contributions_are_not_double_counted(scenario):
    kinds = {"CONTRIBUTION", "FX_CONVERSION", "BUY", "SELL"}
    for name, account in scenario.accounts.items():
        contributions = [e for e in account.journal if e.kind == "CONTRIBUTION"]
        assert len(contributions) == 1, name
        assert contributions[0].amount == CONTRIBUTIONS[name]
        assert account.contributions_cad == CONTRIBUTIONS[name]
        assert {entry.kind for entry in account.journal} <= kinds, name

    assert sum(a.contributions_cad for a in scenario.accounts.values()) == sum(
        CONTRIBUTIONS.values()
    )
    # Contributions reconcile the accounts but are not added to portfolio value.
    assert scenario.total_value_cad() == pytest.approx(PORTFOLIO_TOTAL)


# -- Determinism and report --------------------------------------------------


def test_deterministic_replay():
    first = run_reference_scenario()
    second = run_reference_scenario()
    assert first.snapshot() == second.snapshot()


def test_report_reconciles_every_account(scenario):
    report = scenario.report()
    for fragment in (
        "ACCOUNT RECONCILIATION",
        "TFSA-A",
        "TAXABLE-A",
        "TAXABLE-B",
        "CURRENCY BREAKDOWN",
        "TAXPAYER TAX ACB",
        "CASH CONSERVATION",
        "40,000.00",
        "15,000.00",
        "20,000.00",
        "30,000.00",
    ):
        assert fragment in report


# -- Ledger guards -----------------------------------------------------------


def test_purchase_beyond_available_cash_is_rejected():
    account = InvestmentAccount("TEST", account_type="TAXABLE")
    account.contribute(1_000.0, PURCHASE_DATE)
    with pytest.raises(ValueError, match="cannot fund"):
        account.buy(CAD_SECURITY, 20.0, 100.0, "CAD", CAD_PER_USD, PURCHASE_DATE)


def test_conversion_beyond_available_cash_is_rejected():
    account = InvestmentAccount("TEST", account_type="TAXABLE")
    account.contribute(1_000.0, PURCHASE_DATE)
    with pytest.raises(ValueError, match="cannot convert"):
        account.convert(1_500.0, "CAD", "USD", CAD_PER_USD, PURCHASE_DATE)


def test_oversell_is_rejected():
    account = InvestmentAccount("TEST", account_type="TAXABLE")
    account.contribute(1_000.0, PURCHASE_DATE)
    account.buy(CAD_SECURITY, 5.0, 100.0, "CAD", CAD_PER_USD, PURCHASE_DATE)
    with pytest.raises(ValueError, match="cannot sell"):
        account.sell(CAD_SECURITY, 6.0, 150.0, "CAD", CAD_PER_USD, SALE_DATE)
