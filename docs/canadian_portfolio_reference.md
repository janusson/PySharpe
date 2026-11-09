# Canadian Portfolio Reference Example

A single reproducible, end-to-end walkthrough of a synthetic Canadian taxpayer
with three accounts: CAD and USD cash, per-account book cost, and a
taxpayer-level pooled tax ACB. It exists to show how the ledgers relate, not to
model a market.

**Deterministic data only.** No network calls, no brokerage access, no real
securities, no account credentials, and no personal portfolio information.

## Files

| Path | Role |
|------|------|
| `examples/canadian_portfolio_reference.py` | Self-contained executable example: account/cash ledger, scenario, reconciliation report |
| `tests/test_reference_example.py` | Focused integration test with independently derived expected values |

The example borrows one accepted implementation from the package — the CRA
pooled ACB ledger `pysharpe.guardrails.tax_compliance.ACBTracker` — and supplies
its own per-account CAD/USD cash and book-cost ledger, because the package has
no account-level cash ledger.

## How to run

```bash
# prints the reconciliation report
uv run python examples/canadian_portfolio_reference.py

# integration tests
uv run pytest tests/test_reference_example.py
```

If `pytest` is missing, install the development extras first:
`uv sync --extra dev`.

## Scenario

| Account | Action | Amount |
|---------|--------|--------|
| `TFSA-A` | Contribute CAD; convert to USD | CAD 10,000; CAD 6,750 → USD 5,000 |
| `TFSA-A` | Buy 10 units of the USD security at USD 50 | USD 500 |
| `TAXABLE-A` | Contribute CAD; buy 100 shares of the CAD security at CAD 100 | CAD 10,000 |
| `TAXABLE-B` | Contribute CAD; buy 100 shares of the identical CAD security at CAD 200 | CAD 20,000 |
| `TAXABLE-A` | Later sell the 100 shares at CAD 150 | CAD 15,000 |

Both taxable purchases settle before the taxable sale. There are no fees, no
price changes other than the stated purchase and sale prices, and no FX changes.

Valuation uses explicit synthetic prices — CAD security CAD 150/share, USD
security USD 50/share, and CAD 1.35 per USD — giving a total portfolio value of
**CAD 40,000**.

## Expected output

```text
PySharpe — Canadian portfolio reference example (synthetic data)
==============================================================================
prices: SYN-CAD.TO CAD 150.00 | SYN-USD USD 50.00
valuation FX: 1.35 CAD per USD

ACCOUNT RECONCILIATION (CAD unless the column says otherwise)
Account     Type           CAD cash     USD cash   Securities    Book cost  Total value     Contrib.     Realized
TFSA-A      TFSA           3,250.00     4,500.00       675.00       675.00    10,000.00    10,000.00         0.00
TAXABLE-A   TAXABLE       15,000.00         0.00         0.00         0.00    15,000.00    10,000.00     5,000.00
TAXABLE-B   TAXABLE            0.00         0.00    15,000.00    20,000.00    15,000.00    20,000.00         0.00
TOTAL                     18,250.00     4,500.00    15,675.00    20,675.00    40,000.00    40,000.00     5,000.00

CURRENCY BREAKDOWN (CAD equivalent at the valuation FX rate)
Ccy            Cash   Cash (CAD)   Securities   Sec. (CAD)  Total (CAD)
CAD       18,250.00    18,250.00    15,000.00    15,000.00    33,250.00
USD        4,500.00     6,075.00       500.00       675.00     6,750.00
TOTAL                  24,325.00                 15,675.00    40,000.00

TAXPAYER TAX ACB — pooled SYN-CAD.TO, non-registered only
  original pooled cost        30,000.00
  pooled quantity                   200
  ACB per share                  150.00
  realized taxable gain            0.00
  remaining tax ACB           15,000.00

  TAXABLE-B book cost is 20,000.00 on 100 shares, while its pooled tax ACB is 15,000.00.
  Account book cost records what an account paid; tax ACB pools
  identical property across all non-registered accounts.

CASH CONSERVATION (contributions + realized − cash − book cost)
  TFSA-A 0.00   TAXABLE-A 0.00   TAXABLE-B 0.00

TOTAL PORTFOLIO VALUE 40,000.00 CAD
```

## Account book cost versus tax ACB

**Account book cost** answers *what did this account pay for the units it holds?*
It is local to the account, recorded in CAD at the acquisition FX rate, and it is
the figure a brokerage statement reports. It is the right basis for performance
reporting.

**Tax ACB** answers *what cost basis does the CRA allow on a disposition?* Under
ITA s. 47(1), identical property held by one taxpayer is a **single pool spanning
all non-registered accounts**. Acquisitions add cost to the pool, dispositions
remove cost pro rata at the pooled average, and the realized gain is proceeds
minus the removed pooled cost. Registered accounts (TFSA, RRSP, FHSA, RESP) are
outside the pool entirely — which is why `TFSA-A`'s CAD 675 of book cost never
appears in the tax ledger.

In this example the two concepts deliberately disagree:

| Account | Shares held | Account book cost | Share of pooled tax ACB |
|---------|-------------|-------------------|-------------------------|
| `TAXABLE-A` | 0 | CAD 0 (fully sold) | CAD 0 (released on sale) |
| `TAXABLE-B` | 100 | CAD 20,000 | CAD 15,000 |
| Pool before the sale | 200 | CAD 30,000 | CAD 30,000 |

`TAXABLE-A` bought its 100 shares at CAD 100 and sold at CAD 150, an
account-level gain of CAD 5,000. Because the pool's average cost is CAD 150/share
(100 shares at CAD 100 plus 100 shares at CAD 200), the **taxable** gain is zero,
and the CAD 15,000 of remaining tax ACB belongs proportionally to `TAXABLE-B`'s
100 shares even though that account paid CAD 20,000.

Practical consequence: report account book cost for performance, and pooled tax
ACB for tax. Never treat an account-level realized gain as a taxable gain when
identical property is held elsewhere.

## Validation performed

| Validation | Test |
|------------|------|
| Cash conservation per account and portfolio | `test_cash_conservation` |
| Holdings reconciliation against the journal | `test_holdings_reconcile_by_account` |
| FX conversion consistency | `test_fx_conversion_consistency`, `test_usd_purchase_uses_acquisition_fx` |
| Tax ACB pooling | `test_taxpayer_tax_acb_pooling`, `test_tfsa_is_excluded_from_tax_acb` |
| Realized gains | `test_realized_gains` |
| Account and portfolio valuation | `test_account_and_portfolio_valuation`, `test_currency_breakdown` |
| No double-counted contributions or withdrawals | `test_contributions_are_not_double_counted` |
| Deterministic replay | `test_deterministic_replay` |
| Ledger guards (no negative cash, no oversell) | `test_purchase_beyond_available_cash_is_rejected`, `test_conversion_beyond_available_cash_is_rejected`, `test_oversell_is_rejected` |

Cash conservation is asserted as

```text
contributions + realized gain
    = cash_CAD + (cash_USD × FX) + book cost
```

with a residual of exactly zero for every account. With a moving FX rate the
residual would additionally contain the unrealized FX gain on foreign-currency
cash; this example holds FX constant.

## Scope

- No dashboard, no import/brokerage integration, no new dependencies.
- Fees, FX changes, return-of-capital distributions, and superficial-loss rules
  are out of scope; the scenario is deliberately minimal so every number can be
  checked by hand.
