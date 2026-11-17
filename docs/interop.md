# Holdings Import (Interop)

PySharpe is an analytics and execution-planning layer, not a book of record.
This page documents the boundary that follows from that, and the import path
that respects it.

## Division of responsibility

| Question | Owner |
|----------|-------|
| What do I own, what did I pay, which dividends landed? | Your record-keeper (Sharesight, broker statements, a spreadsheet) |
| What should I buy next, in which account, and is my tax location sane? | PySharpe |

PySharpe imports a **point-in-time snapshot** — positions, cash, and cost basis
when the export supplies it — and never reconstructs history from it.
Transactions, dividends, distributions, DRIP, splits, and tax slips stay with
the system that produced the file.

## Canonical snapshot schema

| Canonical | Type | Meaning |
|-----------|------|---------|
| `ticker` | str | Instrument identifier exactly as exported |
| `account` | str | Account label from the exporting system |
| `quantity` | float | Units held; strictly positive |
| `price` | float | Price per unit in `currency` at the snapshot date |
| `currency` | str | ISO 4217 code of `price`; defaults to `CAD` |
| `cost_basis` | float | Optional CAD cost basis; blank means **unknown** |

CSV is read as text and a `columns` mapping renames your export's headers onto
these names, so any vendor format works without a new adapter.

### Header inference

Omit `columns` and `infer_column_mapping` matches the header row by documented
synonym, ignoring case, punctuation, and parenthetical qualifiers such as
`(CA$)`:

| Canonical | Recognised labels (normalised) |
|-----------|-------------------------------|
| `ticker` | ticker, symbol, instrument, instrument code, security |
| `account` | account, holding account, account name, portfolio account |
| `quantity` | quantity, units, units held, quantity held, shares |
| `price` | price, current price, price per unit, market price |
| `currency` | currency, currency code, ccy |
| `cost_basis` | cost basis, cost base, book cost, acb, total cost |

Ambiguous matches raise instead of guessing; pass `columns` (or
`infer_column_mapping(..., overrides={...})`) to resolve them. When the header
row changes between exports, add a synonym rather than writing a new adapter.

**`ticker` must be the ticker PySharpe prices with**, which is the yfinance
convention: a TSX listing is `VFV.TO`, not `VFV`. Exports usually append a
market code instead (`VFV.TSE`, `VOO.NYSE`, or `VFV | TSE` in dashboard
labels). Pass the shipped normaliser to convert them:

```python
from pysharpe import normalise_ticker, read_holdings_snapshot

snapshot = read_holdings_snapshot(
    "export.csv", as_of=date(2025, 3, 31), fx_rates={"USD": 1.36},
    ticker_transform=normalise_ticker,
)
```

`normalise_ticker` maps known market codes (`TSE`/`TSX` → `.TO`, `CVE`/`TSXV`
→ `.V`, US venues → no suffix) and leaves everything else untouched, including
share classes (`BRK.B`) and identifiers that already carry a yfinance suffix.
Unknown codes are never guessed: supply `markets={...}` to extend the map. Every
rename is logged, and duplicate detection runs on the normalized tickers, so
`VFV.TSE` and `VFV.TO` in one account fail loudly instead of double-counting a
position.

### Report-shaped exports

Performance and holdings reports usually embed group headers, per-currency
subtotals, and a grand total in the same table:

```text
Instrument,Holding Account,Units Held,Current Price,Currency,Value,Capital Gains
NYSE,,,,,,
EXAMPLE.NYSE,Taxable,2,100.00,USD,"200.00","20.00"
Total,,,,,"200.00","20.00"
TSE,,,,,,
EXAMPLE.TSE,Taxable,4,50.00,CAD,"200.00","15.00"
Total,,,,,"200.00","15.00"
Grand Total,,,,,"400.00","35.00"
```

A row with neither a quantity nor a price cannot be a position, so group
headers and totals are skipped. Everything else must parse: a row with units
but no price, or numbers without a ticker, raises with its row number. Report
columns PySharpe does not model (`Value`, `Capital Gains`, `Dividends`) are
ignored — value is recomputed from quantity × price × FX so the snapshot stays
internally consistent.

## Importing

```python
from datetime import date
from pathlib import Path

from pysharpe import read_holdings_snapshot

snapshot = read_holdings_snapshot(
    Path("positions-export.csv"),
    columns={
        "ticker": "Instrument",
        "account": "Holding Account",
        "quantity": "Units Held",
        "price": "Current Price",
        "currency": "Currency",
        "cost_basis": "Cost Basis",
    },
    as_of=date(2025, 3, 31),
    fx_rates={"USD": 1.35},
    cash={"TFSA-A": {"CAD": 250.0}},
)

snapshot.total_value_cad()
snapshot.missing_cost_basis()   # positions whose basis the export omitted
```

Header names differ between vendors and change over time; the mapping above is
an illustration of the shape, not a claim about any one product's current
export. Check your file's header row and map it accordingly.

## Downstream contracts

`snapshot.current_balances()` returns `ticker` / `account` / `current_value`
(CAD), the schema
`pysharpe.execution.cash_flow_rebalance.allocate_contribution_cash_flow`
takes.

`snapshot.holdings_frame()` returns `ticker` / `account` / `shares` /
`current_value` / `currency`, which the existing CLI already consumes:

```python
snapshot.holdings_frame().to_csv("data/holdings.csv", index=False)
```

```bash
uv run pysharpe rebalance --portfolio cad_growth \
    --holdings-csv data/holdings.csv --new-cash 5000
```

## What the import refuses to guess

- **Blank cost basis stays unknown.** It is never coerced to `0.0`;
  `missing_cost_basis()` lists what is missing.
- **Missing FX coverage is rejected**, naming the affected positions. Rates are
  supplied by the caller and never backfilled or inferred.
- **Duplicate `(account, ticker)` rows fail.** Aggregate them in the exporting
  system; PySharpe never merges rows whose identity it cannot verify.
- **Non-positive quantities or prices fail**, as does a blank `ticker` or
  `account`.
- **No price refresh.** Prices are exactly what the export says; `as_of` is
  caller-supplied.

## Scope

Not in this slice: transaction histories, dividends, corporate actions,
per-year realized gains, persisted portfolios, and vendor-specific adapters.
Persistence is tracked as #43/#79. Account book cost and pooled tax ACB remain
in `pysharpe.execution.tax_tracker` and
`pysharpe.guardrails.tax_compliance`; see
[Canadian Portfolio Reference](canadian_portfolio_reference.md) for how those
ledgers fit together.
