# Portfolio X-ray

`portfolio_xray` reads a [holdings snapshot](interop.md) and reports what the
current shape of the portfolio *implies* — the analysis a record-keeper does not
do. It is deterministic, offline, and needs nothing beyond the snapshot.

## What it reports

| Section | Content |
|---------|---------|
| Positions | Per-account value in CAD and share of securities value |
| Currency exposure | CAD vs foreign exposure in CAD, cash included |
| Concentration | Largest position weight and effective holdings (`1 / Σw²`) |
| Identical exposure | The same bet held more than once, across accounts |
| Tax location | Withholding drag in TFSA/FHSA and income-tax drag in non-registered accounts, in basis points |

## Usage

```python
from datetime import date
from pysharpe import portfolio_xray, read_holdings_snapshot
from pysharpe.optimization.tax_location import AssetTaxCharacteristics

snapshot = read_holdings_snapshot(
    "positions.csv", as_of=date(2025, 3, 31), fx_rates={"USD": 1.36}
)

report = portfolio_xray(
    snapshot,
    account_types={"TFSA-A": "TFSA", "TAXABLE-A": "NON_REG"},
    asset_characteristics={
        "VFV.TO": AssetTaxCharacteristics(
            "VFV.TO",
            dividend_yield=0.012,
            is_cad_wrapped_us_equity=True,
            income_frac_eligible_dividends=0.0,
            income_frac_foreign_income=1.0,
        ),
    },
)
print(report.report())
report.to_frame()   # one row per finding
```

## Findings and their rules

| Code | Severity | Rule |
|------|----------|------|
| `identical_exposure` | warning | Two or more `(account, ticker)` pairs fall in one identical-property compound — different tickers tracking the same index, or the same ticker in several accounts |
| `concentration` | warning | Largest position exceeds `concentration_threshold` (default 25% of securities value) |
| `withholding_drag` | warning | The asset loses US withholding tax in the wrapper it is held in (TFSA/FHSA always; CAD-wrapped US equity in every wrapper) |
| `income_tax_drag` | info | Non-registered income is taxed; the drag is reported as a share of the income component |
| `unassessed_asset` | info | No `AssetTaxCharacteristics` supplied, so drag is **unknown**, not zero |

Identical property is resolved with the same map that drives
superficial-loss compliance
(`build_default_identical_map`), so the X-ray and the tax guardrail cannot
disagree about what counts as the same security.

## Assumptions you can see

- **Account types are explicit.** Every account in the snapshot must be declared;
  an unlisted or unrecognised label raises rather than defaulting to taxable.
- **The marginal rate is printed.** Without a `tax_profile`, the X-ray uses a 40%
  marginal rate and states it in the report header.
- **Drags are yield-based estimates in basis points**, computed by
  `AssetLocationEngine`, not tax advice and not a projection of returns.
- **`MER` is not deducted here.** Expense drag is a fund property, not an account
  property; it belongs with optimisation and reporting.

## Limits

- Identical property comes from a curated map of common Canadian ETF pairs. A
  ticker outside it is treated as unique — check `identical_groups` before
  trusting a clean report.
- No look-through to fund constituents; two different indices can still overlap.
- Cash is reported but never taxed or location-scored.

## Tests

```bash
uv run pytest tests/test_analysis_xray.py
```
