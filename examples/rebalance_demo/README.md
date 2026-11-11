# Five-minute demo: a buy-only rebalance plan

> **Synthetic data.** `SYNA.TO`–`SYND.TO`, their prices, the holdings, and the
> target weights are invented for this example. Nothing is downloaded, nothing
> here is market data, and nothing here is investment advice.

**The question PySharpe answers here:** "I hold four funds that have drifted
from my target mix, and I have $2,150 of new cash. What whole-share buy orders
move me back toward target without selling anything?"

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync

uv run pysharpe rebalance \
  --portfolio synthetic_demo \
  --holdings-csv examples/rebalance_demo/holdings.csv \
  --new-cash 2150 \
  --export-dir examples/rebalance_demo/exports \
  --config examples/rebalance_demo/config.json \
  --include-zero-buys
```

Expected output (stdout):

```text
Rebalance plan for synthetic_demo
New cash: $2,150.00
Unallocated (fractional leftover): $85.00

 Ticker   Price Target Current  Score     Buy $ Buy Shares
SYNC.TO  $40.00 20.00%  10.00% 0.8000 $1,040.00    26.0000
SYND.TO $100.00 10.00%   5.00% 0.5000   $500.00     5.0000
SYNB.TO  $25.00 30.00%  25.00% 0.5000   $525.00    21.0000
SYNA.TO  $50.00 40.00%  60.00% 0.2000     $0.00     0.0000
```

The first run takes a few seconds while Python imports the statistics stack.
A `MigrationWarning` from the third-party `arviz` package may appear on
stderr; it does not affect the result.

Then verify the numbers against an independent calculation:

```bash
uv run python examples/rebalance_demo/check_example.py
```

`check_example.py` does not import PySharpe. It recomputes the plan from the
input files with exact fractions (standard library only), runs the installed
`pysharpe` console script as a subprocess, prints expected versus observed for
25 values, and exits non-zero on any mismatch. The last line should read
`25/25 values match.`

## Inputs

| File | Role |
| ---- | ---- |
| `exports/synthetic_demo_weights.txt` | Target weights, in the format `pysharpe optimise` writes |
| `exports/synthetic_demo_collated.csv` | Five days of prices; the last row is the execution price |
| `holdings.csv` | Current position in shares |
| `config.json` | `allow_fractional: false` (whole shares), no FX fee |

## The arithmetic, by hand

Portfolio value is 120×50 + 100×25 + 25×40 + 5×100 = **$10,000**.

| Ticker | Shares × price | Value | Current | Target | Gap (target − current, floored at 0) |
| ------ | -------------- | ----: | ------: | -----: | -----------------------------------: |
| SYNA.TO | 120 × $50  | $6,000 | 60% | 40% | 0 |
| SYNB.TO | 100 × $25  | $2,500 | 25% | 30% | 5 pts |
| SYNC.TO | 25 × $40   | $1,000 | 10% | 20% | 10 pts |
| SYND.TO | 5 × $100   |   $500 |  5% | 10% | 5 pts |

Gaps total 20 points. New cash is split in proportion to each gap, then
rounded **down** to whole shares at the latest price:

| Ticker | Share of cash | Dollars | ÷ price | Whole shares | Spent | Unspent |
| ------ | ------------: | ------: | ------: | -----------: | ----: | ------: |
| SYNB.TO | 5/20  | $537.50   | 21.5   | **21** | **$525**   | $12.50 |
| SYNC.TO | 10/20 | $1,075.00 | 26.875 | **26** | **$1,040** | $35.00 |
| SYND.TO | 5/20  | $537.50   | 5.375  | **5**  | **$500**   | $37.50 |
| SYNA.TO | 0     | $0        | 0      | **0**  | **$0**     | $0 |

Spent $2,065; **$85 left over**; $2,065 + $85 = $2,150, so cash is conserved.

**Score** is the documented 60/40 blend: 0.6 × (gap ÷ largest gap) + 0.4 ×
valuation. This example supplies no valuation data, so valuation is the neutral
0.5 for every fund: SYNC 0.6×1 + 0.2 = 0.8; SYNB and SYND 0.6×0.5 + 0.2 = 0.5;
SYNA 0.2.

## Expected versus observed

| Ticker | Expected buy | Observed buy | Expected shares | Observed shares | Expected score | Observed score |
| ------ | -----------: | -----------: | --------------: | --------------: | -------------: | -------------: |
| SYNA.TO | $0.00     | $0.00     | 0  | 0  | 0.2000 | 0.2000 |
| SYNB.TO | $525.00   | $525.00   | 21 | 21 | 0.5000 | 0.5000 |
| SYNC.TO | $1,040.00 | $1,040.00 | 26 | 26 | 0.8000 | 0.8000 |
| SYND.TO | $500.00   | $500.00   | 5  | 5  | 0.5000 | 0.5000 |
| Leftover cash | $85.00 | $85.00 | | | | |

The same figures are pinned by `tests/test_rebalance_demo_example.py`.

## What this does and does not show

- It shows the execution step of the pipeline: saved target weights + latest
  prices + current holdings → scored, buy-only, whole-share orders.
- It never sells. SYNA.TO stays overweight (60% falls to about 49% only through
  dilution by the new cash).
- Rounding leftovers are reported, not redeployed. The $85 here would cover
  two more SYNC.TO shares; PySharpe leaves that decision to the investor.
- Gaps are measured against the portfolio **before** the contribution, which is
  PySharpe's documented allocation heuristic rather than an exact
  "distance to target after the deposit" solve.
- The target weights are supplied directly. Producing them with
  `pysharpe optimise` needs downloaded price history, so it is out of scope for
  an offline example.
