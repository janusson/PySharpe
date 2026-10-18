# Tactical research engine

PySharpe's tactical research engine is a Python library under
`pysharpe.analysis`. Its strategy functions accept caller-supplied data and
return reproducible research results; they do not depend on Streamlit or
Northstar and do not fetch data or place orders.

## Modular direction

| Responsibility | Module boundary | Current state |
| --- | --- | --- |
| Market data and provenance | Callers supply named total-return, executable-price, and FX series with `MarketDataProvenance` records | Validation and quality reporting are implemented; no network provider is coupled to the strategy |
| Signal rules and simulation | `pysharpe.analysis.tactical_momentum` owns momentum, moving-average trend, prior-high breakout configuration, simulation, costs, and result serialization | Implemented for monthly, long-only rules with equal weights |
| Performance and risk | `pysharpe.metrics` supplies CAGR, volatility, Sharpe, and drawdown calculations | Reused by this strategy |
| Validation | `pysharpe.analysis.tactical_walk_forward` evaluates fixed configurations over chronological windows; `pysharpe.validation` owns purged resampling and trial accounting | Fixed-configuration monthly walk-forward evaluation is implemented; no parameter search or tuning is performed |
| Portfolio overlap | `pysharpe.analysis.portfolio_overlap` expands caller-supplied fund holdings into explicit constituents and reports common underlying weights | Implemented as a pure analysis of supplied weights; no holdings are loaded or modified |
| User interface and transport | Northstar or another client reads the versioned JSON result | Contract is documented below; no Northstar runtime, file access, or service is assumed |

Parameter searches remain separate validation work. The trend and breakout
rules below consume the same explicit market-data contract as momentum.

## Implemented strategy

`MomentumConfig` describes a ticker universe, lookback and skip months, number
of holdings, benchmark ticker, trading costs, FX freshness, and annualization
frequency. Each month, the engine ranks eligible assets by CAD-denominated
cumulative total return over the lookback window, excluding the most recent
month, then holds up to `top_n` assets at equal weights. Short positions are
not permitted. If fewer than `top_n` assets have a complete history, the
available selections share the portfolio equally. At the start of the
simulation the portfolio stays in cash until an executable signal is available;
on a later month with no eligible signal, existing holdings are retained.

A signal uses observations through the last available market session of its
month. The score window uses complete month-end observations and enforces the
configured price-staleness limit. A signal takes effect at the first supplied
price timestamp after its signal timestamp. The engine requires an exact
execution close and a sufficiently fresh FX quote on that date. It never moves
future prices or FX rates backward. Assets with incomplete initial listing
history become eligible only after the entire scoring window is available.

### Trend and breakout strategies

`TrendConfig` selects `rule="moving_average"` or `rule="breakout"`. Both rules
evaluate each asset monthly using CAD-converted total-return levels, then hold
up to `top_n` qualifying assets at equal weights. Ties resolve by ticker. They
are long-only; a month with no qualifying asset targets cash. A cash signal is
exported explicitly with ticker `CASH`, score `0`, and target weight `1`.

- **Moving average:** at month end, score is `P[t] / mean(P[t-L+1], ..., P[t])
  - 1`. The asset qualifies only when the score is positive. The current
  completed month is included in the trailing `L` observations.
- **Breakout:** a new position qualifies when `P[t]` exceeds the maximum of
  `P[t-L], ..., P[t-1]`. A held position remains eligible while `P[t]` stays
  above the minimum of those prior `L` levels; otherwise the rule exits to cash.
  The reported score is `P[t] / prior_high - 1`.

The signal timestamp is the final supplied market session in the month. Its
target takes effect at the first later supplied session, after that session's
return has been applied to the prior portfolio. The rules never use that
execution session's price or FX rate to form the signal. `max_price_staleness_days`
and the FX staleness/provenance checks apply as they do for momentum. Use
`run_trend_strategy(..., config=TrendConfig(...))`; it returns the common
`TacticalResearchResult` structure. `run_tactical_strategy()` dispatches a
`MomentumConfig` or `TrendConfig` directly. The old `MomentumResearchResult`
name remains an alias for compatibility with existing momentum clients.

```python
from pysharpe import TrendConfig, run_trend_strategy

config = TrendConfig(
    universe=("VFV.TO", "VCN.TO", "QQC.TO"),
    rule="breakout",
    lookback_months=12,
    top_n=2,
)
result = run_trend_strategy(
    total_return_prices,
    execution_prices,
    config=config,
    currencies={"VFV.TO": "CAD", "VCN.TO": "CAD", "QQC.TO": "CAD"},
)
```

### Input price and currency contract

- `total_return_prices` is a local-currency level series adjusted for splits and
  distributions. ETF histories are declared `net_of_expenses`; the strategy
  does not deduct MER again.
- `execution_prices` is a split-adjusted close series with distributions
  excluded. Both input frames must have identical, unique, ascending
  `DatetimeIndex` values. The frame is used to confirm that orders had a
  contemporaneous executable close; simulation returns come from the
  total-return series.
- `currencies` maps ticker symbols to ISO currency codes. CAD is the default.
  `fx_to_cad` contains CAD per unit of each foreign currency (for example,
  `1.35` CAD for one USD). Alignment is backward-only with a maximum age from
  configuration. Leading uncovered observations remain missing, and an
  interior FX coverage gap in a converted series is rejected.
- Source panels may contain leading missing values for assets that listed
  later. Interior missing price observations are rejected to prevent a stale
  quote from being mistaken for a current mark.

`provenance` optionally maps series keys such as `total_return_prices:VFV.TO`,
`execution_prices:VFV.TO`, and `fx_to_cad:USD` to `MarketDataProvenance`
records. Each record declares `source`, `retrieved_at`, `currency`,
`price_basis`, and `max_staleness_days`. The engine checks the declared basis
and currency against its input contract and rejects data that was already
stale when retrieved. Omitted records are reported as unattributed; PySharpe
does not invent a vendor or retrieval timestamp. FX series use currency
`CAD/USD` and price basis `CAD_per_currency_unit`.

The caller remains responsible for supplying trustworthy, correctly adjusted
histories and their provenance. The module does not invent missing history or
infer whether a vendor column is adjusted.

### Costs and benchmarks

For each rebalance, gross traded notional is the portfolio value multiplied by
the sum of absolute changes between target weights and the portfolio weights
after applying returns through the execution date. Thus the transaction record
captures drift trades even when the target allocation is unchanged. The cost is:

```text
gross_traded_notional * (slippage_bps + spread_bps / 2) / 10_000
+ foreign_currency_traded_notional * fx_fee_bps / 10_000
+ order_count * fee_per_order
```

Spread is the full quoted spread, so the model charges half the spread per
traded side. Commissions use a fixed-per-order model and default to `$0`;
spread, slippage, and FX cost remain independent assumptions. Inputs are
reported as supplied, not presented as a broker quote. VFV (configured benchmark ticker) and a
zero-return cash baseline use the same initial capital and reported period.
VFV must have a CAD-convertible observation at the simulation start; otherwise
the run fails instead of reporting benchmark metrics over a shorter interval.
The cash baseline assumes no yield and is not a cash-rate forecast.

## JSON contract for Northstar

`TacticalResearchResult.to_dict()` and `.to_json()` expose a read-only,
versioned result. Momentum results retain schema `1.1`; trend and breakout
results use schema `1.2` to identify the additional strategy rules:

```json
{
  "schema_version": "1.2",
  "generated_at": "2024-01-31T00:00:00",
  "strategy": {"name": "breakout", "config": {}},
  "signals": [],
  "weights": [],
  "backtest": [],
  "benchmarks": [],
  "trades": [],
  "risk": {},
  "benchmark_risk": {},
  "data_quality": {},
  "data_provenance": {},
  "cost_assumptions": {}
}
```

`generated_at` is the final input observation timestamp, making repeated runs on
the same inputs deterministic. Each signal row contains `signal_date`,
`effective_date`, `ticker`, `score`, and `target_weight`. Backtest rows contain
portfolio value and net return. Trade rows expose date, gross traded notional,
turnover, order count, transaction cost, and foreign-currency notional. Risk
metrics contain CAGR, annualized volatility, Sharpe ratio, and maximum drawdown.
Undefined statistics serialize as JSON `null`, never `NaN` or `Infinity`.

`data_quality` identifies the input return and execution-price bases, FX
alignment rule and coverage gaps, incomplete listing history, signal warm-up,
signal count, data end timestamp, provenance completeness, and stale-series
status. `signal_freshness_verified` is false when any required input has no
provenance record or has no observations. `data_provenance` records each
declared source alongside its latest observation and age at retrieval. A
declared empty series has `latest_observation: null` and appears in
`unverified_series`; it is not treated
as fresh merely because staleness cannot be calculated. Missing provenance
records appear in both `unattributed_series` and `unverified_series`.
`cost_assumptions` reports commissions,
spread, slippage, and FX parameters. Clients should branch on
`schema_version`, treat the payload as research output, and display its
timestamps and quality fields with the results.

### Fixed-configuration walk-forward evaluation

`run_walk_forward_momentum()` and `run_walk_forward_strategy()` use
`WalkForwardConfig(training_months, test_months)` to report non-overlapping
chronological test windows. The first training period must exceed the rule's
lookback (and momentum's skip window); later periods expand through the history
available before each fold. Signals use only prior observations, and each test
window exports its equity path, returns, strategy metrics, and same-period
VFV/cash metrics. Strategy settings remain fixed: there is no parameter search,
fit step, or claim that chosen parameters generalize beyond the evaluated
periods.

### Portfolio overlap analysis

`analyze_portfolio_overlap(portfolios, fund_constituents=..., fund_tickers=...)`
is a pure function over caller-supplied weights. A holding with explicit
constituents is expanded as `portfolio_weight × constituent_weight`. A holding
listed in `fund_tickers` without constituent data contributes entirely to
`unknown_weight`; a non-fund holding without look-through is treated as a
directly held security. If constituent weights sum below 1, the residual stays
unknown. Neither holdings nor constituent rows are renormalized or persisted.

For two portfolios, common exposure to security `i` is
`min(weight_a[i], weight_b[i])`. `shared_weight` sums that amount across shared
securities; `overlap_of_smaller_known_portfolio` divides by the smaller known
exposure total. The output includes per-portfolio known exposure, unknown
weight, unallocated portfolio weight, and coverage. Its independent JSON
contract is version `1.0`. Constituent inputs are flat; nested fund look-through,
classification, and issuer-data retrieval are out of scope.

Each out-of-sample fold starts at the strategy's actual simulated NAV on its
training boundary. This preserves fixed-dollar order-fee effects accumulated
before that boundary; benchmarks are scaled to the same starting NAV for a
comparable fold path.

Feasible Northstar integrations include reading a user-exported JSON artifact,
receiving the payload from an authenticated API that runs PySharpe elsewhere,
or consuming a versioned object-storage artifact. Those choices require a
separately designed transport and trust boundary. This repository implements
only JSON serialization; Northstar cannot be assumed to execute Python or read
PySharpe's local files.

## Validation scope and limitations

The deterministic synthetic fixtures cover future-data leakage, incomplete
listing windows, interior missing observations, backward-only FX and uncovered
history, transaction costs and zero-commission defaults, split-adjusted
execution-price separation, benchmark alignment, provenance basis/freshness,
chronological walk-forward folds for momentum and trend rules, trend exits, and
unknown look-through coverage. No real market observations are mixed into test
fixtures.

This slice does not fetch or independently verify live vendor data, provide
dividend or corporate-action reconstruction, estimate market impact, model
interest on cash, calculate tax, infer missing fund constituents, or search
parameters. Walk-forward folds evaluate a fixed rule and do not correct for
strategy selection outside the engine. Slippage, spread, FX fees, and cash
returns remain explicit user assumptions.
