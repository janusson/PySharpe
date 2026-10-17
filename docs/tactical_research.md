# Tactical research engine

PySharpe's tactical research engine is a Python library under
`pysharpe.analysis`. Its strategy functions accept caller-supplied data and
return reproducible research results; they do not depend on Streamlit or
Northstar and do not fetch data or place orders.

## Modular direction

| Responsibility | Module boundary | Current state |
| --- | --- | --- |
| Market data and provenance | `pysharpe.data` adapters supply named total-return, executable-price, FX, currency, and observation-time inputs | Existing fetchers and FX tools are reusable; no tactical data adapter is added |
| Signal rules and simulation | `pysharpe.analysis.tactical_momentum` owns strategy configuration, momentum ranks, portfolio simulation, trading costs, and result serialization | Implemented for monthly, long-only cross-asset momentum |
| Performance and risk | `pysharpe.metrics` supplies CAGR, volatility, Sharpe, and drawdown calculations | Reused by this strategy |
| Validation | `pysharpe.validation` owns walk-forward, purged resampling, and trial accounting | Existing tools are reusable; this slice does not claim walk-forward validation |
| Portfolio exposures | `pysharpe.analysis.portfolio_analytics` can report look-through exposure for saved holdings | Existing holdings analytics are separate from this strategy result; overlap analysis is deferred |
| User interface and transport | Northstar or another client reads the versioned JSON result | Contract is documented below; no Northstar runtime, file access, or service is assumed |

Trend-following, breakout rules, and walk-forward evaluation should be separate
strategy or validation components that consume the same explicit market-data
contract. They are not part of this first vertical slice.

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
traded side. Cost inputs default to zero and are reported as supplied; they
are not presented as a broker quote. VFV (configured benchmark ticker) and a
zero-return cash baseline use the same initial capital and reported period.
VFV must have a CAD-convertible observation at the simulation start; otherwise
the run fails instead of reporting benchmark metrics over a shorter interval.
The cash baseline assumes no yield and is not a cash-rate forecast.

## JSON contract for Northstar

`MomentumResearchResult.to_dict()` and `.to_json()` expose a read-only,
versioned result. Version `1.0` has this top-level shape:

```json
{
  "schema_version": "1.0",
  "generated_at": "2024-01-31T00:00:00",
  "strategy": {"name": "cross_asset_momentum", "config": {}},
  "signals": [],
  "weights": [],
  "backtest": [],
  "benchmarks": [],
  "trades": [],
  "risk": {},
  "benchmark_risk": {},
  "data_quality": {}
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
signal count, and data end timestamp. Clients should branch on
`schema_version`, treat the payload as research output, and display its
timestamps and quality fields with the results.

Feasible Northstar integrations include reading a user-exported JSON artifact,
receiving the payload from an authenticated API that runs PySharpe elsewhere,
or consuming a versioned object-storage artifact. Those choices require a
separately designed transport and trust boundary. This repository implements
only JSON serialization; Northstar cannot be assumed to execute Python or read
PySharpe's local files.

## Validation scope and limitations

The deterministic test fixtures cover future-data leakage, incomplete listing
windows, interior missing observations, backward-only FX and uncovered history,
transaction costs, split-adjusted execution-price separation, benchmark output,
and insufficient lookback history. They use synthetic values only.

This slice does not fetch or validate live vendor data, provide dividend or
corporate-action reconstruction, estimate market impact, model interest on
cash, calculate tax, measure portfolio look-through overlap, or run
walk-forward/out-of-sample folds. Slippage, spread, commission, FX fees, and
cash returns must therefore be interpreted as explicit user assumptions.
