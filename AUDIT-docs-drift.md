# Documentation drift audit — PySharpe

**Date:** 2026-09-23 · **Scope:** everything in the repo that describes the project —
`docs/`, root markdown (`README.md`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`,
`CHANGELOG.md`, `ROADMAP.md`), and `.agents/skills/**` · **Method:** every claim
was checked against code or a command; evidence is cited per finding and the raw
commands are in the appendix.

Verdict: the **code** documentation that matters most (`AGENTS.md`,
`CLAUDE.md`, `docs/GOTCHAS.md`, `docs/TEST_MAP.md`) is largely accurate and
unusually disciplined. The drift concentrates in three places: the
`.agents/skills/` packages (agent-facing), the published API reference, and the
guardrail *guards themselves* — three of which can no longer pass, so they have
quietly stopped protecting anything.

Severity scale: **S1** = wrong numbers or broken documented commands · **S2** =
documented behaviour that does not exist · **S3** = cosmetic / stale / unverifiable.

---

## A. Blocking findings

### A1 — `pysharpe-metrics` documents the risk-free-rate convention backwards (S1)

> **Fixed 2026-09-23.** The skill's Sharpe, Sortino and Constraints sections now
> state the annual convention with the real signatures (`*` keyword-only), and
> `tests/test_metrics.py::test_sortino_ratio_uses_annual_risk_free_rate` pins the
> convention so a future inversion fails a test rather than a doc.

| | |
| --- | --- |
| Claim | `pysharpe-metrics/SKILL.md:35` — "`risk_free_rate` must be passed as a **daily** decimal (e.g., 0.05/252 for 5% annual), never annual." Repeated at `:75` — "Risk-free rate is a daily decimal, never annualized inside the function." |
| Reality | `metrics.sharpe_ratio` (line 227) and `metrics.sortino_ratio` (line 329) take an **annual** rate: docstring "Annual risk-free rate expressed as a decimal", and the body computes `excess = annualised_return - risk_free_rate`. `sortino_ratio` converts it internally (`daily_rf = risk_free_rate / periods_per_year`). The dashboard and benchmarks both pass `0.02` annual (`app/analytics.py:25`, `analysis/benchmarks.py:129`). |
| Impact | An agent or contributor following the skill passes `0.02/252 ≈ 0.000079`; the Sharpe ratio is then computed against essentially a zero risk-free rate — silently inflated, never an error. This is the highest-risk doc defect in the repo because it changes results, not just prose. |
| Fix | Invert the skill text; add one doctest per metric asserting the annual convention. |

### A2 — Three skills document test commands that cannot run (S1)

> **Fixed 2026-09-23.** All seven references remapped to the suites that actually
> cover those modules (`docs/TEST_MAP.md` is authoritative), and
> `tests/test_skill_docs.py` now fails when a skill names a test file that does
> not exist. All five skill `Run:` commands execute green (161 / 38 / 24 / 75 / 40
> tests passing).

Seven referenced test files do not exist:

| Documented in | Phantom file | Proof |
| --- | --- | --- |
| `pysharpe-allocation/SKILL.md:110,113` | `tests/test_rebalance.py` | `pytest tests/test_rebalance.py` → `ERROR: file or directory not found` |
| `pysharpe-backtesting/SKILL.md:110,112,118` | `tests/test_analysis_walk_forward.py`, `test_analysis_benchmarks.py`, `test_analysis_visualization.py` | same |
| `pysharpe-data-pipeline/SKILL.md:95` | `tests/test_collation_proxy.py` | merged into `test_collation.py` (`docs/TEST_MAP.md:75`) |
| `pysharpe-optimization/SKILL.md:86,90` | `tests/test_optimization_models.py`, `tests/test_constraints_verification.py` | same |

The `Run:` blocks in three of the five skills therefore fail immediately — the
"load only the skill a task touches" design in the README degrades to a dead end
on first use. Note `docs/TEST_MAP.md` is **not** affected (see C1).

### A3 — The no-`.bfill()` rule is not enforced, and three guards can no longer pass (S1)

The repo's flagship correctness claim is "no lookahead bias". The checklist item
is explicit (`pysharpe-guardrails/references/precommit-checklist.md:21`): *"`grep -rn
'bfill()' src/pysharpe/ --include='*.py'` must return zero results."*

Current result: **5 matches**, all on price frames.

| Site | Object | Note |
| --- | --- | --- |
| `analysis/categorization.py:109` | `prices` frame → `ffill().bfill()` | feeds correlation grouping used before optimisation |
| `analysis/categorization.py:139` | normalised price frame → `ffill().bfill()` | same |
| `app/data.py:79` | cleaned price frame → `ffill().bfill()` | dashboard inputs |
| `app/data.py:289` | combined price frame → `ffill().bfill()` | same |
| `app/data.py:374` | combined price frame → `ffill().bfill()` | same |

`docs/GOTCHAS.md` narrowed the guard to `src/pysharpe/data/` in the 2026 FX
entry, so the data layer is clean (PASS) — but that narrowing is exactly what let
these five survive while the checklist still demands zero. Either the rule is
data-layer-only (then the checklist, `AGENTS.md`, `CLAUDE.md` and the README
pillar must say so) or these five are violations (then they are lookahead in the
UI path, which is also `pyright`-ignored — see A4). Decision required, not just
edits.

Three documented guards are now permanently red, which means nobody runs them:

| Guard | Documented at | Current result |
| --- | --- | --- |
| `grep -rn 'test_end_idx >= n' validation/resampling.py` "must be empty" | `docs/GOTCHAS.md` (PurgedKFold entry) | **matches** `resampling.py:879` — now a deliberate defensive `raise` ("Unreachable by construction… defensive"). The guard is stale, not the code. |
| `grep -rn '_fallback_result\|if not result.success'` | `docs/GOTCHAS.md` (silent-fallback entry) | **matches** `sharpe_optimizer.py:453` — now `raise RuntimeError(result.message)`, i.e. compliant code. The guard would flag a correct fix. |
| `grep -rn 'tax.loss\|TLH\|loss.harvest' src/pysharpe/` "must return zero" | `pysharpe-allocation/SKILL.md:101` | **matches** `execution/tax_tracker.py:6` — a docstring stating TLH is prohibited. The guard can never pass. |

### A4 — The type gate does not cover the UI, and the badge overstates the mode (S2)

| | |
| --- | --- |
| Claim | README badge + `docs/index.md:12` — "pyright **strict** + warnings-fatal"; `pyrightconfig.json` is the stated evidence. |
| Reality | `pyrightconfig.json:4` — `"typeCheckingMode": "standard"`; `:16` — `"ignore": ["src/pysharpe/app/**"]`; `include: ["src"]` while the UI entry point is root `app.py`. `CLAUDE.md:56` states it correctly ("standard mode with warnings promoted to fatal"), so README/index are the outliers. |
| Impact | ~2,000 lines of the primary user surface (`app.py` 737 lines + `src/pysharpe/app/*` ~1,000) are outside the type gate; combined with A3's five `bfill()` sites landing in `app/data.py`, the least-verified code is the code users touch first. |

The badges themselves also overstate: the three README badge URLs
(`badge.svg?job=quality|docs|test`) all render an identical whole-workflow badge —
GitHub ignores the unknown `job=` parameter (verified: same `<title>PySharpe CI -
passing</title>` for all four URLs including the bare one).

### A5 — The shipped `portfolio_config.json` breaks optimisation for any unmapped ticker (S1, runtime-reproduced)

Follows from issue #22 but is worse than reported there, and now proven:

```python
from pysharpe.portfolio_optimization import optimise_from_prices
cfg = json.load(open("portfolio_config.json"))     # auto-loaded by `optimise` from CWD
optimise_from_prices(name="repro", prices=prices,  # VFV.TO, VCN.TO, ZAG.TO
                     mer_mapping=cfg["mer_mapping"], geo_mapping=cfg["geo_mapping"],
                     max_portfolio_mer=cfg["constraints"]["max_portfolio_mer"], ...)
# -> ConfigurationError: Configuration missing for portfolio 'repro'.
#    MER data missing for: ZAG.TO; Geographic data missing for: ZAG.TO
```

- `cli.py:139-141` auto-loads `portfolio_config.json` from the CWD when `--config`
  is absent, then `cli.py:148-154` forwards `mer_mapping`, `geo_mapping` and the
  `constraints` block; `portfolio_optimization.py:340-352` **raises** if any ticker
  is missing from either map. Shipped maps cover 14 (MER) and 16 (geo) tickers.
- So running `pysharpe optimise` from the repo root on your own portfolio dies with
  an error that never mentions the shipped file. The same portfolio optimises fine
  without it (second half of the repro).
- `constraints.max_portfolio_mer` is `1.0`: as a cap that is 100%, i.e. inert
  (`w·MER − 1.0·Σw ≤ 0` is vacuous) — and it reads like a 100% MER to any human.
- Two loaders read the same file with different key sets: `load_execution_config`
  (`config.py:80-114`) reads only `account_type` / `allow_fractional` / `fx_fee_bps`
  for `rebalance`/`allocate`, while `optimise` parses `mer_mapping` / `geo_mapping` /
  `constraints` inline. Same file, two contracts, neither documented.

---

## B. Smaller drift (S2/S3)

| Doc | Line | Claims | Reality |
| --- | --- | --- | --- |
| `pysharpe-data-pipeline/SKILL.md` | 66-68 | `proxy_map.json` keys are `ticker`, `fx`, `weight` | Real schema is `proxy`, `fx_adjust`, `start_date`, `is_us_domiciled`, `is_cad_denominated` (`collation.py:175`; `config.py:225-239`) |
| `pysharpe-data-pipeline/SKILL.md` | 37-38 | `YFinancePriceFetcher.fetch(ticker, start, end, base_currency)` → columns `Date, Close, Currency` | Actual API: `fetch_history(ticker, *, period, interval, start, end)` (`fetcher.py:385`); FX is `apply_fx_conversion`, separate |
| `pysharpe-optimization/SKILL.md` | 43 | Geo constraints loaded from `portfolio_config.json` under `geo_constraints` | No such key exists anywhere in `src/`; the real key is `constraints.geo_lower_bounds` / `geo_upper_bounds`, read in `cli.py:151-154` |
| `pysharpe-optimization/SKILL.md` | 38 | `mu_adjusted = mu - mer/252` | No per-day MER deduction exists; MER enters as a **constraint** on weighted MER (`sharpe_optimizer.py:398-422`; `portfolio_optimization.py:457-463`) and as reporting (`:495-497`) |
| `pysharpe-backtesting/SKILL.md` | 40 | Backtests "Track: portfolio value over time, **cash flows**, turnover, drawdowns" | No cash-flow support in `analysis/backtest_engine.py` (grep "withdraw"/"contribution" → nothing). This is P5.3 on the roadmap, stated as existing behaviour |
| `pysharpe-backtesting/SKILL.md` | 77-80 | `benchmarks.py` builds equal-weight, market-cap-proxy and 60/40 benchmarks | It holds `CANADIAN_BENCHMARKS` (VEQT/XEQT/VGRO/XGRO/VBAL/XBAL) + `fetch_benchmark_metrics` + tax characteristics |
| `pysharpe-metrics/SKILL.md` | 64-68 | Presents `numpy.cov(returns.T)` / `numpy.corrcoef(...)` as the covariance/correlation helpers | `metrics.py` provides no covariance or correlation function at all; those are numpy recipes, not the module's API |
| `pysharpe-allocation/SKILL.md` | 25 | `execution/tax_tracker.py` — "TFSA tax tracking" | It implements CRA ACB tracking (weighted-average cost, commissions, return-of-capital) |
| `pysharpe-development/references/architecture.md` | 66-83 | Module inventory | Omits `execution/tax_tracker.py`, `cash_flow_rebalance.py`, `brokerage.py`, `data/linkage.py`, `data/portfolio.py`, `data/workflows.py`, `validation/*`, `guardrails/*`, `exceptions.py` |
| `pysharpe-development/references/test-map.md` | whole file | Duplicate of `docs/TEST_MAP.md` | Two sources of truth, already divergent (89 vs 206 lines, different organisation) for the same mapping |
| `docs/api.md` | whole file | "API Reference" | Missing 14 modules: `cli`, `workflows`, `portfolio_optimization`, `data_collector`, `logging_utils`, all six `app/*`, `analysis.visualization`, `data.workflows`, `visualization.utils` — i.e. the published API omits the CLI and top-level orchestrators |
| `docs/architecture.md` | 1-6 | "guiding principles and planned components … before implementation work begins" | Pre-implementation framing in a doc whose `:34` then claims v1.0 shipped; contradicts `CLAUDE.md`, the authoritative reference |
| `docs/benchmarks.md` | 4 | Baseline is "commit `v0.1.0`" | No tags exist in the repo (`git tag -l` → 0), so the baseline is unidentifiable |
| `docs/articles/pysharpe_showcase.mdx` | 1-19 | Astro article for a docs site | Not renderable by MkDocs (`.mdx` is not a page type) and absent from `mkdocs.yml` nav → orphaned; the `AssetLocationMatrix.astro` / `FwtDragBars.astro` components it references do not exist in this repo |
| `docs/proxies.md` | 11-12 | Crypto as "Tactical (Alt)" | Contradicts `AGENTS.md`'s CAD-ETF-only universe (tracked as issue #25) |
| `mkdocs.yml` | 9-10 | `exclude_docs: superpowers/**` | `docs/superpowers/` is untracked (local-only plan files), so the exclusion is never exercised in CI |
| `pysharpe-guardrails/references/precommit-checklist.md` | 75 | Item 7 guard: no `yfinance`/`requests.get`/`urllib` in tests | 8 matches: docstring mentions plus legitimate monkeypatch imports in `test_fx_adjustment.py`. The guard's own parenthetical ("excluding test infrastructure for mocking") excuses them, but as written it cannot be automated |
| `README.md` | 258-260 | "`portfolio_config.json` … auto-loaded for MER/geo constraints" | True for `optimise` only; `rebalance`/`allocate` read the same file through a different loader that ignores those keys (see A5) |

Already tracked elsewhere, so not repeated as new issues: `CONTRIBUTING.md` (`black`, manual venv, "no automated releases yet") → #30; `AGENTS.md` §5 AgentRQ workflow with a deleted account → #30; README "990+ tests" (actual 997) → #30; crypto scope → #25; config landmine → #22.

---

## C. Verified accurate (no action)

| Check | Result |
| --- | --- |
| `docs/TEST_MAP.md` coverage | All 39 `tests/test_*.py` files appear — nothing omitted |
| README/CLI flag examples | Every documented flag exists in the parser (`--shrinkage-floor`, `--return-model`, `--base-currency`, `--max-weight`, `--holdings-json`, `--holdings-csv`, `--new-cash`, `--amount`, `--months/--initial/--monthly/--rate`, `--config`) |
| Artefact naming | `{name}_collated.csv`, `{name}_weights.txt` as documented (`cli.py:208,536`; `portfolio_optimization.py:88,162`) |
| DuckDB cache freshness | "fresh for 24 hours" matches `fetcher.py:182,256` |
| Benchmark MERs | Decimal fractions in `BENCHMARK_MERS`, as the convention requires |
| README CLI/benchmark lists | 5 subcommands, 6 Canadian benchmarks — both correct |
| Example portfolios | `cad_portfolio`, `canadian_etfs`, `demo`, `veqt_style` all exist as documented |
| `make` targets | Every target documented in `CLAUDE.md`/README exists in the Makefile; `repomix` is installed locally; `PYSHARPE_REQUIRE_PYRIGHT` is wired in `tests/conftest.py:72` |
| CI description | Three jobs (quality / test / docs) with SHA-pinned actions — as documented |
| Data-layer guardrails | All `docs/GOTCHAS.md` data-layer guards pass (`bfill` in `data/`, `use_container_width`, hard-coded `2015`, `except ValueError` shrink/cov, `LedoitWolf` in estimators, `/ 100` near MER, `groupby(axis=1)`) |

---

## D. Recommended sequence

1. **A1 + A2** (same file family): the metrics skill inverts a numerical convention
   and three skills ship unrunnable commands — both are one-pass fixes with
   outsized trust impact, and both are agent-facing.
2. **A3 decision**: scope the `.bfill()` rule and reconcile the three dead guards.
   This is a correctness claim, so it belongs with the P2 verification work.
3. **A4**: extend the type gate to `app.py` + `src/pysharpe/app/**`, then correct the
   badge/index "strict" wording.
4. **A5**: fold into #22 — the fix (stop auto-loading a personal config; ship an
   example) also removes the unmapped-ticker hard failure.
5. **B rows** as a single docs-cleanup pass, with `docs/api.md` and the duplicate
   test map handled by generating from source rather than hand-maintaining.
6. **C rows** are regression anchors: they show the sweep actually checked rather
   than skimming.

## Appendix — method

```bash
# gates and surfaces
make check ; make build_docs ; uv run pysharpe <cmd> --help
# claim-vs-code checks (subset)
for f in $(ls tests/test_*.py | xargs -n1 basename); do grep -q "$f" docs/TEST_MAP.md || echo "missing: $f"; done
for m in $(find src/pysharpe -name "*.py" ! -name "__init__.py" | sed 's|src/||;s|\.py$||;s|/|.|g'); do grep -q "::: $m" docs/api.md || echo "not in api.md: $m"; done
grep -rn 'bfill()' src/pysharpe/ --include='*.py'          # 5 hits (price frames)
grep -rn 'test_end_idx >= n' src/pysharpe/validation/resampling.py   # 1 hit (defensive)
grep -rn '_fallback_result\|if not result.success' src/pysharpe/     # 1 hit (raises now)
grep -rn 'tax.loss\|TLH\|loss.harvest' src/pysharpe/ --include='*.py' # 1 hit (docstring)
grep -rn "geo_constraints" src/                            # 0 hits
grep -rn "mer.*/ *252" src/pysharpe/optimization/sharpe_optimizer.py   # 0 hits
curl -s "<badge url>?job=quality" | grep -o '<title>[^<]*</title>'
uv run pytest tests/test_rebalance.py                      # ERROR: file not found
# runtime repro for A5: optimise_from_prices with the shipped maps + ZAG.TO
```
