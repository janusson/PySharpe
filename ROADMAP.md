# PySharpe Roadmap to v1.0

**v1.0 means:** an installable, released, documented, self-hosted tool whose
numbers are defensible. Concretely — a tagged `v1.0.0` release with built
artifacts, published docs, a working `pip`/`uv` install on a clean machine, a
first-run experience that needs no network, and a verification suite that pins
results against *external* published values rather than only self-consistent
synthetic data.

Hosted public product features (Monte Carlo, inflation/real returns, cash flows
inside backtests, factor analysis, accounts and saved models) are **1.1+**, listed
in P5 and deliberately excluded from the 1.0 gate.

Issue IDs below map 1:1 to GitHub issues on
[project #11 "PySharpe Core"](https://github.com/users/janusson/projects/11).
Phases are milestones. The 2026-09-23 documentation sweep (claim-by-claim drift,
evidence per finding) is in `AUDIT-docs-drift.md`; its blocking findings are
tracked as P0.2, P1.5, P1.8, P2.10, and P2.11.

## Where the project actually is (measured 2026-09-23)

| Signal | Result | Command |
| --- | --- | --- |
| Test suite | **997 passed, 3 skipped** (PyMC/compiler-gated), 25.5 s | `make check` |
| Lint / types | ruff clean; pyright **0 errors, 0 warnings** | `make check` |
| Coverage | **79.68%** against a 75% floor — but over `src/pysharpe` only | `make test` |
| Docs | `mkdocs build --strict` passes | `make build_docs` |
| CLI | 5 subcommands respond | `uv run pysharpe --help` |
| Dashboard | boots, `/_stcore/health` → 200 | `streamlit run app.py` |
| Git tags / releases | **none** | `git tag -l`, `gh release list` |
| PyPI artifact | **none**; name `pysharpe` **owned by another project** | `GET pypi.org/pypi/pysharpe/json` |
| `pysharpe.__version__` | **does not exist** | `grep -rn "__version__" src/` |
| Open issues / open PRs | 0 / 3 (Dependabot, since 2026-09-13) | `gh issue list`, `gh pr list` |

The engineering core is genuinely strong — hermetic 25-second suite, strict
pyright with warnings fatal, SHA-pinned CI actions, `docs/GOTCHAS.md` with
loud grep-guard discipline. What is missing is everything *around* the engine:
release truth, reproducible evidence, and the packaging that turns a repo into a
program other people can run.

## P0 — Security & version truth (immediate)

| ID | Title | Why now (evidence) | Done when |
| --- | --- | --- | --- |
| P0.1 | Purge the leaked AgentRQ MCP token from the public repo and rewrite history | `.mcp.json` was committed 2026-07-14 (`8666514`) and is publicly readable at HEAD: `raw.githubusercontent.com/janusson/PySharpe/main/.mcp.json` → HTTP 200, 173-char `token=…`. History scan finds exactly 1 blob containing it. Runbook: `PURGE-RUNBOOK-mcp-token.md` | Token rotated and old token rejected; history rewritten + force-pushed; old-SHA raw fetch 404s; `gitleaks detect --all` clean; `AGENTS.md` §5 reads credentials from env |
| P0.2 | Remove the shipped `portfolio_config.json` landmine (vacuous MER cap + cwd auto-load) | Three faults in one file. (1) `optimise` auto-loads `portfolio_config.json` from the CWD when `--config` is absent (`cli.py:139-141`) and forwards `mer_mapping`/`geo_mapping`/`constraints` (`cli.py:148-154`), so running from the repo root applies the maintainer's constraints to anyone's run. (2) `portfolio_optimization.py:340-352` **raises** for any ticker missing from either map, and the shipped maps cover only 14 (MER) / 16 (geo) tickers — runtime-reproduced with a 3-asset frame including ZAG.TO: `ConfigurationError: MER data missing for: ZAG.TO; Geographic data missing for: ZAG.TO`; the same portfolio optimises fine without the file. (3) `"max_portfolio_mer": 1.0` is a 100% cap, algebraically inert (`w·MER − 1.0·Σw ≤ 0`), so the MER cap documented in README does nothing — while reading like a 100% MER. Plus: `rebalance`/`allocate` read the same file through `load_execution_config` (`config.py:80-114`), which honours only `account_type`/`allow_fractional`/`fx_fee_bps` — two loaders, two contracts, neither documented | Repo ships `portfolio_config.example.json`; the auto-load path cannot hard-fail a user's own tickers; one documented loader contract (or an explicit statement of both); a realistic `max_portfolio_mer` or none; regression test for the unmapped-ticker case; `docs/GOTCHAS.md` entry + grep guard per the repo's bug-fix discipline |
| P0.3 | Single source of truth for the version; expose `pysharpe.__version__` | `pyproject.toml:7` says `1.0.0`, `CHANGELOG.md:3` says "v1.0.0 (2026-08-19)", `docs/architecture.md:34` says "As of v1.0.0 … shipped" — and nothing else agrees: no tag, no release, no runtime version attribute | `pysharpe.__version__` resolves from installed metadata (`importlib.metadata`); version declared once; `CHANGELOG.md` opens with an `## Unreleased` section; decision recorded (keep `1.0.0` as an internal label, or renumber to `0.9.0` until the release lands) |
| P0.4 | Resolve the PyPI distribution-name collision (`pysharpe` is taken) | `GET pypi.org/pypi/pysharpe/json` → 200, unrelated `PySharpe 0.1.0` (pysharpe-research, docs at pysharpe.org). Publishing under that name is impossible. Verified available: `pysharpe-ca`, `pysharpe-cad`, `pysharpe-canada`, `pysharpe-ca-core` (all 404) | Distribution name chosen and set in `pyproject.toml`; import name stays `pysharpe`; README/docs/badges updated; name-and-namespace collision note added |
| P0.5 | Decide and record crypto/gold scope against `AGENTS.md` | `AGENTS.md` restricts the universe to broad-market CAD-denominated index ETFs and prohibits single-stock models, while `proxy_map.json:79-92` maps `BTC`→`BTC-USD` and `FETH.TO`→`ETH-USD`, `docs/proxies.md:11-12` labels them "Tactical (Alt)", and `data/portfolio/current_state.csv` holds FBTC alongside ETFs | One recorded decision (ADR or `docs/scope.md`); `proxy_map.json`, `docs/proxies.md`, and sample portfolios agree with it; if in scope, the tax/asset-location treatment of crypto is stated |

## P1 — Release engineering (make "released" true)

| ID | Title | Why now (evidence) | Done when |
| --- | --- | --- | --- |
| P1.1 | Tag-driven release workflow (build, artifacts, GitHub Release) | `.github/workflows/` contains only `ci.yml`; `CONTRIBUTING.md:44` says "we are not publishing automated releases yet"; `dist/` holds untracked, stale (2026-08-26) artifacts; zero tags exist | Pushing `v*` builds sdist+wheel, creates a GitHub Release with notes pulled from `CHANGELOG.md` and the artifacts attached; workflow pinned by SHA like `ci.yml` |
| P1.2 | Publish the MkDocs site (`site_url` + Pages deploy) and fix the docs badge | `mkdocs.yml` has no `site_url`, no deploy workflow — the "Docs" badge points only at a CI job, so the built site is never reachable | `site_url` set; docs deploy on push to `main`; badge points at the live site; `--strict` still fatal |
| P1.3 | Publish to PyPI via Trusted Publishing under the new dist name | No published artifact exists; the big public-install story is broken by P0.4 | `uv pip install <dist-name>` works on a clean machine; release workflow publishes via OIDC (no long-lived PyPI token); README install section matches reality |
| P1.4 | Wheel-install smoke test in CI (fresh venv, no source tree) | All 997 tests import the package from the checkout; nothing proves the *built* wheel is importable or that the console script works once installed | CI job installs the built wheel into a clean venv, runs `pysharpe --help`, imports the package, and executes one offline synthetic end-to-end |
| P1.5 | Fix contributor-doc drift (CONTRIBUTING, AGENTS §5, architecture.md, README test count) | `CONTRIBUTING.md` prescribes `python -m venv` and `black --check` (black is not a dependency; ruff is authoritative per `CLAUDE.md:131`), and claims manual releases; `docs/architecture.md:1-6` is written pre-implementation; README says "990+ tests" (actual 997); `AGENTS.md` §5 mandates an AgentRQ blocking-question flow that depended on the leaked credential | Contributor docs match the Makefile/uv/pyright reality; test counts and module lists corrected or generated; the AgentRQ workflow references env-var credentials |
| P1.6 | Repo hygiene: branch cleanup, Dependabot triage, templates, SECURITY.md, CODE_OF_CONDUCT | 4 unmerged local branches plus a merged remote; 3 Dependabot PRs open since 2026-09-13; no in-repo `dependabot.yml`, no issue/PR templates, no `SECURITY.md`, no code of conduct | Stale branches deleted or closed with a reason; Dependabot PRs merged/closed with the lockfile regenerated; `dependabot.yml` with grouping; templates + `SECURITY.md` (with a private disclosure path) + CODE_OF_CONDUCT present |
| P1.7 | Add gitleaks secret scanning to CI and enable push protection | The leak in P0.1 was a tracked credential in a public repo; CodeQL is scheduled but secrets are outside its scope | `gitleaks` runs on push/PR and fails on findings; repo-level secret scanning + push protection enabled |
| P1.8 | Repair the `.agents/skills/` documentation defects (documented commands fail; conventions inverted) | The skill packages are the repo's advertised agent framework, and three of the five document runnable commands that error out: `tests/test_rebalance.py`, `test_analysis_walk_forward.py`, `test_analysis_benchmarks.py`, `test_analysis_visualization.py`, `test_collation_proxy.py`, `test_optimization_models.py`, `test_constraints_verification.py` do not exist (`pytest tests/test_rebalance.py` → `ERROR: file or directory not found`). Worse, `pysharpe-metrics` inverts a numerical convention: it says `risk_free_rate` "must be passed as a **daily** decimal", while `metrics.sharpe_ratio`/`sortino_ratio` take an **annual** rate (`metrics.py:227,329`, and the app passes `0.02`). Also wrong: `proxy_map.json` key names, `YFinancePriceFetcher.fetch` signature, "geo constraints under `geo_constraints`", `mu_adjusted = mu - mer/252`, backtests "track cash flows", and `benchmarks.py` contents. Full detail in `AUDIT-docs-drift.md` §A1, §A2, §B | Every `Run:` block in `.agents/skills/**` executes successfully; the risk-free-rate convention matches the code in all five places it is described; proxy-map schema, fetcher API, MER handling and benchmark contents match implementation; a CI check (or a test) asserts that every test path named in a skill exists |

## P2 — Verification hardening (make the numbers defensible)

| ID | Title | Why now (evidence) | Done when |
| --- | --- | --- | --- |
| P2.1 | Coverage gate must measure `app.py`; raise the floor from 75% | `--cov=src/pysharpe` (Makefile:12) excludes the 307-statement root `app.py` that Streamlit actually executes — measured directly it is 89%; regressions there are invisible to CI | Coverage surface includes `app.py`; floor raised with a documented ratchet plan; CI fails on a simulated `app.py` regression |
| P2.2 | Per-module coverage floors for the high-stakes modules | Current worst: `visualization/equity_curve.py` 8%, `visualization/correlation.py` 25%, `app/rebalance_ui.py` 31%, `workflows.py` 35%, `app/backtest.py` 45%, `execution/rebalance.py` 63% (emits the actual buy plan), `config.py` 61%, `cli.py` 68%, `portfolio_optimization.py` 72% | Floors declared per module in one place; the order-emission path (`execution/rebalance.py`, `execution/brokerage.py`) and tax guardrails are at ≥ 90% |
| P2.3 | Golden regression suite anchored to issuer-published ETF results | `tests/golden/` holds two small fixture CSVs; nothing ties computed CAGR/vol/MDD to an external authority, so a systematic bias in returns, FX, or dividend handling would pass CI | Golden cases for VEQT/XEQT/VGRO/VBAL (and VFV↔VOO FX-adjusted parity) pinned to issuer-published values with stated tolerances and a documented refresh procedure |
| P2.4 | Golden fixtures for CRA ACB and superficial-loss worked examples | `guardrails/tax_compliance.py` (97% covered) and `execution/tax_tracker.py` encode high-stakes rules; coverage proves the code runs, not that the rules are right | Worked CRA examples (ACB with commissions and return-of-capital; identical-property ±30-day interactions across TFSA/NON_REG) reproduced exactly, with the source cited in the test |
| P2.5 | Checked-in price snapshots and deterministic end-to-end metric assertions | The suite is intentionally synthetic-only, so no test asserts a *specific* number end to end; `data/price_hist/*` are untracked working artefacts | A versioned snapshot of real adjusted closes drives exact-value assertions on returns/metrics/optimizer output; identical inputs produce identical outputs across runs and machines |
| P2.6 | Data-integrity verification and a `pysharpe doctor` command | The whole product rests on one uncontrolled input: the yfinance scrape behind `DuckDBCache` (`data/fetcher.py`, `auto_adjust=True`). No test cross-checks adjusted closes or distributions; the UI never states the data vintage | `pysharpe doctor` reports data freshness, ticker coverage, adjustment sanity, and vendor reachability; checks cross-validate against a second reference; "data as of" is shown in the UI and in every export |
| P2.7 | Surface estimation uncertainty in comparison output | `validation/sample_size.py`, `friction.py`, and `resampling.py` exist and are tested, but the comparison table (`app/analytics.py`) presents point estimates with no interval or significance statement — the opposite of the evidence-tier promise | Confidence intervals (or standard errors) shown for expected return / Sharpe per row; cells below the sample-size threshold are flagged; caption states the estimator and the assumption set |
| P2.8 | Typed error UX in dashboard and CLI (no tracebacks) | `PySharpeError`/`DataValidationError`/`DataIngestionError` contracts exist and are documented, but users still meet raw stack traces (and Streamlit's own); a CLI run also leaks an `arviz.MigrationWarning` banner on startup | User-facing surfaces render actionable messages with a next step, never a traceback; a test drives each error class through the UI/CLI paths |
| P2.9 | Offline end-to-end acceptance test of the CLI pipeline | Every test uses synthetic data by design, so nothing has ever exercised `optimise → rebalance → allocate` as a user would, including artefact round-trips | A cassette/fixture server stands in for the network and the full CLI pipeline runs offline in CI with assertions on emitted artefacts |
| P2.10 | Reconcile the no-`.bfill()` rule: five price-path sites and three dead guards | The pre-commit checklist demands `grep -rn 'bfill()' src/pysharpe/ --include='*.py'` return **zero**; it returns five — `analysis/categorization.py:109,139` and `app/data.py:79,289,374`, all on price frames. Meanwhile three documented guards can no longer pass, so nobody runs them: the PurgedKFold guard still matches `validation/resampling.py:879` (now a deliberate defensive `raise`), the silent-fallback guard still matches `optimization/sharpe_optimizer.py:453` (now `raise RuntimeError` — compliant code), and the allocation skill's TLH guard matches `execution/tax_tracker.py:6`'s prohibition docstring. Detail: `AUDIT-docs-drift.md` §A3 | One recorded decision on the rule's scope; checklist, `AGENTS.md`, README pillar and `GOTCHAS.md` agree with the code; the five sites fixed or explicitly sanctioned with a reason; every documented guard either green in CI or rewritten so it can pass; UI price handling no longer backfills leading gaps |
| P2.11 | Extend the type gate to the UI and correct the "pyright strict" claim | `pyrightconfig.json` sets `typeCheckingMode: "standard"`, `ignore: ["src/pysharpe/app/**"]`, `include: ["src"]` — so the six `app/*` modules plus root `app.py` (737 lines) are outside the gate, while the README and `docs/index.md` badges claim "pyright **strict** + warnings-fatal" (`CLAUDE.md` states it correctly). Those same files hold four of the five `.bfill()` sites. Detail: `AUDIT-docs-drift.md` §A4 | `app.py` and `src/pysharpe/app/**` are type-checked (or the exclusion carries a written reason); README/index wording matches the configured mode; badge URLs stop claiming per-job status that GitHub ignores |

## P3 — v1.0 product surface (small, self-hosted)

| ID | Title | Why now (evidence) | Done when |
| --- | --- | --- | --- |
| P3.1 | Zero-network first run: bundled demo dataset and `make demo` | The dashboard's first action is a live download; with no cache and no network the first-run experience is an error, and the docs' promise of a quickstart assumes Yahoo is reachable | `make demo` (or `pysharpe demo`) seeds the DuckDB cache from bundled snapshots and opens a working dashboard with no network; documented in the README quickstart |
| P3.2 | Local portfolio store with import/export | Portfolios are hand-edited files under `data/portfolio/` (`current_state.csv` mixes holdings, target weights, and valuation inputs in one CSV), and nothing survives a reinstall or moves between machines | Local store (SQLite or JSON) for portfolios/holdings/runs; import/export round-trips the existing CSV formats; README documents migration |
| P3.3 | One-click HTML/PDF analysis report | `docs/flowchart.md:175` lists PDF reports as planned; today output is CSV/JSON plus screen charts, so nothing is shareable as one artefact | One action exports a self-contained report (metrics, comparison, frontier, backtest, target weights, data vintage, assumption list) that renders correctly offline |
| P3.4 | User-facing docs: getting started + methodology | `docs/` is developer-facing (test map, gotchas, API); the evidence canon, the VA-vs-MPT choice, tax assumptions, and metric definitions live in README prose and `AGENTS.md` rules | A getting-started walkthrough and a methodology page define every reported metric, the estimator behind it, and its assumptions; both in the mkdocs nav |
| P3.5 | Educational-use disclaimers across README, UI, and exports | The tool gives allocation and order-level advice and has no disclaimer anywhere; `LICENSE` is MIT with no warranty framing for financial use | "Educational and research use; not financial advice" stated in README, dashboard footer, CLI `--help` epilogue, and every export; reviewed for wording consistency |
| P3.6 | Packaging diet: move heavyweight dependencies out of core | `pyproject.toml:27-43` puts `pymc`, `streamlit`, `matplotlib`, `seaborn`, `statsmodels`, `arch`, and `cvxpy` in the core `dependencies`, so a library user installs the whole app stack | Core install is small (numpy/pandas/scipy + data layer); heavy engines and UI behind extras that stay import-lazy; documented footprint and the `[all]` convenience path preserved |
| P3.7 | Self-host deploy recipe (Dockerfile + compose + auth notes) | The dashboard is `streamlit run app.py` on a laptop; there is no container, no auth story, and no note that the server binds 0.0.0.0 by default | `Dockerfile` + compose recipe builds from the lockfile, persists cache/exports as volumes, exposes a healthcheck, and documents auth/reverse-proxy and single-user expectations |
| P3.8 | Docs-surface completeness: API reference gaps, orphaned article, unverifiable baseline | `docs/api.md` omits 14 modules including `cli`, `workflows`, `portfolio_optimization`, `data_collector` and all six `app/*` modules — the published API reference has no CLI and no orchestration; `docs/articles/pysharpe_showcase.mdx` is not a MkDocs page type and is absent from the nav, so a 254-line article that references Astro components absent from this repo is unreachable; `docs/benchmarks.md:4` cites a `v0.1.0` baseline that no tag identifies; `.agents/skills/pysharpe-development/references/test-map.md` duplicates `docs/TEST_MAP.md` (89 vs 206 lines) as a second source of truth. Detail: `AUDIT-docs-drift.md` §B | `docs/api.md` covers every public module (generated from source, not hand-listed); the showcase article is published properly or moved out of `docs/`; the benchmark baseline names a commit that exists in the repo; one test-map source of truth with the other referencing it |

## P4 — GA gate & launch

| ID | Title | Why now (evidence) | Done when |
| --- | --- | --- | --- |
| P4.1 | Freeze v1.0 scope and write the release notes | The changelog currently mixes an unreleased "v1.0.0" label with a superseded v0.3.0 development log, so no reader can tell what shipped when | Scope freeze recorded (everything in P0–P3 either closed or explicitly deferred with a milestone); `CHANGELOG.md` has a real `v1.0.0` entry written for users |
| P4.2 | External beta with three users and blocker triage | Every test is synthetic and every run to date is the author's; nothing has met a real user's portfolio, broker CSV, or patience | Three users complete install → demo → their own portfolio unaided; findings triaged into 1.0 blockers vs 1.1+; blockers closed or documented as known limitations |
| P4.3 | Clean-machine reproducibility rehearsal from the published wheel | The install path has never been exercised outside this checkout (P1.4 covers CI; this covers reality: fresh OS user, no cache, no uv cache, no network at section boundaries) | Install, `pysharpe doctor`, `make demo`, own-portfolio run, and report export all succeed on a clean machine following only the published docs, with timings recorded |
| P4.4 | Cut `v1.0.0`: tag, release, and docs nav | Nothing external marks a 1.0; the roadmap itself is not yet part of the docs site | `v1.0.0` tagged and released with artifacts; docs site live and linking the roadmap; README's status/badges match the release |
| P4.5 | Post-release operating checklist | There is no stated triage cadence, security contact, or data-handling/backup note for a tool holding a user's real holdings | Written checklist covering issue triage cadence, security disclosure path, data-retention/backup and export guarantees, and the dependency-update rhythm |

## P5 — Post-1.0 (1.1+, explicitly outside the 1.0 gate)

These close the gap to `portfoliovisualizer.com`. They are the reason the staged
approach works: 1.0 ships the engine that is already strongest, then the product
surface grows without holding the release hostage.

| ID | Title | Scope note |
| --- | --- | --- |
| P5.1 | Monte Carlo simulation engine | Bootstrap (`validation/resampling.py` regime bootstrap is the seed) plus parametric draws; outcome distributions and funded-ratio framing |
| P5.2 | Inflation and real-return modeling | Real vs nominal reporting throughout: metrics, frontier, backtests, DCA, reports; CPI data source with the same FX-grade guardrails |
| P5.3 | Cash flows inside backtests | Scheduled/per-period contributions and withdrawals, schedule import, cash-flow-aware rebalancing |
| P5.4 | Withdrawal and decumulation modeling | Safe-withdrawal-rate sweeps, VPW, sequence-of-returns stress, TFSA/RRSP/RRIF withdrawal sequencing |
| P5.5 | Fama-French factor exposure analysis | Regression-based factor loadings for a portfolio or fund; directly serves the factor-tilt portfolios in the project's research notes |
| P5.6 | Compare N portfolios and saved-model benchmarks | Saved models usable as benchmarks across tools, per the PV "save as benchmark / custom series" pattern |
| P5.7 | Hosted public instance (auth, accounts, persistence) | Deployment, accounts, saved models, per-user data isolation, quotas, cost and abuse controls |
| P5.8 | Tactical/valuation-driven model library | Moving-average, momentum, volatility-targeting, valuation-tilt models on top of the existing valuation scoring |

## Explicitly out of scope (unchanged)

`AGENTS.md` prohibitions stand: no predictive price ML, no day-trading logic, no
gamified UI, no single-stock idiosyncratic models, no options pricing, no
tax-loss harvesting (TFSA), no sentiment analysis.

## Sequencing logic

P0 items are hours-to-days and remove active risk (a live public credential, a
config landmine, three contradictory statements of what version this is). P1–P2
are the difference between "the author's working tree" and "a release with
evidence". P3 is the smallest product surface worth calling 1.0 for self-hosting,
with P3.1/P3.2/P3.5 doing the most work per unit of effort. P4 is the gate you
cannot skip: it is the phase where someone other than the author proves it works.
