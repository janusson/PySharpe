# PySharpe

> **Evidence-based portfolio optimization for Canadian investors.** Construct,
> compare, and validate long-term investment portfolios using modern financial
> research — with every recommendation traceable to published literature,
> transparent assumptions, and reproducible quantitative analysis.

[![Build status](https://github.com/janusson/PySharpe/actions/workflows/ci.yml/badge.svg)](https://github.com/janusson/PySharpe/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/janusson/PySharpe/ci.yml?label=docs)](https://github.com/janusson/PySharpe/actions/workflows/ci.yml)
[![Coverage gate](https://img.shields.io/badge/coverage-75%25%2B-success)](https://github.com/janusson/PySharpe/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://github.com/janusson/PySharpe/blob/main/pyproject.toml)
[![Pyright strict](https://img.shields.io/badge/pyright-strict%20%2B%20warnings--fatal-3178c6)](https://github.com/janusson/PySharpe/blob/main/pyrightconfig.json)
[![Ruff](https://img.shields.io/badge/ruff-lint%20%2B%20format-d7ff64)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/uv-locked%20dependencies-261230)](https://docs.astral.sh/uv/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/janusson/PySharpe/blob/main/LICENSE)

PySharpe is a portfolio research platform that **does not predict the market**.
It mathematically manages risk, minimizes uncompensated drag, and optimizes for
after-tax real wealth, calibrated to the structural realities of Canadian
retail investing: registered account types, foreign withholding tax treaties,
CAD/USD conversion frictions, and CRA tax rules.

## Quick Start

```bash
git clone https://github.com/janusson/PySharpe.git
cd PySharpe
uv pip install -e ".[all]"

# Launch the Streamlit dashboard
uv run streamlit run app.py

# Or drive it from the CLI
uv run pysharpe --help
```

## Highlights

### 🇨🇦 2-D Asset Location Engine

Simultaneously solves *what to hold* **and** *where to hold it* across TFSA,
RRSP, FHSA, LIRA, RRIF, and Non-Registered accounts, using tax-adjusted expected
returns that price in US foreign withholding tax treaties and unrecoverable
fund-level FWT on CAD-wrapped US ETFs.

### ⚖️ CRA Superficial Loss Guardrail

A compliance interlock, not just math: loss-sales in a Non-Registered account
conflict with any identical-property purchase in a sheltered account within
±30 days (same-day included). Violating buys are blocked and re-routed to the
next-best non-identical asset. ACB tracking follows the CRA weighted-average
method (ITA s. 47(1)) with commission-adjusted cost bases.

### 🤖 Agentic Development Framework (`.agents/skills/`)

The repository ships a modular context protocol for LLM coding agents: each
domain owns a skill package encoding its architecture, guardrails, test maps,
and prohibitions. Agents load only the skills a task touches, so an allocation
change cannot violate the 60/40 VA scoring blend and a data-pipeline change
cannot reintroduce `.bfill()` lookahead.

### 🧮 Shrinkage & Uncertainty Everywhere

Bayes-Stein expected returns, analytical Ledoit-Wolf linear and nonlinear
covariance shrinkage (eigen-clipped to strict positive-definiteness), PyMC
posterior estimation feeding `EfficientFrontier` through the posterior
covariance — never the sample — and Hierarchical Risk Parity as an
inversion-free fallback.

### 🔬 Overfitting-Resistant Validation

Deflated Sharpe Ratio with Lo's autocorrelation adjustment, Probability of
Backtest Overfitting, and Combinatorial Purged Cross-Validation with
autocorrelation-decay gap sizing. Backtests charge bid-ask spread, slippage,
and commissions using only execution-date prices.

### 🏗️ Production-Grade Engineering

- **Strict typing**: Pyright `--warnings` promoted to fatal — 0 errors, 0 warnings.
- **Ruff** sole linter + formatter; **uv** lockfile reproducibility.
- **CI**: lint + typecheck, tests with a 75% coverage floor, strict docs build.
- **975+ tests**, all synthetic with fixed seeds — no network calls.
- **DuckDB write-through cache** with mtime invalidation — stale data is impossible.

## Documentation

| Page | Contents |
| ---- | -------- |
| [Architecture](architecture.md) | Module boundaries and pipeline layering |
| [Data Flow](flowchart.md) | Mermaid diagram of the data pipeline |
| [Test Map](TEST_MAP.md) | Test-file to source-module mapping |
| [Gotchas](GOTCHAS.md) | Known failure patterns and guardrails |
| [Benchmarks](benchmarks.md) | Canadian ETF baselines |
| [Proxy Mapping](proxies.md) | Ticker proxy resolution and FX adjustment |
| [API Reference](api.md) | Auto-generated API documentation |

## The Evidence Canon

PySharpe organizes its optimization engine around a tiered evidence system.
Higher tiers are opt-in and must demonstrably improve on the null hypothesis
net of taxes, fees, and behavioral friction.

| Tier | Label | Behavior | Examples |
|------|-------|----------|----------|
| **0** | Null Hypothesis | Baselines. Any higher-tier recommendation must justify its divergence. | Market-cap indexing (`VEQT`, `XEQT`, `S&P 500`), equal-weight 1/N |
| **1** | Canonical | Default engine parameters. | Markowitz mean-variance, Sharpe ratio maximization, CRA superficial loss rules, basic asset location |
| **2** | Strong Evidence | Opt-in via CLI flags or config toggles. | Ledoit-Wolf covariance shrinkage, Bayes-Stein expected return shrinkage, Hierarchical Risk Parity, Purged/Embargoed cross-validation |
| **3** | Experimental | Sandboxed and flagged to the user. | LLM-generated active views for Black-Litterman, novel signal generation |

## Developer Workflow

```bash
make install      # uv pip install -e ".[dev]"
make lint         # ruff check + format check
make typecheck    # pyright with warnings-as-errors
make test         # pytest with coverage (fails below 75%)
make check        # full pre-commit gate
make build_docs   # this site, built in --strict mode
```

## License

MIT — see the [repository license](https://github.com/janusson/PySharpe/blob/main/LICENSE).
