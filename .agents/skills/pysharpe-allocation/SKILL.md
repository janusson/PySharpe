---
name: pysharpe-allocation
description: >-
  PySharpe value averaging (VA) allocation and rebalancing engine. Invoke
  when working with the deterministic VA strategy, contribution allocation
  (allocate_contribution), opportunity scoring (score_opportunities: 60%
  path drift + 40% fundamental valuation/mean-reversion by default),
  rebalancing plans (build_rebalance_plan), tax tracking for Canadian TFSA
  accounts, foreign withholding tax drag on US dividends (VFV/QQC yield
  reduction), or AllocationConfig tuning. STRICT BOUNDARIES: no tax-loss
  harvesting (TFSA prohibits), no Markowitz mean-variance solvers (VA is
  authoritative), no single-stock models (broad-market CAD ETFs only).
---

# PySharpe Value Averaging Allocation Engine

Covers the deterministic value averaging strategy, contribution allocation,
opportunity scoring, rebalancing, and tax tracking for Canadian TFSA accounts.

## Module Locations

- `src/pysharpe/execution/allocator.py` — `score_opportunities()`,
  `allocate_contribution()`, `AllocationConfig`.
- `src/pysharpe/execution/rebalance.py` — `build_rebalance_plan()`.
- `src/pysharpe/execution/tax_tracker.py` — TFSA tax tracking.

## Value Averaging Strategy

PySharpe's primary allocation engine implements a **deterministic Value
Averaging (VA)** strategy. The core objective: calculate the capital
contribution required for each asset to meet a pre-defined, compounding target
value path.

### Opportunity Scoring

`score_opportunities()` computes a blended score for each asset:

$$\text{Score} = 0.6 \times \text{PathDrift} + 0.4 \times \text{ValuationMeanReversion}$$

- **Path Drift (60%)**: How far the current value deviates from the target
  value path. Below-target assets score higher (more attractive to buy).
- **Valuation/Mean-Reversion (40%)**: Fundamental valuation signal. Above
  long-term mean → lower score; below mean → higher score.

**These weights are stable and authoritative. Do not alter the 60/40 blend
unless explicitly instructed.**

Configuration is via `AllocationConfig` dataclass — the weights can be
overridden there, but the default is canonical.

### Contribution Allocation

`allocate_contribution()` converts opportunity scores into dollar amounts:

1. Normalize scores to a 0–1 range.
2. Scale by the total contribution budget.
3. Apply asset-level minimum/maximum contribution bounds.
4. Ensure the total allocation does not exceed the budget.

### Standard Allocation Template

For a runnable template demonstrating the full allocation pipeline (score
calculation, normalization, budget allocation, and bounds enforcement), load:

- `scripts/allocation-pipeline.py`

## Rebalancing

`build_rebalance_plan()`:
1. Loads saved optimization artefacts (`<name>_weights.txt`,
   `<name>_collated.csv`) from `data/exports/`.
2. Merges target weights with current holdings.
3. Calls `allocate_contribution()` to determine buy/sell amounts.
4. Returns a rebalance plan with dollar amounts per asset.

## TFSA Constraints

All portfolios operate within Canadian Tax-Free Savings Account rules:

- **Capital gains are tax-exempt** — no capital gains tax logic.
- **Losses cannot be claimed** — **tax-loss harvesting (TLH) is strictly
  prohibited.** Never implement TLH.
- **Foreign withholding tax** on US dividends (e.g., VFV, QQC): modeled as a
  strict yield reduction on the foreign income portion. This is a yield drag,
  not a capital gains concern.
- **No tax-bracket routing** — TFSA is flat; do not build progressive tax
  layers.

## Asset Universe

The allocation engine targets **broad-market, CAD-denominated index ETFs**
(e.g., VFV, VDY, QQC). Strict boundaries:

- **No single-stock models** — no idiosyncratic volatility, sentiment
  analysis, or stock-specific risk factors.
- **No options pricing** — pure equity ETF allocation.
- **No predictive price models** — VA is deterministic, not forecast-driven.

## Critical Guardrails

- **No TLH**: `grep -rn 'tax.loss\|TLH\|loss.harvest' src/pysharpe/ --include='*.py'` must return zero results.
- **60/40 weights**: `grep -rn '0\.6\|0\.4\|path_drift\|valuation_weight' src/pysharpe/execution/allocator.py`
- **No MPT**: The allocator uses VA, not Markowitz. Optimization (efficient
  frontier) is in `pysharpe-optimization`.

## Testing

Relevant test files:
- `tests/test_2d_allocation.py`
- `tests/test_rebalance.py`
- `tests/test_tax_tracker.py`

Run: `uv run pytest tests/test_2d_allocation.py tests/test_rebalance.py`
