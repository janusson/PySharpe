# PySharpe Examples

This directory contains executable examples and research notebooks demonstrating
how to use PySharpe for quantitative portfolio analysis, asset allocation, and
risk evaluation.

All examples use deterministic synthetic data and execute fully offline without
network dependencies.

---

## Catalog

### 1. [Canadian ETF Portfolio Research](canadian_etf_research.py)
- **File:** `examples/canadian_etf_research.py`
- **Companion Documentation:** [`docs/canadian_etf_research.md`](../docs/canadian_etf_research.md)
- **Description:** A standalone, fully reproducible research example comparing five
  long-term portfolio structures (US Large-Cap, Global All-Equity, US Tech Tilt,
  Balanced 60/40, and Equal Weight 1/N) calibrated for Canadian DIY investors.
- **Key Concepts Covered:**
  - In-Sample (2019–2021) vs. Out-of-Sample (2022–2023) performance evaluation.
  - Risk-return inversion: demonstrating why the highest nominal return portfolio is not necessarily the best risk-adjusted choice.
  - Concentration metrics ($HHI$, Effective $N$) and Diversification Ratio.
  - Quarterly rebalancing backtest with realistic transaction costs (commissions, bid-ask spread, slippage).
  - Value Averaging (VA) smart contribution routing under the 60% path-drift / 40% valuation heuristic.
  - Independent analytical verification asserting mathematical precision against pure NumPy formulas.
- **Run Command:**
  ```bash
  export MPLCONFIGDIR=/tmp/mpl
  uv run python examples/canadian_etf_research.py
  ```

### 2. [Quickstart Notebook](quickstart.ipynb)
- **File:** `examples/quickstart.ipynb`
- **Description:** Interactive Jupyter notebook introducing PySharpe's core optimization workflows,
  efficient frontier generation, and performance metric calculations.

### 3. [Opportunity Scoring Analysis](scoring_analysis.ipynb)
- **File:** `examples/scoring_analysis.ipynb`
- **Description:** In-depth analysis of the value averaging (VA) opportunity scoring algorithm,
  examining sensitivity to path drift and valuation parameters across market cycles.

---

## Testing & Verification

The example script is accompanied by an automated regression test suite:

```bash
uv run pytest tests/test_canadian_etf_research_example.py -v
```
