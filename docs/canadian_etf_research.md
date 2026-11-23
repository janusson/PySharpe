# Reproducible Canadian ETF Portfolio Research Examples

This guide demonstrates how a Canadian DIY investor can use PySharpe's public APIs
to evaluate, compare, and validate long-term portfolio structures using reproducible
quantitative methods.

All data, calculations, and simulations in this example are **100% deterministic and offline**,
requiring zero external network requests. Fictional ETF identifiers are used to ensure
synthetic simulation results are never conflated with actual historical performance of live funds.

---

## 1. Portfolio Structures & Conceptual Analogues

Canadian self-directed investors commonly deliberate between single-country US equity,
globally diversified all-equity solutions, sector or tech tilts, and classic balanced asset allocations.
We construct five representative portfolio architectures:

| Portfolio Key | Architecture Description | Target Weights | Conceptual Analogue |
| :--- | :--- | :--- | :--- |
| **`US_LargeCap`** | 100% US Large-Cap Equity | `F_US_LC`: 100% | Vanguard S&P 500 Index ETF (`VFV.TO`) |
| **`Global_AllEquity`** | Globally Diversified All-Equity (45% US, 30% CA, 25% INTL) | `F_US_LC`: 45%<br>`F_CAN_EQ`: 30%<br>`F_INTL_EQ`: 25% | Vanguard All-Equity ETF Portfolio (`VEQT.TO`) / iShares (`XEQT.TO`) |
| **`US_TechTilt`** | US Equity with High-Beta Technology Tilt | `F_US_LC`: 60%<br>`F_TECH_TILT`: 40% | Core S&P 500 (`VFV.TO`) + NASDAQ-100 (`QQC.TO`) |
| **`Balanced_60_40`** | Balanced 60/40 Equity & Fixed Income | `F_US_LC`: 28%<br>`F_CAN_EQ`: 18%<br>`F_INTL_EQ`: 14%<br>`F_CAN_BOND`: 40% | Vanguard Balanced ETF Portfolio (`VBAL.TO`) |
| **`Equal_Weight_1_over_N`** | Naive $1/N$ Unconstrained Baseline | 20% in each of the 5 constituent assets | Equal-weighted cross-asset benchmark |

### Fictional ETF Universe

To prevent lookahead bias and avoid misrepresenting synthetic models as historical fund performance,
the research uses five synthetic series with realistic return and covariance characteristics:

- **`F_US_LC`**: Synthetic CAD-denominated US Large Cap Equity (annual drift $\sim 13\%$, vol $\sim 16\%$).
- **`F_CAN_EQ`**: Synthetic Canadian Broad Market Equity (annual drift $\sim 10\%$, vol $\sim 15\%$, dividend-heavy).
- **`F_INTL_EQ`**: Synthetic International Developed & Emerging Markets Equity (annual drift $\sim 9\%$, vol $\sim 14\%$).
- **`F_TECH_TILT`**: Synthetic High-Beta Growth/Technology Equity (annual drift $\sim 14\%$, vol $\sim 25\%$, elevated downside kurtosis).
- **`F_CAN_BOND`**: Synthetic Canadian Aggregate Fixed Income (annual drift $\sim 4.5\%$, vol $\sim 5\%$, near-zero equity correlation).

---

## 2. Research Methodology

### Evaluation Windows
Financial models frequently suffer from in-sample over-fitting. To demonstrate robust validation:
1. **In-Sample (IS) Training Window (2019-01-02 to 2021-12-31, 755 trading days):**
   Used to estimate historical baseline metrics, cross-asset correlations, and portfolio risk parameters.
2. **Out-of-Sample (OOS) Testing Window (2022-01-03 to 2023-12-29, 504 trading days):**
   Used for forward evaluation of risk-adjusted performance, drawdowns, and rebalancing friction.

### Financial Assumptions
- **Risk-Free Rate ($R_f$):** 2.50% annualized, representing Canadian cash yields or Government of Canada short-term Treasury Bills.
- **Trading Days per Year:** 252 days.
- **Transaction Frictions:** CAD $4.95 fixed ticket commission per traded asset, 10 basis points (bps) bid-ask half-spread, and 5 bps market-impact slippage.
- **Value Averaging Objective:** Deterministic 60% path-drift deviation + 40% fundamental valuation.

---

## 3. Performance & Risk Comparison

### Out-of-Sample Forward Performance (2022 – 2023)

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Sortino | Max Drawdown | CAGR | HHI | Effective N | Diversification Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`US_LargeCap`** | 9.92% | 15.74% | 0.47 | 0.76 | -19.39% | 9.70% | 1.0000 | 1.00 | 1.00 |
| **`Global_AllEquity`** | 10.49% | 13.23% | 0.60 | 0.93 | -14.35% | 10.72% | 0.3550 | 2.82 | 1.13 |
| **`US_TechTilt`** | **10.98%** | 20.32% | 0.42 | 0.71 | **-20.75%** | 10.29% | 0.5200 | 1.92 | 1.03 |
| **`Balanced_60_40`** | 8.65% | **8.02%** | **0.77** | **1.14** | **-8.64%** | 8.87% | 0.2904 | 3.44 | **1.38** |
| **`Equal_Weight_1_over_N`** | 9.89% | 12.80% | 0.58 | 0.88 | -13.92% | 9.84% | 0.2000 | 5.00 | 1.23 |

### In-Sample Baseline Performance (2019 – 2021)

| Portfolio | Ann. Return | Ann. Vol | Sharpe | Sortino | Max Drawdown | CAGR | HHI | Effective N | Diversification Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`US_LargeCap`** | 17.11% | 16.34% | 0.89 | 1.33 | -14.96% | 17.55% | 1.0000 | 1.00 | 1.00 |
| **`Global_AllEquity`** | 14.54% | 13.57% | 0.89 | 1.32 | -11.73% | 14.94% | 0.3550 | 2.82 | 1.13 |
| **`US_TechTilt`** | 13.42% | 17.65% | 0.62 | 0.97 | -16.72% | 13.73% | 0.5200 | 1.92 | 1.04 |
| **`Balanced_60_40`** | 12.70% | 8.39% | 1.22 | 1.77 | -7.66% | 13.11% | 0.2904 | 3.44 | 1.34 |
| **`Equal_Weight_1_over_N`** | 12.10% | 11.64% | 0.83 | 1.23 | -11.09% | 12.45% | 0.2000 | 5.00 | 1.24 |

---

## 4. Key Quantitative Insights

### 1. The Return vs. Risk-Adjusted Inversion
A central trap in DIY investing is selecting asset allocations based purely on nominal trailing return.
The synthetic experiment highlights this danger:

- **Highest Nominal Return:** `US_TechTilt` delivered the highest annualized return in the out-of-sample window (**10.98%** annualized, **10.29%** CAGR).
- **Worst Risk Efficiency:** However, it suffered the **lowest Sharpe ratio (0.42)**, the **lowest Sortino ratio (0.71)**, and the **deepest maximum drawdown (-20.75%)**.
- **Best Risk-Adjusted Allocation:** Conversely, `Balanced_60_40` delivered a more modest annualized return (**8.65%**), but achieved the **highest Sharpe ratio (0.77)**, the **highest Sortino ratio (1.14)**, and cut drawdown risk by more than half (**-8.64%** vs. -20.75%).

### 2. Benefits of Geographic Diversification
Comparing `Global_AllEquity` (VEQT/XEQT style) to single-country `US_LargeCap` (VFV style):
- Volatility decreased from **15.74% to 13.23%** (a 2.51% volatility reduction).
- Maximum drawdown improved from **-19.39% to -14.35%** (a 5.04% downside cushion).
- Out-of-sample Sharpe ratio rose from **0.47 to 0.60**.

### 3. Portfolio Concentration & Effective Number of Bets
Using the Herfindahl-Hirschman Index ($HHI = \sum w_i^2$) and Inverse Concentration ($N_{\text{eff}} = 1 / HHI$):
- `US_LargeCap`: $HHI = 1.000$, $N_{\text{eff}} = 1.00$ (complete single-market concentration).
- `US_TechTilt`: $HHI = 0.520$, $N_{\text{eff}} = 1.92$.
- `Global_AllEquity`: $HHI = 0.355$, $N_{\text{eff}} = 2.82$.
- `Balanced_60_40`: $HHI = 0.290$, $N_{\text{eff}} = 3.44$.
- `Equal_Weight_1_over_N`: $HHI = 0.200$, $N_{\text{eff}} = 5.00$ (maximum mathematical parity).

### 4. Diversification Ratio
The Diversification Ratio ($DR = \frac{\sum w_i \sigma_i}{\sigma_p}$) measures how effectively constituent volatilities
cancel each other out through imperfect correlations:
- `US_LargeCap`: $DR = 1.00$ (no diversification).
- `Balanced_60_40`: $DR = 1.38$ (a 38% volatility reduction relative to weighted constituent risks), driven by the near-zero correlation between Canadian bonds and global equities.

---

## 5. Rebalancing & Execution Frictions

Rebalancing creates value by restoring target weights, but incurs commissions, bid-ask spread, and slippage.
PySharpe's `HistoricalBacktester` simulates quarterly rebalancing starting with CAD $100,000 across 2022–2023:

```text
Rebalancing Backtest (Quarterly Execution with Fees & Slippage):
  • US_LargeCap            | Terminal: CAD $120,040.94 | Net Return: 20.04% | Rebalances: 8
  • Global_AllEquity       | Terminal: CAD $122,354.28 | Net Return: 22.35% | Rebalances: 8
  • US_TechTilt            | Terminal: CAD $121,316.07 | Net Return: 21.32% | Rebalances: 8
  • Balanced_60_40         | Terminal: CAD $118,101.66 | Net Return: 18.10% | Rebalances: 8
  • Equal_Weight_1_over_N  | Terminal: CAD $120,354.81 | Net Return: 20.35% | Rebalances: 8
```

All transaction fees reduce cash and investable capital realistically; no capital is created or destroyed.

---

## 6. Value Averaging (VA) Smart Contribution Routing

For Canadian investors in registered accounts (TFSA, RRSP, FHSA), rebalancing by selling shares can trigger
unnecessary transaction costs or friction. PySharpe provides deterministic Value Averaging (VA)
contribution routing via `score_opportunities` and `allocate_contribution`.

When new savings are injected (e.g. CAD $1,000 monthly contribution), the engine scores assets
based on:
$$\text{Opportunity Score} = 0.60 \times \text{Underweight Drift} + 0.40 \times \text{Valuation Divergence}$$

### Contribution Allocation Example:
Given a CAD $100,000 portfolio tracking `Balanced_60_40` where fixed income has drifted 5.0% below target:

```text
    ticker  current_value  current_weight  target_weight  underweight  opportunity_score  allocated_dollars
F_CAN_BOND    CAD $35,000           35.0%          40.0%         5.0%              0.800      CAD $1,000.00
   F_US_LC    CAD $32,000           32.0%          28.0%         0.0%              0.200           CAD $0.00
  F_CAN_EQ    CAD $19,000           19.0%          18.0%         0.0%              0.200           CAD $0.00
 F_INTL_EQ    CAD $14,000           14.0%          14.0%         0.0%              0.200           CAD $0.00
```

The full CAD $1,000 is directed to `F_CAN_BOND` to rebalance the portfolio organically through new cash,
avoiding selling equity positions.

---

## 7. Mathematical Definitions & Verification

PySharpe's calculations enforce exact financial formulas rather than heuristic approximations:

1. **Annualized Compounded Growth Rate (CAGR):**
   $$\text{CAGR} = \left(\prod_{t=1}^T (1 + R_t)\right)^{\frac{252}{T}} - 1$$

2. **Annualized Volatility:**
   $$\sigma_{\text{ann}} = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (R_t - \bar{R})^2} \times \sqrt{252}$$

3. **Annualized Sharpe Ratio:**
   $$\text{Sharpe} = \frac{R_{\text{ann}} - R_f}{\sigma_{\text{ann}}}$$

4. **Annualized Sortino Ratio:**
   $$\text{Sortino} = \frac{R_{\text{ann}} - R_f}{\sigma_{\text{downward}}}$$
   where:
   $$\sigma_{\text{downward}} = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(R_t - R_{f,\text{daily}}, 0)^2} \times \sqrt{252}$$

5. **Maximum Drawdown:**
   $$\text{MDD} = \min_{t} \left( \frac{V_t - \max_{\tau \le t} V_\tau}{\max_{\tau \le t} V_\tau} \right)$$

6. **Herfindahl-Hirschman Index & Effective N:**
   $$HHI = \sum_{i=1}^N w_i^2, \quad N_{\text{eff}} = \frac{1}{HHI}$$

7. **Diversification Ratio:**
   $$DR = \frac{\sum_{i=1}^N w_i \sigma_i}{\sqrt{w^T \Sigma w}}$$

---

## 8. Running the Example & Verification Suite

To run the complete example script locally:

```bash
export MPLCONFIGDIR=/tmp/mpl
uv run python examples/canadian_etf_research.py
```

To run the dedicated regression and smoke test suite:

```bash
uv run pytest tests/test_canadian_etf_research_example.py -v
```
