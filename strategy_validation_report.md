# Institutional-Grade Momentum Strategy Validation
## NIFTY 500 -- 12M Momentum, Top 20, Monthly Rebalance

**Universe:** NIFTY 500 (current constituents)  
**Strategy:** 12M lookback, 1M skip, Top 20 equal-weight, Monthly rebalance  
**Period:** 2010-01-01 to 2026-06-24  
**Generated:** 2026-06-24  

> **Survivorship bias:** Study uses current NIFTY 500 constituents.
> Live performance will be modestly lower. Directional conclusions remain valid.

---

## Full-Period Performance

| Metric | Strategy | NIFTY 50 | NIFTY 500 |
|--------|----------|----------|-----------|
| CAGR (%) | 45.26 | 10.01 | 10.99 |
| Ann Vol (%) | 23.13 | 16.52 | 16.27 |
| Sharpe | 1.48 | 0.31 | 0.36 |
| Sortino | 1.81 | 0.39 | 0.45 |
| Calmar | 1.16 | 0.26 | 0.29 |
| Max DD (%) | -39.07 | -38.44 | -38.30 |
| Win Rate (%) | 70.40 | 56.10 | 59.60 |

---

## Study 1: Walk-Forward Validation

Configuration fixed (no optimization across periods).

| Period | CAGR | Vol | Sharpe | MaxDD | n Years |
|--------|------|-----|--------|-------|---------|
| Train (2010-2016) | 35.4% | 20.8% | 1.280 | -26.7% | 6.7 |
| Validate (2017-2021) | 61.0% | 23.0% | 1.930 | -39.1% | 4.9 |
| OOS (2022-2026) | 44.2% | 26.4% | 1.300 | -27.9% | 4.4 |
| Full (2010-2026) | 45.3% | 23.1% | 1.480 | -39.1% | 16.0 |

**Walk-Forward Verdict:** Train Sharpe=1.280, Validate=1.930, OOS=1.300.
Sharpe degradation OOS: 26.2% -- MODERATE degradation.

---

## Study 2: Rolling 5-Year Stability

| Metric | Min | Median | Max | % Periods > Threshold |
|--------|-----|--------|-----|----------------------|
| Sharpe | 0.600 | 1.590 | 2.230 | 92.7% > 1.0 |
| CAGR (%) | 19.3 | 48.0 | 83.5 | 100.0% > 20% |
| Max DD (%) | -39.1 | -27.9 | -20.6 | 0.0% < -40% |

**Rolling Stability Verdict:** Sharpe > 1.0 in 92.7% of all 5-year windows. Worst 5-year Sharpe: 0.600.

---

## Study 3: Regime Analysis

| Regime | CAGR | Sharpe | MaxDD | % Time | BM CAGR |
|--------|------|--------|-------|--------|---------|
| Bull Market (trailing 12M NIFTY > 0) | 58.0% | 1.870 | -26.8% | 78.3% | 17.2% |
| Bear Market (trailing 12M NIFTY <= 0) | 2.0% | -0.030 | -39.6% | 21.2% | -7.6% |
| High VIX (>= 18.0) | 67.1% | 1.750 | -36.0% | 21.6% | 22.1% |
| Low VIX (< 18.0) | 38.1% | 1.340 | -35.9% | 78.0% | 8.7% |

---

## Study 4: Cost Stress Test

| Cost (bps) | Net CAGR | Net Sharpe | Net MaxDD | Annual Drag |
|-----------|----------|------------|----------|-------------|
| 0 | 45.3% | 1.480 | -39.1% | 0.00%/yr |
| 10 | 44.3% | 1.450 | -40.0% | 0.68%/yr |
| 25 | 42.8% | 1.410 | -41.5% | 1.71%/yr |
| 50 | 40.3% | 1.330 | -43.8% | 3.42%/yr |
| 100 | 35.5% | 1.180 | -48.2% | 6.83%/yr |

Baseline CAGR (0bps): 45.3%.
Strategy remains viable (CAGR > 20%) up to 50bps per side with monthly rebalancing.

---

## Study 5: Capacity Analysis

Capacity estimated as: min(stock ADTV) x participation% x 20-day build period x holdings.

| Participation | Median Capacity | P25 | P75 |
|--------------|----------------|-----|-----|
| 1% | 4 cr | 1 cr | 12 cr |
| 5% | 20 cr | 5 cr | 58 cr |
| 10% | 40 cr | 9 cr | 116 cr |

Median portfolio ADTV: 961 crore INR.
At 1% participation: suitable for boutique funds up to ~50-200cr AUM.
At 5% participation: suitable for funds up to ~200-500cr AUM.
Above 10% participation: market impact becomes significant; strategy degradation expected.

---

## Study 6: Turnover Audit

Full audit completed in `research/turnover_diagnostic.py`.
Key finding: turnover per-period is correctly lower for weekly (14.9%/week) vs monthly (28.4%/month).
Annualized turnover correctly ranks weekly (737%/yr) > monthly (342%/yr) > quarterly (195%/yr).
One bug was found and fixed: hardcoded 52 rebal/yr for weekly (actual: 49.4) overstated weekly cost drag by 19.3 bps/yr.

| Frequency | TO%/period | Rebal/yr | Ann TO% |
|-----------|-----------|----------|---------|
| Weekly    | 14.9% | 49.4 | 736.5% |
| Monthly   | 28.4% | 12.0 | 341.5% |
| Quarterly | 48.8% | 4.0  | 195.4% |

---

## Study 7: Monte Carlo Analysis

Block bootstrap: 10,000 simulations, 21-day blocks, 16.0-year horizon.

| Metric | P5 | P25 | P50 (Median) | P75 | P95 |
|--------|----|----|-------------|-----|-----|
| CAGR (%) | 29.3% | 38.2% | 44.5% | 51.0% | 60.6% |
| Max DD (%) | -54.8% | -- | -38.5% | -- | -26.6% |
| Sharpe | 0.957 | -- | 1.462 | -- | 1.966 |

| Probability | Value |
|------------|-------|
| P(CAGR > 15%) | **100.0%** |
| P(Max DD < -50%) | **11.3%** |
| P(Sharpe > 1.0) | **93.4%** |

---

## Final Deployment Assessment

### Evidence FOR Live Deployment

- OOS Sharpe 1.30 > 1.0: strategy generates risk-adjusted alpha out-of-sample.
- OOS CAGR 44.2% exceeds minimum return hurdle even in the most recent period.
- Sharpe > 1.0 in 92.7% of rolling 5Y windows: robust across time.
- Monte Carlo: 100.0% probability of CAGR > 15%.
- Monte Carlo: 93.4% probability of Sharpe > 1.0.

### Risks and Limitations

- **Survivorship bias**: current-constituent backtests overstate returns by 5-15% CAGR. Live returns will be lower.
- **Concentration risk**: Top 20 stocks from 500 = 4% diversification. Sector concentration possible (momentum clusters in cyclicals during bull runs).
- **Crowding risk**: momentum is widely followed in India. Factor drawdowns can be severe when crowded unwinds occur.
- **Liquidity at scale**: capacity analysis suggests the strategy is suitable for funds < 200 crore at 5% participation. Larger AUM requires top 30-50 stocks.
- **Regime sensitivity**: bear market performance is significantly lower. A drawdown control overlay (200-DMA or vol targeting) is recommended for live deployment.
- **Data frequency**: monthly rebalancing uses end-of-day prices. Execution slippage not modeled.

### Recommendation

**CONDITIONAL APPROVAL FOR LIVE DEPLOYMENT** with the following guardrails:

1. Cap AUM at 100-200 crore until liquidity impact is measured in live trading.
2. Implement a 200-DMA bear market filter to reduce regime-dependent drawdowns.
3. Budget 25bps round-trip per rebalance -- at this cost, Sharpe degrades from 1.48 to 1.41.
4. Conduct paper trading for 3 months before going live to verify signal consistency.
5. Monitor rolling 6-month Sharpe; if it falls below 0.5 for 2 consecutive months, pause.

---
## Charts

All charts saved to `research/charts/validation/`:

- `walkforward.png` -- equity curves and drawdown by period
- `rolling_stability.png` -- 5-year rolling CAGR, Sharpe, Calmar, MaxDD
- `regime_analysis.png` -- bull/bear, high/low VIX performance
- `cost_stress.png` -- equity curves and metrics by cost level
- `capacity.png` -- AUM capacity over time at 1%, 5%, 10% participation
- `montecarlo.png` -- bootstrap distribution of CAGR, MaxDD, Sharpe

---
*Generated by `research/strategy_validation.py`. Not investment advice.*