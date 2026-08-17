# MAX Effect / Indian Equity Factor Project — Master Summary

**Universe:** NIFTY 500 (current constituents, yfinance data) · **Period:** 2010-01-01 → 2026-06 (≈16 years)
**Benchmarks:** NIFTY 50 (10.0% CAGR, Sharpe 0.31) · NIFTY 500 (11.0% CAGR, Sharpe 0.36)
**Reports consolidated:** 7 · **Charts produced:** ~80 · **Code modules:** 24 Python files

---

## 1. What the project set out to do

Test whether the **MAX lottery-stock anomaly** (Bali, Cakici & Whitelaw 2011 — stocks with the highest
recent single-day gains underperform) holds in Indian equities. When it did not, the project expanded
into a full factor-research programme: screening six factors, building a production momentum strategy,
and subjecting it to institutional-grade validation.

---

## 2. Work completed (7 studies)

| # | Study | Script | Output |
|---|-------|--------|--------|
| 1 | MAX factor baseline backtest | `main.py` | `backtest_metrics.csv`, 6 charts |
| 2 | MAX comprehensive validation (8 audits) | `run_validation.py` + `validation/` | `validation_report.md`, 15 charts |
| 3 | Six-factor decile research | `research/run_research.py` | `factor_research_report.md`, 32 charts |
| 4 | Momentum parameter study (6 studies + grids) | `research/momentum_study.py` | `momentum_study_report.md`, 4 grid CSVs |
| 5 | Momentum × MAX composite | `research/mom_max_composite.py` | `mom_max_composite_report.md` |
| 6 | Momentum × Low-Vol combination | `research/combo_mom_lowvol.py` | `combo_mom_lowvol_report.md` |
| 7 | Institutional validation + 2 survivorship studies | `research/strategy_validation.py`, `survivorship_*.py` | 3 reports, 10 charts |

---

## 3. Finding 1 — The MAX anomaly does **not** hold in India

| Portfolio | CAGR | Vol | Sharpe | Max DD |
|-----------|------|-----|--------|--------|
| Low MAX (D1) | 20.76% | 14.1% | **0.99** | -40.4% |
| High MAX (D10) | 27.18% | 25.4% | 0.85 | -65.8% |
| L/S Spread (D10−D1) | **−8.48%** | 16.9% | −0.78 | -85.9% |

High-MAX stocks earned *more*, the opposite of the US result — but **low-MAX wins on Sharpe**.
Three co-existing explanations were isolated and quantified:

1. **Momentum confound (CONFIRMED).** IC(MAX, MOM_1M) = **0.526** (t=40.2, positive 99.5% of months);
   IC(MAX, MOM_12M) = 0.210. MAX is largely proxying for momentum in India's bull market.
   After orthogonalising MAX against MOM_1M/3M/12M, the spread narrows to **−3.2%** but never turns
   positive — the anomaly is *not* recovered.
2. **Risk premium, not mispricing (CONFIRMED).** D10 vol 25% vs D1 vol 14%; the raw return advantage
   is fully explained by higher systematic risk (D1 Sharpe > D10 Sharpe).
3. **Survivorship bias (MODERATE).** Only 55.2% of current NIFTY 500 members had data in 2010;
   the top-MAX decile holds 73.3% full-history survivors. Estimated to overstate the high-MAX
   advantage by 5–15pp of CAGR.

**Robustness:** the reversal persists across all liquidity screens (ADTV >1cr −8.5%, >10cr −7.1%,
>25cr −5.8%) and all universes (NIFTY100 −5.8% → SMALLCAP250 −14.0%), i.e. it is stronger in small caps.
**Look-ahead audit: CLEAN — 4/4 tests passed** (temporal integrity, phantom leak, fence-post, index integrity).

**Verdict:** MAX is not a tradable short signal in India. Low-MAX is usable only as a *defensive/low-vol tilt*.

---

## 4. Finding 2 — Factor screen: momentum is the only strong signal

| Factor | D1 CAGR | D10 CAGR | Spread | D10 Sharpe | Monotonicity ρ | Signal |
|--------|---------|----------|--------|-----------|----------------|--------|
| **MOMENTUM_12M** | 14.7% | **41.0%** | **+20.9%** | **1.43** | 0.973*** | **WORKS** |
| MOMENTUM_6M | 14.4% | 40.0% | +20.8% | 1.40 | 0.964*** | WORKS |
| MAX | 20.8% | 27.2% | +6.2% | 0.85 | 0.648* | risk premium only |
| LOW_VOL | 27.5% | 19.4% | −10.7% | 1.09 | −0.818** | reversed (but best Sharpe tilt) |
| QUALITY | 0.5% | −11.1% | −17.3% | −0.35 | −0.333 ns | flat / no signal |
| VALUE | 5.4% | −2.5% | −9.6% | −0.16 | −0.091 ns | flat / no signal |

Momentum shows near-perfect decile monotonicity in both CAGR and Sharpe — the strongest, cleanest
result in the whole project. Quality and Value showed no usable signal on the available fundamentals.

---

## 5. Finding 3 — Production momentum strategy (the main deliverable)

**Optimal configuration** (from a 6-part parameter study + 4 heatmap grids):
**12M lookback · 1M skip · Top 20 equal-weight · Monthly rebalance · ATR risk filter · ≤25bps cost budget**

| Metric | Strategy | NIFTY 50 | NIFTY 500 |
|--------|----------|----------|-----------|
| CAGR | **45.3%** | 10.0% | 11.0% |
| Ann Vol | 23.1% | 16.5% | 16.3% |
| Sharpe | **1.48** | 0.31 | 0.36 |
| Sortino / Calmar | 1.81 / 1.16 | 0.39 / 0.26 | 0.45 / 0.29 |
| Max DD | −39.1% | −38.4% | −38.3% |
| Win rate | 70.4% | 56.1% | 59.6% |

Parameter sensitivity was low — Sharpe stayed in 1.16–1.48 across every lookback tested, indicating
the result is not curve-fit. Skip period had minimal impact (Indian momentum autocorrelation is high).
Weekly rebalancing wins pre-cost (1.49) but loses to monthly after costs.

### Validation evidence (7 studies)

- **Walk-forward:** Train 2010-16 Sharpe 1.28 → Validate 2017-21 1.93 → **OOS 2022-26 1.30**. 26% degradation — moderate, acceptable.
- **Rolling 5Y stability:** Sharpe > 1.0 in **92.7%** of all 5-year windows; worst window 0.60; CAGR > 20% in 100% of windows.
- **Regimes:** Bull 58.0% CAGR (Sharpe 1.87) vs **Bear 2.0% CAGR (Sharpe −0.03)** — the key weakness.
- **Cost stress:** viable to 50bps (40.3% CAGR); at 25bps Sharpe 1.48 → 1.41 (1.7%/yr drag).
- **Capacity:** median portfolio ADTV ₹961cr → ~₹200–500cr AUM at 5% participation.
- **Monte Carlo** (10,000 block-bootstraps): P(CAGR>15%) = **100%**, P(Sharpe>1.0) = **93.4%**, P(MaxDD < −50%) = 11.3%.
- **Turnover audit:** one real bug found and fixed — hardcoded 52 weekly rebalances/yr (actual 49.4) had overstated weekly cost drag by 19.3 bps/yr.

### Concentration & survivorship haircut

- Gini **0.541**; top 20 of 332 stocks ever held = 32.8% of total return. But excluding the top 10
  contributors still leaves **39.8% CAGR / Sharpe 1.31** — **robust to winner exclusion**.
- Survivorship correction (listing-date filter, market-cap proxy, combined) costs 1.1–2.9pp of CAGR.
  Adding literature calibration (Agarwalla 2014, Sehgal & Tripathi 2005: 1.5–4% for India):
  **realistic live CAGR estimate 37.2%–41.2%** — still >3x the benchmark.

### Combination attempts — both rejected

- **Momentum + MAX overlay:** every weighting *reduced* Sharpe (1.48 → 1.25–1.34) and tripled turnover. **MAX does not improve momentum.**
- **Momentum + Low-Vol:** only 1 of 3 blends beat pure momentum (70/30 combo, Sharpe 1.42 vs 1.40) while giving up 4.6pp of CAGR. **Verdict: MIXED — not worth the complexity.**

---

## 6. Bottom line & recommendation

**CONDITIONAL APPROVAL FOR LIVE DEPLOYMENT** of the 12M momentum strategy, with guardrails:

1. Cap AUM at ₹100–200cr until live market impact is measured.
2. Add a **200-DMA bear-market filter** — bear-regime Sharpe of −0.03 is the strategy's main flaw.
3. Budget 25bps round-trip; monthly rebalancing only.
4. Paper-trade 3 months before going live.
5. Kill-switch: pause if rolling 6-month Sharpe < 0.5 for two consecutive months.
6. Consider a 10% per-stock concentration cap given the 0.541 Gini.

**Principal caveat across all results:** the entire project uses *current* NIFTY 500 constituents.
Point-in-time index membership is not freely available; ~200–250 stocks that were in the 2010 index
and were later dropped are absent from the dataset entirely. All proxy corrections therefore
*understate* the true bias. Directional conclusions are robust; absolute return levels are not.

---
*Consolidated from `validation_report.md`, `research/factor_research_report.md`, `momentum_study_report.md`,
`mom_max_composite_report.md`, `combo_mom_lowvol_report.md`, `strategy_validation_report.md`,
`survivorship_report.md`, `survivorship_bias_report.md`. Not investment advice.*
