# Indian Equity Factor Research – Comprehensive Study

**Universe:** NIFTY 500 (current constituents)  
**Backtest:** 2010-01-01 to 2026-06-22  
**Generated:** 2026-06-22  
**Factors:** MAX, MOMENTUM_6M, MOMENTUM_12M, LOW_VOL, QUALITY, VALUE  

> **Survivorship bias caveat:** This study uses current NIFTY 500 constituents. Stocks that > delisted or dropped out of the index since 2010 are not included. This inflates returns, > particularly for high-volatility factors. Results are directionally informative but should > not be taken as live trading estimates.

---

## Executive Summary

| Factor | D1 CAGR | D10 CAGR | Spread | D1 Sharpe | D10 Sharpe | CAGR Mono | Signal |
|--------|---------|----------|--------|-----------|------------|-----------|--------|
| MAX | 20.8% | 27.2% | 6.2% | 0.990 | 0.850 | 0.648 | **POSITIVE** |
| MOMENTUM_6M | 14.4% | 40.0% | 20.8% | 0.450 | 1.400 | 0.964 | **POSITIVE** |
| MOMENTUM_12M | 14.7% | 41.0% | 20.9% | 0.460 | 1.430 | 0.973 | **POSITIVE** |
| LOW_VOL | 27.5% | 19.4% | -10.7% | 0.830 | 1.090 | -0.818 | **REVERSED** |
| QUALITY | 0.5% | -11.1% | -17.3% | -0.000 | -0.350 | -0.333 | **FLAT** |
| VALUE | 5.4% | -2.5% | -9.6% | 0.070 | -0.160 | -0.091 | **FLAT** |

> **Best factor by risk-adjusted return:** MOMENTUM_12M (best decile Sharpe: 1.430)

---

## MAX

### Decile Performance

| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |
|--------|----------|--------------|--------|---------|--------|------------|
| D1 | 20.76 | 14.12 | 0.990 | 1.230 | 0.510 | -40.43 |
| D2 | 20.52 | 15.50 | 0.910 | 1.140 | 0.560 | -36.36 |
| D3 | 21.14 | 16.42 | 0.900 | 1.120 | 0.550 | -38.38 |
| D4 | 20.36 | 17.04 | 0.830 | 1.030 | 0.510 | -39.58 |
| D5 | 21.39 | 18.28 | 0.830 | 1.050 | 0.500 | -42.71 |
| D6 | 21.17 | 19.00 | 0.800 | 0.970 | 0.480 | -43.78 |
| D7 | 20.22 | 19.89 | 0.730 | 0.910 | 0.390 | -52.51 |
| D8 | 22.88 | 20.89 | 0.810 | 1.000 | 0.470 | -48.65 |
| D9 | 24.31 | 22.61 | 0.820 | 1.020 | 0.450 | -53.56 |
| D10 | 27.18 | 25.41 | 0.850 | 1.060 | 0.410 | -65.79 |

### Factor Spread (D10 – D1)

| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |
|--------|----------|------------|-----------------|
| CAGR | 20.76% | 27.18% | 6.19% |
| Sharpe | 0.990 | 0.850 | — |
| Ann. Vol | 14.1% | 25.4% | — |
| Max DD | -40.4% | -65.8% | — |

### Monotonicity Analysis

| Metric | Direction | Spearman rho | Significance |
|--------|-----------|-------------|--------------|
| CAGR (%) | POSITIVE | 0.6485 | * (p=0.0425) |
| Sharpe | NEGATIVE | -0.6322 | * (p=0.0498) |
| Ann. Vol (%) | POSITIVE | 1.0000 | *** (p=0.0000) |
| Max Drawdown (%) | NEGATIVE | -0.9152 | *** (p=0.0002) |

### Interpretation

**Factor WORKS**: D10 outperforms D1 by 6.2% CAGR. Higher MAX score → higher returns.
**Risk-adjusted**: D1 Sharpe (0.990) > D10 Sharpe (0.850). Even if D10 has higher raw return, it is NOT more efficient per unit of risk.

---

## MOMENTUM_6M

### Decile Performance

| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |
|--------|----------|--------------|--------|---------|--------|------------|
| D1 | 14.41 | 22.28 | 0.450 | 0.600 | 0.230 | -62.61 |
| D2 | 15.02 | 19.64 | 0.510 | 0.670 | 0.270 | -56.42 |
| D3 | 18.00 | 18.10 | 0.680 | 0.850 | 0.310 | -58.47 |
| D4 | 19.64 | 17.66 | 0.770 | 0.970 | 0.370 | -53.47 |
| D5 | 23.58 | 17.01 | 0.990 | 1.240 | 0.550 | -42.85 |
| D6 | 21.63 | 17.32 | 0.880 | 1.070 | 0.480 | -45.08 |
| D7 | 22.29 | 17.41 | 0.910 | 1.090 | 0.570 | -39.02 |
| D8 | 28.36 | 18.01 | 1.150 | 1.390 | 0.740 | -38.47 |
| D9 | 33.07 | 19.33 | 1.280 | 1.480 | 0.790 | -42.00 |
| D10 | 40.04 | 21.61 | 1.400 | 1.650 | 1.130 | -35.37 |

### Factor Spread (D10 – D1)

| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |
|--------|----------|------------|-----------------|
| CAGR | 14.41% | 40.04% | 20.77% |
| Sharpe | 0.450 | 1.400 | — |
| Ann. Vol | 22.3% | 21.6% | — |
| Max DD | -62.6% | -35.4% | — |

### Monotonicity Analysis

| Metric | Direction | Spearman rho | Significance |
|--------|-----------|-------------|--------------|
| CAGR (%) | POSITIVE | 0.9636 | *** (p=0.0000) |
| Sharpe | POSITIVE | 0.9636 | *** (p=0.0000) |
| Ann. Vol (%) | NEGATIVE | -0.1394 | ns (p=0.7009) |
| Max Drawdown (%) | POSITIVE | 0.9394 | *** (p=0.0001) |

### Interpretation

**Factor WORKS**: D10 outperforms D1 by 20.8% CAGR. Higher MOMENTUM_6M score → higher returns.
**Risk-adjusted**: D10 Sharpe (1.400) > D1 Sharpe (0.450). D10 is more efficient on a risk-adjusted basis.

---

## MOMENTUM_12M

### Decile Performance

| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |
|--------|----------|--------------|--------|---------|--------|------------|
| D1 | 14.72 | 22.84 | 0.460 | 0.610 | 0.220 | -65.91 |
| D2 | 14.91 | 19.77 | 0.510 | 0.650 | 0.260 | -57.86 |
| D3 | 18.94 | 18.55 | 0.710 | 0.910 | 0.330 | -56.89 |
| D4 | 18.47 | 17.29 | 0.730 | 0.920 | 0.350 | -52.22 |
| D5 | 21.20 | 17.28 | 0.860 | 1.070 | 0.480 | -44.09 |
| D6 | 21.20 | 17.20 | 0.870 | 1.060 | 0.480 | -44.01 |
| D7 | 28.50 | 17.37 | 1.200 | 1.470 | 0.780 | -36.43 |
| D8 | 26.05 | 17.91 | 1.060 | 1.260 | 0.700 | -37.03 |
| D9 | 33.28 | 19.35 | 1.280 | 1.500 | 0.890 | -37.26 |
| D10 | 40.95 | 21.54 | 1.430 | 1.680 | 1.030 | -39.88 |

### Factor Spread (D10 – D1)

| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |
|--------|----------|------------|-----------------|
| CAGR | 14.72% | 40.95% | 20.93% |
| Sharpe | 0.460 | 1.430 | — |
| Ann. Vol | 22.8% | 21.5% | — |
| Max DD | -65.9% | -39.9% | — |

### Monotonicity Analysis

| Metric | Direction | Spearman rho | Significance |
|--------|-----------|-------------|--------------|
| CAGR (%) | POSITIVE | 0.9726 | *** (p=0.0000) |
| Sharpe | POSITIVE | 0.9879 | *** (p=0.0000) |
| Ann. Vol (%) | NEGATIVE | -0.1152 | ns (p=0.7514) |
| Max Drawdown (%) | POSITIVE | 0.8788 | *** (p=0.0008) |

### Interpretation

**Factor WORKS**: D10 outperforms D1 by 20.9% CAGR. Higher MOMENTUM_12M score → higher returns.
**Risk-adjusted**: D10 Sharpe (1.430) > D1 Sharpe (0.460). D10 is more efficient on a risk-adjusted basis.

---

## LOW_VOL

### Decile Performance

| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |
|--------|----------|--------------|--------|---------|--------|------------|
| D1 | 27.53 | 26.40 | 0.830 | 1.060 | 0.420 | -64.86 |
| D2 | 28.61 | 24.03 | 0.930 | 1.150 | 0.450 | -63.61 |
| D3 | 23.35 | 22.32 | 0.790 | 0.970 | 0.430 | -54.69 |
| D4 | 24.17 | 20.40 | 0.880 | 1.100 | 0.430 | -56.06 |
| D5 | 26.10 | 19.26 | 1.000 | 1.240 | 0.620 | -42.31 |
| D6 | 21.40 | 17.73 | 0.850 | 1.050 | 0.500 | -42.77 |
| D7 | 22.78 | 16.38 | 0.980 | 1.190 | 0.580 | -39.05 |
| D8 | 23.95 | 14.88 | 1.130 | 1.350 | 0.640 | -37.29 |
| D9 | 17.48 | 13.63 | 0.820 | 1.000 | 0.520 | -33.34 |
| D10 | 19.39 | 11.56 | 1.090 | 1.320 | 0.630 | -30.84 |

### Factor Spread (D10 – D1)

| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |
|--------|----------|------------|-----------------|
| CAGR | 27.53% | 19.39% | -10.72% |
| Sharpe | 0.830 | 1.090 | — |
| Ann. Vol | 26.4% | 11.6% | — |
| Max DD | -64.9% | -30.8% | — |

### Monotonicity Analysis

| Metric | Direction | Spearman rho | Significance |
|--------|-----------|-------------|--------------|
| CAGR (%) | NEGATIVE | -0.8182 | ** (p=0.0038) |
| Sharpe | POSITIVE | 0.4424 | ns (p=0.2004) |
| Ann. Vol (%) | NEGATIVE | -1.0000 | *** (p=0.0000) |
| Max Drawdown (%) | POSITIVE | 0.9758 | *** (p=0.0000) |

### Interpretation

**Factor REVERSED**: D10 underperforms D1 by 10.7% CAGR. Lower LOW_VOL score → higher returns. This is ANOMALOUS — consider that D10 may carry excess risk rather than being a genuine return driver.
**Risk-adjusted**: D10 Sharpe (1.090) > D1 Sharpe (0.830). D10 is more efficient on a risk-adjusted basis.

---

## QUALITY

### Decile Performance

| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |
|--------|----------|--------------|--------|---------|--------|------------|
| D1 | 0.49 | 32.51 | -0.000 | -0.010 | 0.020 | -30.94 |
| D2 | -5.50 | 31.10 | -0.210 | -0.300 | -0.140 | -39.32 |
| D3 | -12.98 | 26.60 | -0.610 | -0.930 | -0.380 | -33.97 |
| D4 | 13.24 | 23.14 | 0.400 | 0.590 | 0.720 | -18.45 |
| D5 | 27.49 | 34.62 | 0.700 | 1.050 | 1.070 | -25.64 |
| D6 | 14.33 | 24.81 | 0.430 | 0.640 | 0.800 | -18.01 |
| D7 | -18.65 | 33.63 | -0.620 | -0.770 | -0.330 | -55.98 |
| D8 | -6.25 | 22.25 | -0.440 | -0.660 | -0.220 | -27.99 |
| D9 | -5.80 | 19.33 | -0.510 | -0.720 | -0.310 | -18.93 |
| D10 | -11.12 | 33.67 | -0.350 | -0.440 | -0.210 | -53.08 |

### Factor Spread (D10 – D1)

| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |
|--------|----------|------------|-----------------|
| CAGR | 0.49% | -11.12% | -17.29% |
| Sharpe | -0.000 | -0.350 | — |
| Ann. Vol | 32.5% | 33.7% | — |
| Max DD | -30.9% | -53.1% | — |

### Monotonicity Analysis

| Metric | Direction | Spearman rho | Significance |
|--------|-----------|-------------|--------------|
| CAGR (%) | NEGATIVE | -0.3333 | ns (p=0.3466) |
| Sharpe | NEGATIVE | -0.3091 | ns (p=0.3848) |
| Ann. Vol (%) | NEGATIVE | -0.1394 | ns (p=0.7009) |
| Max Drawdown (%) | NEGATIVE | -0.0182 | ns (p=0.9602) |

### Interpretation

**Factor REVERSED**: D10 underperforms D1 by 17.3% CAGR. Lower QUALITY score → higher returns. This is ANOMALOUS — consider that D10 may carry excess risk rather than being a genuine return driver.
**Risk-adjusted**: D1 Sharpe (-0.000) > D10 Sharpe (-0.350). Even if D10 has higher raw return, it is NOT more efficient per unit of risk.

---

## VALUE

### Decile Performance

| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |
|--------|----------|--------------|--------|---------|--------|------------|
| D1 | 5.36 | 20.83 | 0.070 | 0.100 | 0.250 | -21.33 |
| D2 | -0.02 | 34.77 | 0.010 | 0.000 | -0.000 | -45.84 |
| D3 | -23.36 | 39.34 | -0.630 | -0.800 | -0.390 | -60.30 |
| D4 | -6.72 | 27.32 | -0.330 | -0.530 | -0.200 | -33.52 |
| D5 | 12.70 | 32.94 | 0.350 | 0.550 | 0.440 | -28.87 |
| D6 | 38.85 | 25.27 | 1.190 | 1.800 | 2.590 | -15.00 |
| D7 | -13.25 | 27.84 | -0.580 | -0.820 | -0.390 | -33.69 |
| D8 | 0.85 | 25.88 | -0.060 | -0.100 | 0.020 | -34.30 |
| D9 | -1.76 | 23.70 | -0.200 | -0.310 | -0.080 | -22.48 |
| D10 | -2.55 | 27.81 | -0.160 | -0.220 | -0.070 | -37.73 |

### Factor Spread (D10 – D1)

| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |
|--------|----------|------------|-----------------|
| CAGR | 5.36% | -2.55% | -9.56% |
| Sharpe | 0.070 | -0.160 | — |
| Ann. Vol | 20.8% | 27.8% | — |
| Max DD | -21.3% | -37.7% | — |

### Monotonicity Analysis

| Metric | Direction | Spearman rho | Significance |
|--------|-----------|-------------|--------------|
| CAGR (%) | NEGATIVE | -0.0909 | ns (p=0.8028) |
| Sharpe | NEGATIVE | -0.1515 | ns (p=0.6761) |
| Ann. Vol (%) | NEGATIVE | -0.2000 | ns (p=0.5796) |
| Max Drawdown (%) | POSITIVE | 0.0182 | ns (p=0.9602) |

### Interpretation

**Factor REVERSED**: D10 underperforms D1 by 9.6% CAGR. Lower VALUE score → higher returns. This is ANOMALOUS — consider that D10 may carry excess risk rather than being a genuine return driver.
**Risk-adjusted**: D1 Sharpe (0.070) > D10 Sharpe (-0.160). Even if D10 has higher raw return, it is NOT more efficient per unit of risk.

---

## Charts

Charts saved to `research/charts/`:

- `factor_comparison.png` — Side-by-side D1/D10 CAGR, Sharpe, and spread for all factors
- `equity_comparison.png` — D1 and D10 equity curves overlaid across all factors
- `sharpe_heatmap.png` — Sharpe ratio by factor × decile
- `max/dashboard.png` — 4-panel summary
- `max/decile_equity_curves.png`
- `max/decile_bars.png`
- `max/monotonicity.png`
- `max/factor_spread.png`
- `momentum_6m/dashboard.png` — 4-panel summary
- `momentum_6m/decile_equity_curves.png`
- `momentum_6m/decile_bars.png`
- `momentum_6m/monotonicity.png`
- `momentum_6m/factor_spread.png`
- `momentum_12m/dashboard.png` — 4-panel summary
- `momentum_12m/decile_equity_curves.png`
- `momentum_12m/decile_bars.png`
- `momentum_12m/monotonicity.png`
- `momentum_12m/factor_spread.png`
- `low_vol/dashboard.png` — 4-panel summary
- `low_vol/decile_equity_curves.png`
- `low_vol/decile_bars.png`
- `low_vol/monotonicity.png`
- `low_vol/factor_spread.png`
- `quality/dashboard.png` — 4-panel summary
- `quality/decile_equity_curves.png`
- `quality/decile_bars.png`
- `quality/monotonicity.png`
- `quality/factor_spread.png`
- `value/dashboard.png` — 4-panel summary
- `value/decile_equity_curves.png`
- `value/decile_bars.png`
- `value/monotonicity.png`
- `value/factor_spread.png`

---
*Generated by `research/run_research.py`. Not investment advice.*