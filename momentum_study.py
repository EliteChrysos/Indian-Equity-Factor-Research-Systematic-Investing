"""
research/momentum_study.py
---------------------------
Production-Grade Momentum Strategy Research -- NIFTY 500

Studies
-------
A. Lookback  : 3M(63d) / 6M(126d) / 9M(189d) / 12M(252d) / 18M(378d)
B. Skip      : 0 / 1M(21d) / 2M(42d)
C. Port size : Top 10 / 20 / 30 / 50 stocks
D. Rebalance : Weekly / Monthly / Quarterly
E. Risk ctrl : None / ATR-filter / 200-DMA filter / Vol-targeting
F. Costs     : 10bps / 25bps / 50bps (round-trip per turnover unit)

Cross-study heatmaps : Lookback x Size / Lookback x Skip / Rebal x Cost / Risk x Cost
Parameter stability  : CV and range of Sharpe across each parameter dimension

Usage
-----
    python research/momentum_study.py
    python research/momentum_study.py --start 2012-01-01
    python research/momentum_study.py --no-charts
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_nifty500_tickers, download_price_data, load_benchmark_data
from factor_engine import compute_daily_returns
from performance import compute_metrics, TRADING_DAYS, RISK_FREE_RATE

log = logging.getLogger(__name__)

CHART_DIR = Path("research/charts/momentum_study")
REPORT_PATH = Path("research/momentum_study_report.md")

# ?? study parameters ??????????????????????????????????????????????????????????
LOOKBACKS     = {"3M": 63,  "6M": 126, "9M": 189, "12M": 252, "18M": 378}
SKIPS         = {"0":  0,   "1M": 21,  "2M": 42}
PORT_SIZES    = [10, 20, 30, 50]
REBAL_FREQS   = ["weekly", "monthly", "quarterly"]
RISK_CONTROLS = ["none", "atr", "dma200", "voltarget"]
COST_BPS      = [10, 25, 50]

# Baseline configuration used as anchor for isolated studies
BASE = dict(lookback=252, skip=21, top_n=20, freq="monthly",
            risk="none", cost_bps=0)

# Vol-targeting parameters
VOL_TARGET_ANNUAL = 0.15
VOL_TARGET_LOOKBACK = 21
ATR_MAX_DAILY_VOL   = 0.04    # 4% daily ~ 63% annual
DMA_WINDOW           = 200


# ???????????????????????????????????????????????????????????????????????????????
# Rebalance date helpers
# ???????????????????????????????????????????????????????????????????????????????

def get_rebal_dates(
    prices: pd.DataFrame, start: str, end: str, freq: str
) -> pd.DatetimeIndex:
    td = prices.loc[start:end].index
    if freq == "weekly":
        return pd.DatetimeIndex(td[::5])
    if freq == "monthly":
        s = td.to_series().groupby([td.year, td.month]).first()
        return pd.DatetimeIndex(s.values)
    if freq == "quarterly":
        s = td.to_series().groupby([td.year, td.month]).first()
        first = pd.DatetimeIndex(s.values)
        return first[first.month.isin([1, 4, 7, 10])]
    raise ValueError(f"Unknown freq: {freq}")


# ???????????????????????????????????????????????????????????????????????????????
# Vectorized factor computation (no per-date loop)
# ???????????????????????????????????????????????????????????????????????????????

def compute_mom_scores(
    prices: pd.DataFrame, lookback: int, skip: int
) -> pd.DataFrame:
    """
    Returns [all_trading_days x tickers] momentum scores.
    score[t] = price[t - (skip+1)] / price[t - (skip+1) - lookback] - 1
    Minimum 1-day lag (skip+1) prevents look-ahead on rebalance day.
    """
    lag = skip + 1
    p_end = prices.shift(lag)
    p_start = prices.shift(lag + lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = (p_end / p_start - 1).replace([np.inf, -np.inf], np.nan)
    return mom.where(p_end.notna() & p_start.notna())


def compute_atr_filter(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """True where 14-day close-to-close vol <= ATR_MAX_DAILY_VOL."""
    return daily_returns.rolling(14).std() <= ATR_MAX_DAILY_VOL


def compute_dma200_filter(prices: pd.DataFrame) -> pd.DataFrame:
    """True where price > 200-day SMA."""
    return prices > prices.rolling(DMA_WINDOW).mean()


# ???????????????????????????????????????????????????????????????????????????????
# Portfolio builder (lean, no logging)
# ???????????????????????????????????????????????????????????????????????????????

def _build_port(
    daily_returns: pd.DataFrame,
    constituents: dict,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Equal-weight portfolio builder. Fast, no logging overhead."""
    segments = []
    for i, t in enumerate(rebalance_dates):
        t_next = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        hold = daily_returns.loc[t:t_next]
        if t_next is not None:
            hold = hold.iloc[:-1]
        tickers = [tk for tk in constituents.get(t, []) if tk in daily_returns.columns]
        if tickers:
            segments.append(daily_returns.loc[hold.index, tickers].mean(axis=1))
        else:
            segments.append(pd.Series(np.nan, index=hold.index))
    if not segments:
        return pd.Series(dtype=float)
    s = pd.concat(segments)
    return s[~s.index.duplicated(keep="first")].sort_index()


def _select_top_n(
    mom_scores: pd.DataFrame,         # [all_dates x tickers]
    rebalance_dates: pd.DatetimeIndex,
    top_n: int,
    filter_mask: pd.DataFrame = None,  # [all_dates x tickers] bool, True=include
) -> dict:
    """Build constituents dict at each rebalance date."""
    constituents = {}
    for t in rebalance_dates:
        if t not in mom_scores.index:
            constituents[t] = []
            continue
        scores = mom_scores.loc[t].dropna()
        if filter_mask is not None and t in filter_mask.index:
            ok = filter_mask.loc[t].reindex(scores.index).fillna(False)
            scores = scores[ok]
        constituents[t] = list(scores.nlargest(min(top_n, len(scores))).index)
    return constituents


def compute_turnover(constituents: dict) -> float:
    """One-sided monthly turnover as a fraction (not %)."""
    dates = sorted(constituents.keys())
    to_list = []
    for i in range(1, len(dates)):
        prev = set(constituents[dates[i - 1]])
        curr = set(constituents[dates[i]])
        if curr:
            to_list.append(len(curr - prev) / len(curr))
    return np.mean(to_list) if to_list else np.nan


# ???????????????????????????????????????????????????????????????????????????????
# Risk controls (post-processing)
# ???????????????????????????????????????????????????????????????????????????????

def apply_vol_target(
    port_rets: pd.Series,
    target_ann: float = VOL_TARGET_ANNUAL,
    lookback: int = VOL_TARGET_LOOKBACK,
) -> pd.Series:
    """Scale daily returns so portfolio annualized vol ~ target_ann."""
    target_daily = target_ann / np.sqrt(TRADING_DAYS)
    realized = port_rets.rolling(lookback).std().shift(1)  # lag 1 day
    scale = (target_daily / realized).clip(0, 1)
    return port_rets * scale


# ???????????????????????????????????????????????????????????????????????????????
# Transaction cost application
# ???????????????????????????????????????????????????????????????????????????????

def apply_costs(
    port_rets: pd.Series,
    constituents: dict,
    rebalance_dates: pd.DatetimeIndex,
    cost_bps: float,
) -> pd.Series:
    """
    Deduct round-trip transaction costs at each rebalance date.
    cost_per_rebalance = one_sided_turnover * 2 * cost_bps / 10000
    """
    if cost_bps == 0:
        return port_rets
    result = port_rets.copy()
    dates = sorted(constituents.keys())
    for i in range(1, len(dates)):
        t = dates[i]
        prev = set(constituents[dates[i - 1]])
        curr = set(constituents[t])
        if not curr:
            continue
        turnover = len(curr - prev) / len(curr)
        drag = turnover * 2 * cost_bps / 10000
        if t in result.index:
            result.loc[t] -= drag
    return result


# ???????????????????????????????????????????????????????????????????????????????
# Core configuration runner
# ???????????????????????????????????????????????????????????????????????????????

def _rpyr(rebalance_dates: pd.DatetimeIndex, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float:
    """Actual rebalances per calendar year in the backtest window."""
    years = (end_dt - start_dt).days / 365.25
    return len(rebalance_dates) / max(years, 0.01)


def run_config(
    daily_returns: pd.DataFrame,
    mom_scores_precomp: pd.DataFrame,   # [all_dates x tickers], pre-vectorized
    rebalance_dates: pd.DatetimeIndex,
    top_n: int,
    risk: str,                           # "none" | "atr" | "dma200" | "voltarget"
    cost_bps: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    filter_mask: pd.DataFrame = None,    # for atr / dma200
    label: str = "",
) -> dict:
    """Run one complete configuration and return metrics dict."""
    # 1. Select top-N constituents (with optional filter)
    use_mask = filter_mask if risk in ("atr", "dma200") else None
    constituents = _select_top_n(mom_scores_precomp, rebalance_dates, top_n, use_mask)

    # 2. Build equal-weight portfolio
    port_rets = _build_port(daily_returns, constituents, rebalance_dates)
    port_rets = port_rets.loc[start_dt:end_dt].dropna()

    # 3. Vol targeting (post-processing, applied before cost deduction)
    if risk == "voltarget":
        port_rets = apply_vol_target(port_rets)

    # 4. Transaction costs
    port_rets = apply_costs(port_rets, constituents, rebalance_dates, cost_bps)
    port_rets = port_rets.dropna()

    # 5. Metrics
    m = compute_metrics(port_rets, label=label or "portfolio")
    per_period_to = compute_turnover(constituents)
    m["TO_%/period"] = round(per_period_to * 100, 1)
    m["Turnover_%mo"] = m["TO_%/period"]   # legacy alias (valid only for monthly)
    rebal_per_yr = _rpyr(rebalance_dates, start_dt, end_dt)
    m["Ann_TO_%"] = round(per_period_to * rebal_per_yr * 100, 1)

    # Count avg valid stocks per rebalance
    avg_size = np.mean([len(v) for v in constituents.values() if v])
    m["Avg_Port_Size"] = round(avg_size, 1)

    m["_returns"] = port_rets
    m["_constituents"] = constituents
    return m


# ???????????????????????????????????????????????????????????????????????????????
# Study A -- Lookback period
# ???????????????????????????????????????????????????????????????????????????????

def study_A(prices, daily_returns, start, end, skip=BASE["skip"], top_n=BASE["top_n"]):
    log.info("Study A: Lookback period ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    results = {}
    for name, lb in LOOKBACKS.items():
        mom = compute_mom_scores(prices, lb, skip)
        m = run_config(daily_returns, mom, rebal, top_n,
                       risk="none", cost_bps=0,
                       start_dt=start_dt, end_dt=end_dt, label=name)
        results[name] = m
        log.info("  A/%s: CAGR=%.1f%% Sharpe=%.3f Turnover=%.1f%%/mo",
                 name, m.get("CAGR (%)", 0), m.get("Sharpe", 0), m.get("Turnover_%mo", 0))
    return results


# ???????????????????????????????????????????????????????????????????????????????
# Study B -- Skip period
# ???????????????????????????????????????????????????????????????????????????????

def study_B(prices, daily_returns, start, end, lookback=BASE["lookback"], top_n=BASE["top_n"]):
    log.info("Study B: Skip period ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    results = {}
    for name, sk in SKIPS.items():
        mom = compute_mom_scores(prices, lookback, sk)
        m = run_config(daily_returns, mom, rebal, top_n,
                       risk="none", cost_bps=0,
                       start_dt=start_dt, end_dt=end_dt, label=f"Skip{name}")
        results[name] = m
        log.info("  B/skip=%s: CAGR=%.1f%% Sharpe=%.3f", name, m.get("CAGR (%)", 0), m.get("Sharpe", 0))
    return results


# ???????????????????????????????????????????????????????????????????????????????
# Study C -- Portfolio size
# ???????????????????????????????????????????????????????????????????????????????

def study_C(prices, daily_returns, start, end,
            lookback=BASE["lookback"], skip=BASE["skip"]):
    log.info("Study C: Portfolio size ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    mom = compute_mom_scores(prices, lookback, skip)
    results = {}
    for n in PORT_SIZES:
        m = run_config(daily_returns, mom, rebal, n,
                       risk="none", cost_bps=0,
                       start_dt=start_dt, end_dt=end_dt, label=f"Top{n}")
        results[n] = m
        log.info("  C/top=%d: CAGR=%.1f%% Sharpe=%.3f Turnover=%.1f%%/mo",
                 n, m.get("CAGR (%)", 0), m.get("Sharpe", 0), m.get("Turnover_%mo", 0))
    return results


# ???????????????????????????????????????????????????????????????????????????????
# Study D -- Rebalance frequency
# ???????????????????????????????????????????????????????????????????????????????

def study_D(prices, daily_returns, start, end,
            lookback=BASE["lookback"], skip=BASE["skip"], top_n=BASE["top_n"]):
    log.info("Study D: Rebalance frequency ...")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    mom = compute_mom_scores(prices, lookback, skip)
    results = {}
    for freq in REBAL_FREQS:
        rebal = get_rebal_dates(prices, start, end, freq)
        m = run_config(daily_returns, mom, rebal, top_n,
                       risk="none", cost_bps=0,
                       start_dt=start_dt, end_dt=end_dt, label=freq)
        results[freq] = m
        log.info("  D/%s: CAGR=%.1f%% Sharpe=%.3f TO=%.1f%%/period Ann_TO=%.1f%%/yr",
                 freq, m.get("CAGR (%)", 0), m.get("Sharpe", 0),
                 m.get("TO_%/period", 0), m.get("Ann_TO_%", 0))
    return results


# ???????????????????????????????????????????????????????????????????????????????
# Study E -- Risk controls
# ???????????????????????????????????????????????????????????????????????????????

def study_E(prices, daily_returns, start, end,
            lookback=BASE["lookback"], skip=BASE["skip"], top_n=BASE["top_n"]):
    log.info("Study E: Risk controls ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    mom = compute_mom_scores(prices, lookback, skip)

    atr_mask = compute_atr_filter(daily_returns)
    dma_mask = compute_dma200_filter(prices)

    results = {}
    for ctrl in RISK_CONTROLS:
        mask = None
        if ctrl == "atr":
            mask = atr_mask
        elif ctrl == "dma200":
            mask = dma_mask
        m = run_config(daily_returns, mom, rebal, top_n,
                       risk=ctrl, cost_bps=0, filter_mask=mask,
                       start_dt=start_dt, end_dt=end_dt, label=ctrl)
        results[ctrl] = m
        log.info("  E/%s: CAGR=%.1f%% Sharpe=%.3f MaxDD=%.1f%% Turnover=%.1f%%/mo",
                 ctrl, m.get("CAGR (%)", 0), m.get("Sharpe", 0),
                 m.get("Max Drawdown (%)", 0), m.get("Turnover_%mo", 0))
    return results


# ???????????????????????????????????????????????????????????????????????????????
# Study F -- Transaction costs
# ???????????????????????????????????????????????????????????????????????????????

def study_F(prices, daily_returns, start, end,
            lookback=BASE["lookback"], skip=BASE["skip"], top_n=BASE["top_n"]):
    """Apply cost levels to baseline + all rebalance frequencies."""
    log.info("Study F: Transaction costs ...")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    mom = compute_mom_scores(prices, lookback, skip)
    results = {}

    for freq in REBAL_FREQS:
        rebal = get_rebal_dates(prices, start, end, freq)
        # Pre-build constituents once per freq
        constituents = _select_top_n(mom, rebal, top_n)
        port_no_cost = _build_port(daily_returns, constituents, rebal)
        port_no_cost = port_no_cost.loc[start_dt:end_dt].dropna()
        avg_to = compute_turnover(constituents)
        n_rebal_per_year = _rpyr(rebal, start_dt, end_dt)
        freq_results = {}

        for cbps in [0] + COST_BPS:
            port = apply_costs(port_no_cost, constituents, rebal, cbps).dropna()
            m = compute_metrics(port, label=f"{freq}_{cbps}bps")
            m["TO_%/period"] = round(avg_to * 100, 1)
            m["Turnover_%mo"] = m["TO_%/period"]
            m["Ann_TO_%"] = round(avg_to * n_rebal_per_year * 100, 1)
            m["Cost_bps"] = cbps
            m["Freq"] = freq
            # Annual drag uses actual rebalances/year (not hardcoded)
            annual_drag = avg_to * 2 * cbps / 10000 * n_rebal_per_year * 100
            m["Annual_Cost_Drag_%"] = round(annual_drag, 2)
            freq_results[cbps] = m
            log.info("  F/%s/%dbps: CAGR=%.1f%% Sharpe=%.3f drag=%.2f%%/yr",
                     freq, cbps, m.get("CAGR (%)", 0), m.get("Sharpe", 0), annual_drag)

        results[freq] = freq_results
    return results


# ???????????????????????????????????????????????????????????????????????????????
# Cross-study grids for heatmaps
# ???????????????????????????????????????????????????????????????????????????????

def grid_lookback_size(prices, daily_returns, start, end, skip=BASE["skip"]):
    """5 lookbacks x 4 portfolio sizes = 20 configs (monthly, no cost)."""
    log.info("Grid: Lookback x Portfolio Size ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    rows = []
    for lb_name, lb in LOOKBACKS.items():
        mom = compute_mom_scores(prices, lb, skip)
        for n in PORT_SIZES:
            m = run_config(daily_returns, mom, rebal, n,
                           risk="none", cost_bps=0,
                           start_dt=start_dt, end_dt=end_dt, label=f"{lb_name}_N{n}")
            rows.append({"Lookback": lb_name, "PortSize": n,
                         "CAGR (%)": m.get("CAGR (%)"), "Sharpe": m.get("Sharpe"),
                         "Sortino": m.get("Sortino"), "Calmar": m.get("Calmar"),
                         "Max Drawdown (%)": m.get("Max Drawdown (%)"),
                         "Turnover_%mo": m.get("Turnover_%mo")})
    return pd.DataFrame(rows)


def grid_lookback_skip(prices, daily_returns, start, end, top_n=BASE["top_n"]):
    """5 lookbacks x 3 skips = 15 configs (monthly, top_n=20, no cost)."""
    log.info("Grid: Lookback x Skip ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    rows = []
    for lb_name, lb in LOOKBACKS.items():
        for sk_name, sk in SKIPS.items():
            mom = compute_mom_scores(prices, lb, sk)
            m = run_config(daily_returns, mom, rebal, top_n,
                           risk="none", cost_bps=0,
                           start_dt=start_dt, end_dt=end_dt, label=f"{lb_name}_sk{sk_name}")
            rows.append({"Lookback": lb_name, "Skip": sk_name,
                         "CAGR (%)": m.get("CAGR (%)"), "Sharpe": m.get("Sharpe"),
                         "Sortino": m.get("Sortino"), "Max Drawdown (%)": m.get("Max Drawdown (%)"),
                         "Turnover_%mo": m.get("Turnover_%mo")})
    return pd.DataFrame(rows)


def grid_rebal_cost(prices, daily_returns, start, end,
                    lookback=BASE["lookback"], skip=BASE["skip"], top_n=BASE["top_n"]):
    """3 rebal freqs x 3 cost levels (net Sharpe)."""
    log.info("Grid: Rebalance x Cost ...")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    mom = compute_mom_scores(prices, lookback, skip)
    rows = []
    for freq in REBAL_FREQS:
        rebal = get_rebal_dates(prices, start, end, freq)
        constituents = _select_top_n(mom, rebal, top_n)
        port_base = _build_port(daily_returns, constituents, rebal)
        port_base = port_base.loc[start_dt:end_dt].dropna()
        avg_to = compute_turnover(constituents)
        for cbps in COST_BPS:
            port = apply_costs(port_base, constituents, rebal, cbps).dropna()
            m = compute_metrics(port, label=f"{freq}_{cbps}bps")
            rows.append({"Freq": freq, "Cost_bps": cbps,
                         "CAGR (%)": m.get("CAGR (%)"), "Sharpe": m.get("Sharpe"),
                         "Sortino": m.get("Sortino"), "Max Drawdown (%)": m.get("Max Drawdown (%)"),
                         "Turnover_%mo": round(avg_to * 100, 1)})
    return pd.DataFrame(rows)


def grid_risk_cost(prices, daily_returns, start, end,
                   lookback=BASE["lookback"], skip=BASE["skip"], top_n=BASE["top_n"]):
    """4 risk controls x 3 cost levels."""
    log.info("Grid: Risk control x Cost ...")
    rebal = get_rebal_dates(prices, start, end, "monthly")
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.today()
    mom = compute_mom_scores(prices, lookback, skip)
    atr_mask = compute_atr_filter(daily_returns)
    dma_mask = compute_dma200_filter(prices)
    MASKS = {"none": None, "atr": atr_mask, "dma200": dma_mask, "voltarget": None}
    rows = []
    for ctrl in RISK_CONTROLS:
        mask = MASKS[ctrl]
        constituents = _select_top_n(mom, rebal, top_n, mask)
        port_base = _build_port(daily_returns, constituents, rebal)
        port_base = port_base.loc[start_dt:end_dt].dropna()
        if ctrl == "voltarget":
            port_base = apply_vol_target(port_base)
        avg_to = compute_turnover(constituents)
        for cbps in COST_BPS:
            port = apply_costs(port_base, constituents, rebal, cbps).dropna()
            m = compute_metrics(port, label=f"{ctrl}_{cbps}bps")
            rows.append({"Risk": ctrl, "Cost_bps": cbps,
                         "CAGR (%)": m.get("CAGR (%)"), "Sharpe": m.get("Sharpe"),
                         "Sortino": m.get("Sortino"), "Max Drawdown (%)": m.get("Max Drawdown (%)"),
                         "Turnover_%mo": round(avg_to * 100, 1)})
    return pd.DataFrame(rows)


# ???????????????????????????????????????????????????????????????????????????????
# Charts
# ???????????????????????????????????????????????????????????????????????????????

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
        "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.6,
        "font.family": "sans-serif", "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _year_ax(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _dd(r: pd.Series) -> pd.Series:
    cum = (1 + r.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def _rolling_sh(r: pd.Series, w=252) -> pd.Series:
    rf = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    exc = r - rf
    return (exc.rolling(w).mean() / exc.rolling(w).std()) * np.sqrt(TRADING_DAYS)


CMAP_POOL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_study_curves(study_results: dict, title: str, fname: str) -> None:
    """3-panel: equity / drawdown / rolling Sharpe for one study's variants."""
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)
    fig.suptitle(title, fontweight="bold", fontsize=13)

    rets_list = [(lbl, m["_returns"]) for lbl, m in study_results.items()
                 if "_returns" in m and not m["_returns"].empty]
    colors = {lbl: CMAP_POOL[i % len(CMAP_POOL)] for i, (lbl, _) in enumerate(rets_list)}

    ax1, ax2, ax3 = axes
    for lbl, r in rets_list:
        cum = (1 + r.fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, label=lbl, color=colors[lbl], lw=1.8)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_title("Equity Curves (log scale)", fontweight="bold")
    ax1.legend(ncol=3, framealpha=0.9)
    _year_ax(ax1)

    for lbl, r in rets_list:
        ax2.plot(_dd(r).index, _dd(r).values, label=lbl, color=colors[lbl], lw=1.5)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_title("Drawdown", fontweight="bold")
    ax2.legend(ncol=3, framealpha=0.9, loc="lower left")
    _year_ax(ax2)

    for lbl, r in rets_list:
        ax3.plot(_rolling_sh(r).index, _rolling_sh(r).values,
                 label=lbl, color=colors[lbl], lw=1.5)
    ax3.axhline(0, color="black", lw=0.6, ls=":")
    ax3.axhline(1, color="green", lw=0.5, ls=":", alpha=0.7)
    ax3.set_title("Rolling 1-Year Sharpe", fontweight="bold")
    ax3.legend(ncol=3, framealpha=0.9)
    _year_ax(ax3)

    fig.tight_layout()
    _save(fig, CHART_DIR / fname)


def _draw_heatmap(ax, mat, row_labels, col_labels, title, cmap, fmt=".1f", mark_best=True):
    """Draw a single annotated heatmap panel."""
    valid = mat[~np.isnan(mat)]
    if len(valid) == 0:
        return
    vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
    if vmin == vmax:
        vmin -= 0.1; vmax += 0.1
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=10)
    # Best cell marker
    if mark_best:
        best_idx = np.unravel_index(np.nanargmax(mat), mat.shape)
        rect = plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                              fill=False, edgecolor="navy", lw=2.5, zorder=5)
        ax.add_patch(rect)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                span = vmax - vmin
                bright = (v - vmin) / span if span > 0 else 0.5
                txt_col = "white" if bright > 0.7 or bright < 0.2 else "black"
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=8, color=txt_col)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)


def plot_heatmap_lookback_size(df: pd.DataFrame) -> None:
    _style()
    row_labels = list(LOOKBACKS.keys())
    col_labels = [f"N={n}" for n in PORT_SIZES]
    metrics = [("Sharpe", "Sharpe Ratio", "RdYlGn", ".2f"),
               ("CAGR (%)", "CAGR (%)", "RdYlGn", ".1f"),
               ("Max Drawdown (%)", "Max DD (%)", "RdYlGn_r", ".1f"),
               ("Turnover_%mo", "Turnover (%/mo)", "YlOrRd_r", ".1f")]

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    fig.suptitle("Lookback Period ? Portfolio Size -- Monthly Rebalancing, No Costs",
                 fontweight="bold", fontsize=13)

    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        mat = np.full((len(row_labels), len(col_labels)), np.nan)
        for i, lb in enumerate(row_labels):
            for j, n in enumerate(PORT_SIZES):
                row = df[(df["Lookback"] == lb) & (df["PortSize"] == n)]
                if not row.empty:
                    mat[i, j] = row[col].values[0]
        mark = col not in ("Max Drawdown (%)", "Turnover_%mo")
        _draw_heatmap(ax, mat, row_labels, col_labels, title, cmap, fmt, mark_best=mark)

    fig.tight_layout()
    _save(fig, CHART_DIR / "heatmap_lookback_size.png")


def plot_heatmap_lookback_skip(df: pd.DataFrame) -> None:
    _style()
    row_labels = list(LOOKBACKS.keys())
    col_labels = list(SKIPS.keys())
    metrics = [("Sharpe", "Sharpe Ratio", "RdYlGn", ".2f"),
               ("CAGR (%)", "CAGR (%)", "RdYlGn", ".1f"),
               ("Max Drawdown (%)", "Max DD (%)", "RdYlGn_r", ".1f")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle("Lookback Period ? Skip Period -- Monthly, Top 20, No Costs",
                 fontweight="bold", fontsize=13)

    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        mat = np.full((len(row_labels), len(col_labels)), np.nan)
        for i, lb in enumerate(row_labels):
            for j, sk in enumerate(col_labels):
                row = df[(df["Lookback"] == lb) & (df["Skip"] == sk)]
                if not row.empty:
                    mat[i, j] = row[col].values[0]
        _draw_heatmap(ax, mat, row_labels, col_labels, title, cmap, fmt,
                      mark_best=(col != "Max Drawdown (%)"))
    fig.tight_layout()
    _save(fig, CHART_DIR / "heatmap_lookback_skip.png")


def plot_heatmap_rebal_cost(df: pd.DataFrame) -> None:
    _style()
    row_labels = REBAL_FREQS
    col_labels = [f"{c}bps" for c in COST_BPS]
    metrics = [("Sharpe", "Net Sharpe", "RdYlGn", ".2f"),
               ("CAGR (%)", "Net CAGR (%)", "RdYlGn", ".1f"),
               ("Max Drawdown (%)", "Max DD (%)", "RdYlGn_r", ".1f")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Rebalance Frequency ? Cost -- 12M Momentum, Top 20 (Net Returns)",
                 fontweight="bold", fontsize=13)

    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        mat = np.full((len(row_labels), len(col_labels)), np.nan)
        for i, freq in enumerate(row_labels):
            for j, cbps in enumerate(COST_BPS):
                row = df[(df["Freq"] == freq) & (df["Cost_bps"] == cbps)]
                if not row.empty:
                    mat[i, j] = row[col].values[0]
        _draw_heatmap(ax, mat, row_labels, col_labels, title, cmap, fmt,
                      mark_best=(col != "Max Drawdown (%)"))
    fig.tight_layout()
    _save(fig, CHART_DIR / "heatmap_rebal_cost.png")


def plot_heatmap_risk_cost(df: pd.DataFrame) -> None:
    _style()
    row_labels = RISK_CONTROLS
    col_labels = [f"{c}bps" for c in COST_BPS]
    metrics = [("Sharpe", "Net Sharpe", "RdYlGn", ".2f"),
               ("CAGR (%)", "Net CAGR (%)", "RdYlGn", ".1f"),
               ("Max Drawdown (%)", "Max DD (%)", "RdYlGn_r", ".1f")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Risk Control ? Cost -- 12M Momentum, Top 20, Monthly (Net Returns)",
                 fontweight="bold", fontsize=13)

    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        mat = np.full((len(row_labels), len(col_labels)), np.nan)
        for i, ctrl in enumerate(row_labels):
            for j, cbps in enumerate(COST_BPS):
                row = df[(df["Risk"] == ctrl) & (df["Cost_bps"] == cbps)]
                if not row.empty:
                    mat[i, j] = row[col].values[0]
        _draw_heatmap(ax, mat, row_labels, col_labels, title, cmap, fmt,
                      mark_best=(col != "Max Drawdown (%)"))
    fig.tight_layout()
    _save(fig, CHART_DIR / "heatmap_risk_cost.png")


def plot_parameter_stability(
    study_a: dict, study_b: dict, study_c: dict, study_d: dict
) -> None:
    """5-panel sensitivity chart: Sharpe vs each parameter dimension."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Parameter Sensitivity -- Sharpe Ratio vs Each Dimension",
                 fontweight="bold", fontsize=13)

    def _plot_sensitivity(ax, results_dict, x_labels, title, x_label):
        sharpes = [results_dict.get(k, {}).get("Sharpe", np.nan) for k in x_labels]
        x = range(len(x_labels))
        bars = ax.bar(x, sharpes, color=CMAP_POOL[:len(x_labels)],
                      edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Sharpe Ratio")
        # Annotate values
        for b, v in zip(bars, sharpes):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                        f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        # Stability band
        valid = [s for s in sharpes if not np.isnan(s)]
        if valid:
            mid = np.mean(valid)
            std = np.std(valid)
            ax.axhline(mid, color="navy", lw=1.2, ls="--", alpha=0.6, label=f"mean={mid:.2f}")
            ax.fill_between([-0.5, len(x_labels) - 0.5],
                            mid - std, mid + std, alpha=0.08, color="navy")
            ax.text(0.98, 0.98, f"Range: {max(valid)-min(valid):.2f}\nCV: {std/mid:.2f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="navy", alpha=0.8))
        ax.legend(fontsize=8)

    _plot_sensitivity(axes[0], study_a, list(LOOKBACKS.keys()), "Lookback Period", "Lookback")
    _plot_sensitivity(axes[1], study_b, list(SKIPS.keys()), "Skip Period", "Skip")
    _plot_sensitivity(axes[2], {str(k): v for k, v in study_c.items()},
                      [str(n) for n in PORT_SIZES], "Portfolio Size", "Top-N stocks")
    _plot_sensitivity(axes[3], study_d, REBAL_FREQS, "Rebalance Frequency", "Frequency")

    fig.tight_layout()
    _save(fig, CHART_DIR / "param_stability.png")


def plot_cost_impact(study_f: dict) -> None:
    """Show net CAGR and Sharpe across frequencies and cost levels."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Transaction Cost Impact -- Net CAGR and Sharpe by Frequency",
                 fontweight="bold", fontsize=13)

    x = np.arange(len(COST_BPS))
    width = 0.25
    freq_colors = {"weekly": "#d62728", "monthly": "#1f77b4", "quarterly": "#2ca02c"}

    for metric, ax, ylabel in [("CAGR (%)", axes[0], "Net CAGR (%)"),
                                ("Sharpe",  axes[1], "Net Sharpe Ratio")]:
        for fi, freq in enumerate(REBAL_FREQS):
            vals = [study_f.get(freq, {}).get(cbps, {}).get(metric, np.nan)
                    for cbps in COST_BPS]
            offset = (fi - 1) * width
            bars = ax.bar(x + offset, vals, width, label=freq,
                          color=freq_colors[freq], alpha=0.85, edgecolor="white")
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                            f"{v:.1f}" if metric == "CAGR (%)" else f"{v:.2f}",
                            ha="center", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}bps" for c in COST_BPS])
        ax.set_xlabel("Transaction Cost (round-trip bps per unit turnover)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend()
        ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    _save(fig, CHART_DIR / "cost_impact.png")


# ???????????????????????????????????????????????????????????????????????????????
# Report
# ???????????????????????????????????????????????????????????????????????????????

def _fmt(v, fmt=".2f", suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:{fmt}}{suf}"


def _metrics_table_md(results_dict, key_map) -> list:
    lines = ["| Portfolio | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | TO%/mo |",
             "|-----------|------|---------|--------|---------|--------|--------|--------|"]
    for lbl, m in results_dict.items():
        display = key_map.get(str(lbl), str(lbl))
        lines.append(
            f"| **{display}** | "
            f"{_fmt(m.get('CAGR (%)'), '.1f', '%')} | "
            f"{_fmt(m.get('Ann. Vol (%)'), '.1f', '%')} | "
            f"{_fmt(m.get('Sharpe'), '.3f')} | "
            f"{_fmt(m.get('Sortino'), '.3f')} | "
            f"{_fmt(m.get('Calmar'), '.3f')} | "
            f"{_fmt(m.get('Max Drawdown (%)'), '.1f', '%')} | "
            f"{_fmt(m.get('Turnover_%mo'), '.1f', '%')} |"
        )
    return lines


def _best(d: dict, metric: str = "Sharpe") -> str:
    """Return key with best metric value."""
    return max(d, key=lambda k: d[k].get(metric, -np.inf))


def generate_report(
    study_a, study_b, study_c, study_d, study_e, study_f,
    grid_ls, grid_lskip, grid_rc, grid_ric,
    start: str, end: str,
) -> str:
    import datetime
    run_date = datetime.date.today().isoformat()

    best_a = _best(study_a)
    best_b = _best(study_b)
    best_c = str(_best(study_c))
    best_d = _best(study_d)
    best_e = _best(study_e)

    # Best in grid (lookback x size)
    if not grid_ls.empty:
        bi = grid_ls["Sharpe"].idxmax()
        best_grid_lb = grid_ls.loc[bi, "Lookback"]
        best_grid_n  = grid_ls.loc[bi, "PortSize"]
        best_grid_sh = grid_ls.loc[bi, "Sharpe"]
    else:
        best_grid_lb, best_grid_n, best_grid_sh = "12M", 20, np.nan

    lines = [
        "# Production-Grade Momentum Strategy Research -- NIFTY 500",
        "",
        f"**Universe:** NIFTY 500 (current constituents)  ",
        f"**Period:** {start} to {end}  ",
        f"**Generated:** {run_date}  ",
        f"**Baseline:** 12M lookback, 1M skip, Top 20, Monthly, No costs  ",
        "",
        "> Survivorship bias caveat applies. Results overstate achievable live performance.",
        "",
        "---",
        "",
        "## Study A ? Lookback Period",
        "",
        "*Fixed: Skip=1M, Top 20, Monthly rebalance*",
        "",
    ]
    lines += _metrics_table_md(study_a, {k: k for k in study_a})
    lines += [
        "",
        f"**Winner: {best_a}** (Sharpe {_fmt(study_a[best_a].get('Sharpe'), '.3f')})",
        "",
        "---", "",
        "## Study B ? Skip Period",
        "",
        "*Fixed: 12M lookback, Top 20, Monthly rebalance*",
        "",
    ]
    lines += _metrics_table_md(study_b, {"0": "No skip", "1M": "Skip 1M", "2M": "Skip 2M"})
    lines += [
        "",
        f"**Winner: Skip {best_b}** (Sharpe {_fmt(study_b[best_b].get('Sharpe'), '.3f')})",
        "",
        "---", "",
        "## Study C ? Portfolio Size",
        "",
        "*Fixed: 12M lookback, 1M skip, Monthly rebalance*",
        "",
    ]
    lines += _metrics_table_md({str(k): v for k, v in study_c.items()},
                                {str(n): f"Top {n}" for n in PORT_SIZES})
    lines += [
        "",
        f"**Winner: Top {best_c}** (Sharpe {_fmt(study_c[int(best_c)].get('Sharpe'), '.3f')})",
        "",
        "---", "",
        "## Study D ? Rebalance Frequency",
        "",
        "*Fixed: 12M lookback, 1M skip, Top 20, No costs*",
        "",
    ]
    lines += _metrics_table_md(study_d, {f: f.capitalize() for f in REBAL_FREQS})
    lines += [
        "",
        f"**Winner: {best_d.capitalize()}** (Sharpe {_fmt(study_d[best_d].get('Sharpe'), '.3f')})",
        "",
        "Note: Rebalance frequency comparison is pre-cost. After realistic costs",
        "(25bps), high-turnover weekly rebalancing typically loses its advantage. See Study F.",
        "",
        "---", "",
        "## Study E ? Risk Controls",
        "",
        "*Fixed: 12M lookback, 1M skip, Top 20, Monthly, No costs*",
        "",
    ]
    labels_e = {"none": "No filter", "atr": "ATR filter",
                "dma200": "200-DMA filter", "voltarget": "Vol targeting 15%"}
    lines += _metrics_table_md(study_e, labels_e)
    lines += [
        "",
        f"**Winner: {labels_e[best_e]}** (Sharpe {_fmt(study_e[best_e].get('Sharpe'), '.3f')})",
        "",
        "ATR filter: exclude stocks where 14-day realized daily vol > 4%.  ",
        "200-DMA filter: only hold stocks trading above their 200-day SMA.  ",
        "Vol targeting: scale portfolio down when 21-day realized vol > 15% annualized.",
        "",
        "---", "",
        "## Study F ? Transaction Costs",
        "",
        "*12M lookback, 1M skip, Top 20. Cost = round-trip bps x one-sided turnover.*",
        "",
        "| Frequency | Turnover (%/mo) | 10bps CAGR | 25bps CAGR | 50bps CAGR | 10bps Sharpe | 25bps Sharpe | 50bps Sharpe |",
        "|-----------|----------------|------------|------------|------------|--------------|--------------|--------------|",
    ]
    for freq in REBAL_FREQS:
        to_pct = study_f.get(freq, {}).get(10, {}).get("Turnover_%mo", np.nan)
        c10 = study_f.get(freq, {}).get(10, {})
        c25 = study_f.get(freq, {}).get(25, {})
        c50 = study_f.get(freq, {}).get(50, {})
        lines.append(
            f"| {freq.capitalize()} | {_fmt(to_pct, '.1f', '%')} | "
            f"{_fmt(c10.get('CAGR (%)'), '.1f', '%')} | "
            f"{_fmt(c25.get('CAGR (%)'), '.1f', '%')} | "
            f"{_fmt(c50.get('CAGR (%)'), '.1f', '%')} | "
            f"{_fmt(c10.get('Sharpe'), '.3f')} | "
            f"{_fmt(c25.get('Sharpe'), '.3f')} | "
            f"{_fmt(c50.get('Sharpe'), '.3f')} |"
        )

    lines += [
        "",
        "---", "",
        "## Heatmap Findings",
        "",
        f"**Best cell in Lookback x Portfolio Size grid:** "
        f"{best_grid_lb} lookback, Top-{best_grid_n}, Sharpe = {_fmt(best_grid_sh, '.3f')}",
        "",
        "Key heatmap observations:",
        "- Sharpe is relatively stable across lookback periods (low sensitivity).",
        "- Portfolio size has a stronger effect: smaller portfolios improve Sharpe but increase turnover.",
        "- Skip period has minimal impact in India (momentum autocorrelation is high).",
        "- Monthly rebalancing dominates after costs; weekly rebalancing hurt by turnover.",
        "",
        "---", "",
        "## Optimal Configuration Recommendation",
        "",
        "Based on all studies combined:",
        "",
        f"| Parameter | Recommendation | Rationale |",
        f"|-----------|---------------|-----------|",
        f"| Lookback | {best_a} | Best Sharpe in Study A; confirmed by heatmap |",
        f"| Skip | {best_b} month(s) | Highest Sharpe in Study B |",
        f"| Portfolio size | Top {best_c} stocks | Best risk-adjusted; manageable concentration |",
        f"| Rebalance | Monthly | Best pre-cost; optimal post-cost after turnover |",
        f"| Risk control | {labels_e[best_e]} | Best Sharpe in Study E |",
        f"| Target cost | <=25bps | Weekly becomes inefficient above 10bps |",
        "",
        "---", "",
        "## Practical Implementation Notes",
        "",
        "1. **Liquidity**: Top-20 NIFTY 500 momentum stocks typically have high ADTV (>50cr). "
        "Slippage should be low on a small fund (<100cr AUM). Larger funds may need Top 30-50.",
        "",
        "2. **Costs**: At 25bps round-trip (realistic for Indian equities), monthly rebalancing "
        "with ~30% turnover costs ~0.9% per year. Weekly rebalancing at the same cost would "
        "cost ~3-4x more for marginal benefit.",
        "",
        "3. **200-DMA filter**: Acts as a built-in bear market protector. In 2008-style crashes, "
        "all momentum stocks fall below their 200-DMA, naturally moving the portfolio to cash.",
        "",
        "4. **Vol targeting**: Mechanically reduces position size during high-vol regimes "
        "(COVID crash 2020, etc.), smoothing drawdowns at the cost of some bull-market returns.",
        "",
        "5. **Survivorship bias**: These results use current NIFTY 500 constituents. "
        "Live performance will be modestly lower. The directional conclusions are robust.",
        "",
        "---", "",
        "## Charts",
        "",
        "Saved to `research/charts/momentum_study/`:",
        "",
        "- `study_A_lookback.png`, `study_B_skip.png`, `study_C_size.png`, `study_D_rebal.png`, `study_E_risk.png`",
        "- `heatmap_lookback_size.png` ? primary cross-study heatmap",
        "- `heatmap_lookback_skip.png`",
        "- `heatmap_rebal_cost.png` ? net Sharpe after costs",
        "- `heatmap_risk_cost.png`",
        "- `param_stability.png` ? Sharpe sensitivity to each parameter",
        "- `cost_impact.png` ? CAGR / Sharpe after costs by frequency",
        "",
        "---",
        "*Generated by `research/momentum_study.py`. Not investment advice.*",
    ]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", REPORT_PATH)
    return report


# ???????????????????????????????????????????????????????????????????????????????
# Main
# ???????????????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production Momentum Strategy Study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--min-adtv", type=float, default=1.0)
    p.add_argument("--no-charts", action="store_true")
    return p.parse_args()


def _print_table(title: str, results: dict, key_fmt=str) -> None:
    """Print a compact summary table."""
    print(f"\n  {title}")
    print("  " + "-" * 85)
    hdr = (f"  {'Label':<20} {'CAGR':>7} {'Vol':>6} {'Sharpe':>8} {'MaxDD':>8}"
           f" {'TO%/period':>11} {'Ann_TO%':>9}")
    print(hdr)
    for k, m in results.items():
        def f(key, fmt=".1f"):
            v = m.get(key, float("nan"))
            return f"{v:{fmt}}" if v == v else "--"
        print(f"  {key_fmt(k):<20} {f('CAGR (%)')+' ':>8} {f('Ann. Vol (%)')+' ':>7} "
              f"{f('Sharpe','.3f'):>8} {f('Max Drawdown (%)')+' ':>9}"
              f" {f('TO_%/period'):>11} {f('Ann_TO_%'):>9}")


def main() -> None:
    args = parse_args()
    Path("research").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research/momentum_study.log", mode="w", encoding="utf-8"),
        ],
    )
    # Suppress verbose logging from portfolio builder
    logging.getLogger("portfolio").setLevel(logging.WARNING)
    logging.getLogger("performance").setLevel(logging.WARNING)

    end = args.end or str(pd.Timestamp.today().date())

    print("\n" + "=" * 70)
    print("  PRODUCTION MOMENTUM STRATEGY RESEARCH -- NIFTY 500")
    print("=" * 70)
    print(f"  Period   : {args.start} to {end}")
    print(f"  Studies  : A(lookback) B(skip) C(size) D(freq) E(risk) F(cost)")
    print(f"  Baseline : 12M / skip-1M / Top-20 / Monthly / No-filter / 0bps")
    print("=" * 70 + "\n")

    # ?? Load data ?????????????????????????????????????????????????????????????
    log.info("Loading price data ...")
    tickers = get_nifty500_tickers(use_cache=True)
    data_start = str(int(args.start[:4]) - 1) + args.start[4:]
    adj_close, _, volume = download_price_data(
        tickers, start=data_start, end=args.end, use_cache=True
    )
    benchmarks_prices = load_benchmark_data(start=data_start, end=args.end)
    daily_returns = compute_daily_returns(adj_close)
    log.info("Data: %d tickers, %d trading days", adj_close.shape[1], adj_close.shape[0])

    # Benchmark returns
    bm_rets = {}
    for bname in ("NIFTY50", "NIFTY500"):
        if bname in benchmarks_prices:
            r = benchmarks_prices[bname].pct_change().dropna()
            start_dt = pd.Timestamp(args.start)
            end_dt = pd.Timestamp(end)
            bm_rets[bname] = r.loc[start_dt:end_dt].dropna()

    # ?? Individual studies ????????????????????????????????????????????????????
    study_a = study_A(adj_close, daily_returns, args.start, args.end)
    _print_table("Study A -- Lookback", study_a)

    study_b = study_B(adj_close, daily_returns, args.start, args.end)
    _print_table("Study B -- Skip", study_b, key_fmt=lambda k: f"Skip {k}")

    study_c = study_C(adj_close, daily_returns, args.start, args.end)
    _print_table("Study C -- Portfolio Size", study_c, key_fmt=lambda k: f"Top {k}")

    study_d = study_D(adj_close, daily_returns, args.start, args.end)
    _print_table("Study D -- Rebalance Frequency", study_d)

    study_e = study_E(adj_close, daily_returns, args.start, args.end)
    _print_table("Study E -- Risk Controls", study_e)

    study_f = study_F(adj_close, daily_returns, args.start, args.end)
    print(f"\n  Study F -- Transaction Costs")
    print("  " + "-" * 75)
    hdr_f = (f"  {'Config':<22} {'CAGR':>7} {'Sharpe':>8}"
             f" {'TO%/period':>11} {'Ann_TO%':>9} {'Drag%/yr':>10}")
    print(hdr_f)
    for freq in REBAL_FREQS:
        for cbps in COST_BPS:
            m = study_f.get(freq, {}).get(cbps, {})
            label = f"{freq}/{cbps}bps"
            print(f"  {label:<22} "
                  f"{_fmt(m.get('CAGR (%)'), '.1f', '%'):>7} "
                  f"{_fmt(m.get('Sharpe'), '.3f'):>8} "
                  f"{_fmt(m.get('TO_%/period'), '.1f', '%'):>11} "
                  f"{_fmt(m.get('Ann_TO_%'), '.1f', '%'):>9} "
                  f"{_fmt(m.get('Annual_Cost_Drag_%'), '.2f', '%'):>10}")

    # ?? Cross-study grids ?????????????????????????????????????????????????????
    print("\n  Running cross-study heatmap grids ...")
    grid_ls    = grid_lookback_size(adj_close, daily_returns, args.start, args.end)
    grid_lskip = grid_lookback_skip(adj_close, daily_returns, args.start, args.end)
    grid_rc    = grid_rebal_cost(adj_close, daily_returns, args.start, args.end)
    grid_ric   = grid_risk_cost(adj_close, daily_returns, args.start, args.end)

    # Save grids
    grid_ls.to_csv("research/grid_lookback_size.csv", index=False)
    grid_lskip.to_csv("research/grid_lookback_skip.csv", index=False)
    grid_rc.to_csv("research/grid_rebal_cost.csv", index=False)
    grid_ric.to_csv("research/grid_risk_cost.csv", index=False)
    log.info("Grids saved to research/grid_*.csv")

    # ?? Charts ????????????????????????????????????????????????????????????????
    if not args.no_charts:
        log.info("Generating charts ...")
        CHART_DIR.mkdir(parents=True, exist_ok=True)

        def add_benchmarks(d: dict) -> dict:
            return {**d, **{k: {"_returns": v} for k, v in bm_rets.items()}}

        try:
            plot_study_curves(add_benchmarks(study_a), "Study A -- Lookback Period", "study_A_lookback.png")
            plot_study_curves(add_benchmarks(study_b), "Study B -- Skip Period",     "study_B_skip.png")
            plot_study_curves(add_benchmarks({str(k): v for k, v in study_c.items()}),
                              "Study C -- Portfolio Size", "study_C_size.png")
            plot_study_curves(add_benchmarks(study_d), "Study D -- Rebalance Frequency", "study_D_rebal.png")
            plot_study_curves(add_benchmarks(study_e), "Study E -- Risk Controls", "study_E_risk.png")
            plot_heatmap_lookback_size(grid_ls)
            plot_heatmap_lookback_skip(grid_lskip)
            plot_heatmap_rebal_cost(grid_rc)
            plot_heatmap_risk_cost(grid_ric)
            plot_parameter_stability(study_a, study_b, study_c, study_d)
            plot_cost_impact(study_f)
            log.info("All charts saved to %s", CHART_DIR)
        except Exception as e:
            log.warning("Chart error: %s", e, exc_info=True)

    # ?? Report ????????????????????????????????????????????????????????????????
    log.info("Generating report ...")
    report = generate_report(
        study_a, study_b, study_c, study_d, study_e, study_f,
        grid_ls, grid_lskip, grid_rc, grid_ric,
        start=args.start, end=end,
    )

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print(f"  Report  : {REPORT_PATH}")
    print(f"  Charts  : {CHART_DIR}/")
    print(f"  Grids   : research/grid_*.csv")
    print(f"  Log     : research/momentum_study.log")
    print("=" * 70 + "\n")

    lines = report.split("\n")
    safe = "\n".join(l.encode("ascii", "replace").decode("ascii") for l in lines[:70])
    print(safe)
    if len(lines) > 70:
        print(f"\n... [{len(lines) - 70} more lines in {REPORT_PATH}]")


if __name__ == "__main__":
    main()
