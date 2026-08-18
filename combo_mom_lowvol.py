"""
research/combo_mom_lowvol.py
-----------------------------
Momentum x Low-Volatility Combination Study – Indian Equities (NIFTY 500)

Tests whether overlaying a Low-Volatility filter on top of a Momentum_12M
selection improves risk-adjusted returns.

Portfolio Methods
-----------------
A  Sequential Filter  : top 30% by MOM_12M, then lowest-vol 50% of that group
B  Composite 70/30    : 0.70 * mom_rank + 0.30 * lowvol_rank, top quintile
C  Composite 50/50    : 0.50 * mom_rank + 0.50 * lowvol_rank, top quintile
D  Momentum baseline  : top quintile by MOM_12M only
E  Low-Vol baseline   : top quintile by LOW_VOL only

Benchmarks
----------
- NIFTY 50     (^NSEI)
- NIFTY 500    (^CRSLDX)
- MOM_12M D10  (top decile pure momentum)
- MAX D1       (bottom decile MAX — lowest lottery stocks)

Usage
-----
    python research/combo_mom_lowvol.py
    python research/combo_mom_lowvol.py --start 2012-01-01
    python research/combo_mom_lowvol.py --no-charts
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ── path wiring ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_nifty500_tickers, download_price_data, load_benchmark_data
from factor_engine import compute_daily_returns, compute_max_factor, get_all_decile_constituents
from portfolio import get_rebalance_dates, build_portfolio_returns, build_all_decile_returns
from performance import compute_metrics, TRADING_DAYS, RISK_FREE_RATE

from factors import compute_momentum_12m, compute_low_vol, compute_max_scores

log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CHART_DIR = Path("research/charts/combo_mom_lowvol")
REPORT_PATH = Path("research/combo_mom_lowvol_report.md")

COLORS = {
    "A_Sequential": "#d62728",    # red
    "B_Combo_7030": "#ff7f0e",    # orange
    "C_Combo_5050": "#2ca02c",    # green
    "D_Mom_Only":   "#1f77b4",    # blue
    "E_LowVol_Only":"#9467bd",    # purple
    "MOM12M_D10":   "#8c564b",    # brown
    "MAX_D1":       "#e377c2",    # pink
    "NIFTY50":      "#555555",    # grey
    "NIFTY500":     "#aaaaaa",    # light grey
}


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _percentile_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank: 0 (worst) → 1 (best)."""
    valid = s.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    ranked = valid.rank(method="average") / len(valid)
    return ranked.reindex(s.index)


def build_method_constituents(
    mom_df: pd.DataFrame,
    lowvol_df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    method: str,
    top_pct: float = 0.20,
) -> tuple[dict, dict]:
    """
    Build constituent maps for each portfolio construction method.

    Returns (constituents_dict, monthly_sizes)
    where constituents_dict = {date: [tickers]}
    """
    constituents: dict = {}
    sizes: dict = {}

    for date in rebalance_dates:
        mom_row = mom_df.loc[date] if date in mom_df.index else pd.Series(dtype=float)
        lv_row = lowvol_df.loc[date] if date in lowvol_df.index else pd.Series(dtype=float)

        if mom_row.empty and lv_row.empty:
            constituents[date] = []
            sizes[date] = 0
            continue

        if method == "A":
            # Sequential: top 30% MOM → lowest vol 50% within
            if mom_row.dropna().empty:
                constituents[date] = []; sizes[date] = 0; continue
            mom_valid = mom_row.dropna().sort_values(ascending=False)
            n_mom = max(1, int(len(mom_valid) * 0.30))
            mom_top = mom_valid.head(n_mom).index

            # Within MOM top, pick lowest vol (highest LOW_VOL score)
            lv_sub = lv_row.reindex(mom_top).dropna()
            if lv_sub.empty:
                # Fallback: if no vol data, take all MOM top
                sel = list(mom_top[:max(1, n_mom // 2)])
            else:
                n_lv = max(1, int(len(lv_sub) * 0.50))
                sel = list(lv_sub.sort_values(ascending=False).head(n_lv).index)
            constituents[date] = sel
            sizes[date] = len(sel)

        elif method in ("B", "C"):
            weight_mom = 0.70 if method == "B" else 0.50
            weight_lv = 1.0 - weight_mom
            # Align on stocks present in both
            common = mom_row.dropna().index.intersection(lv_row.dropna().index)
            if len(common) < 5:
                constituents[date] = []; sizes[date] = 0; continue
            rank_m = _percentile_rank(mom_row.loc[common])
            rank_v = _percentile_rank(lv_row.loc[common])
            composite = weight_mom * rank_m + weight_lv * rank_v
            n_sel = max(1, int(len(composite) * top_pct))
            sel = list(composite.sort_values(ascending=False).head(n_sel).index)
            constituents[date] = sel
            sizes[date] = len(sel)

        elif method == "D":
            valid = mom_row.dropna()
            if valid.empty:
                constituents[date] = []; sizes[date] = 0; continue
            n_sel = max(1, int(len(valid) * top_pct))
            sel = list(valid.sort_values(ascending=False).head(n_sel).index)
            constituents[date] = sel
            sizes[date] = len(sel)

        elif method == "E":
            valid = lv_row.dropna()
            if valid.empty:
                constituents[date] = []; sizes[date] = 0; continue
            n_sel = max(1, int(len(valid) * top_pct))
            sel = list(valid.sort_values(ascending=False).head(n_sel).index)
            constituents[date] = sel
            sizes[date] = len(sel)

        else:
            constituents[date] = []
            sizes[date] = 0

    return constituents, sizes


# ═══════════════════════════════════════════════════════════════════════════════
# Turnover
# ═══════════════════════════════════════════════════════════════════════════════

def compute_turnover(constituents: dict) -> float:
    """
    One-sided monthly portfolio turnover (%).
    turnover_t = |new entries at t| / |portfolio at t|
    """
    dates = sorted(constituents.keys())
    monthly_to = []
    for i in range(1, len(dates)):
        prev = set(constituents[dates[i - 1]])
        curr = set(constituents[dates[i]])
        if not curr:
            continue
        new_entries = curr - prev
        monthly_to.append(len(new_entries) / len(curr))
    return round(np.mean(monthly_to) * 100, 1) if monthly_to else np.nan


def compute_turnover_series(constituents: dict) -> pd.Series:
    """Monthly turnover % as a time series."""
    dates = sorted(constituents.keys())
    data = {}
    for i in range(1, len(dates)):
        prev = set(constituents[dates[i - 1]])
        curr = set(constituents[dates[i]])
        if curr:
            data[dates[i]] = len(curr - prev) / len(curr) * 100
    return pd.Series(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics with turnover
# ═══════════════════════════════════════════════════════════════════════════════

def full_metrics(returns: pd.Series, label: str, turnover_pct: float = np.nan) -> dict:
    m = compute_metrics(returns, label)
    m["Turnover (% /mo)"] = round(turnover_pct, 1) if not np.isnan(turnover_pct) else np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _year_ax(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _max_drawdown_series(r: pd.Series) -> pd.Series:
    cum = (1 + r.fillna(0)).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def _rolling_sharpe(r: pd.Series, window: int = 252) -> pd.Series:
    rf_d = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    exc = r - rf_d
    return (exc.rolling(window).mean() / exc.rolling(window).std()) * np.sqrt(TRADING_DAYS)


def plot_equity_curves(returns_dict: dict[str, pd.Series]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(15, 7))
    for label, r in returns_dict.items():
        color = COLORS.get(label, "#333333")
        lw = 2.2 if label.startswith(("A_", "B_", "C_")) else 1.6
        ls = "--" if label in ("NIFTY50", "NIFTY500", "MAX_D1") else "-"
        cum = (1 + r.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=label, color=color, lw=lw, ls=ls)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_title("Momentum x Low-Vol Combination Study – Equity Curves (log scale)",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Wealth (starting at 1×)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "equity_curves.png")


def plot_drawdown(returns_dict: dict[str, pd.Series]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(15, 6))
    for label, r in returns_dict.items():
        color = COLORS.get(label, "#333333")
        lw = 2.0 if label.startswith(("A_", "B_", "C_")) else 1.4
        ls = "--" if label in ("NIFTY50", "NIFTY500", "MAX_D1") else "-"
        dd = _max_drawdown_series(r)
        ax.plot(dd.index, dd.values, label=label, color=color, lw=lw, ls=ls)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Drawdown Comparison", fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncol=2, framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "drawdown.png")


def plot_rolling_sharpe(returns_dict: dict[str, pd.Series], window: int = 252) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(15, 6))
    for label, r in returns_dict.items():
        color = COLORS.get(label, "#333333")
        lw = 2.0 if label.startswith(("A_", "B_", "C_")) else 1.4
        ls = "--" if label in ("NIFTY50", "NIFTY500", "MAX_D1") else "-"
        rs = _rolling_sharpe(r, window)
        ax.plot(rs.index, rs.values, label=label, color=color, lw=lw, ls=ls)

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.axhline(1, color="green", lw=0.6, ls=":", alpha=0.6)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontweight="bold", pad=12)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "rolling_sharpe.png")


def plot_metrics_bars(metrics_list: list[dict], metric_cols: list[str]) -> None:
    _style()
    labels = [m["Label"] for m in metrics_list]
    colors = [COLORS.get(lbl, "#555555") for lbl in labels]
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle("Risk/Return Metric Comparison", fontweight="bold", fontsize=14)

    for ax, col in zip(axes, metric_cols):
        vals = [m.get(col, np.nan) for m in metrics_list]
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(col, fontweight="bold")
        for bar, v in zip(bars, vals):
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                ypos = bar.get_height() + (0.2 if bar.get_height() >= 0 else -1.5)
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    _save(fig, CHART_DIR / "metrics_comparison.png")


def plot_turnover(turnover_series: dict[str, pd.Series]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, ts in turnover_series.items():
        color = COLORS.get(label, "#555555")
        ax.plot(ts.index, ts.values, color=color, lw=1.5, label=label, marker="o",
                ms=2, alpha=0.8)
        ax.axhline(ts.mean(), color=color, lw=0.8, ls="--", alpha=0.5)

    ax.set_title("Monthly Portfolio Turnover (%) – dashed = mean", fontweight="bold", pad=12)
    ax.set_ylabel("Turnover (%)")
    ax.legend(framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "turnover.png")


def plot_combined_dashboard(returns_dict: dict[str, pd.Series], metrics_list: list[dict]) -> None:
    _style()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Momentum x Low-Vol Study – Summary Dashboard",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Equity curves
    ax1 = fig.add_subplot(gs[0, :2])
    for label, r in returns_dict.items():
        cum = (1 + r.fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, label=label, color=COLORS.get(label),
                 lw=2.0 if label.startswith(("A_","B_","C_")) else 1.3,
                 ls="--" if label in ("NIFTY50","NIFTY500","MAX_D1") else "-")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_title("Equity Curves (log scale)", fontweight="bold")
    ax1.legend(ncol=3, fontsize=7.5)
    _year_ax(ax1)

    # Panel 2: CAGR bars
    ax2 = fig.add_subplot(gs[0, 2])
    labels = [m["Label"] for m in metrics_list]
    cagrs = [m.get("CAGR (%)", np.nan) for m in metrics_list]
    colors_ = [COLORS.get(l, "#555") for l in labels]
    ax2.barh(labels, cagrs, color=colors_, edgecolor="white")
    ax2.set_title("CAGR (%)", fontweight="bold")
    ax2.axvline(0, color="black", lw=0.5)

    # Panel 3: Drawdown
    ax3 = fig.add_subplot(gs[1, :2])
    for label, r in returns_dict.items():
        dd = _max_drawdown_series(r)
        ax3.plot(dd.index, dd.values, label=label, color=COLORS.get(label),
                 lw=1.8 if label.startswith(("A_","B_","C_")) else 1.2,
                 ls="--" if label in ("NIFTY50","NIFTY500","MAX_D1") else "-")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax3.set_title("Drawdown", fontweight="bold")
    ax3.legend(ncol=3, fontsize=7.5, loc="lower left")
    _year_ax(ax3)

    # Panel 4: Sharpe bars
    ax4 = fig.add_subplot(gs[1, 2])
    sharpes = [m.get("Sharpe", np.nan) for m in metrics_list]
    ax4.barh(labels, sharpes, color=colors_, edgecolor="white")
    ax4.set_title("Sharpe Ratio", fontweight="bold")
    ax4.axvline(0, color="black", lw=0.5)

    _save(fig, CHART_DIR / "dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt=".2f", suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:{fmt}}{suf}"


def generate_report(metrics_list: list[dict], backtest_start: str, backtest_end: str) -> str:
    import datetime
    run_date = datetime.date.today().isoformat()

    method_desc = {
        "A_Sequential": "Top 30% MOM -> Lowest vol 50% of that group",
        "B_Combo_7030": "Composite: 70% mom_rank + 30% lowvol_rank, top quintile",
        "C_Combo_5050": "Composite: 50% mom_rank + 50% lowvol_rank, top quintile",
        "D_Mom_Only":   "Top quintile by MOMENTUM_12M only",
        "E_LowVol_Only":"Top quintile by LOW_VOL only",
        "MOM12M_D10":   "Top decile MOMENTUM_12M (D10 — pure factor benchmark)",
        "MAX_D1":       "Bottom decile MAX (low lottery stocks — defensive benchmark)",
        "NIFTY50":      "NIFTY 50 Index",
        "NIFTY500":     "NIFTY 500 Index",
    }

    lines = [
        "# Momentum × Low-Volatility Combination Study",
        "",
        f"**Universe:** NIFTY 500 (current constituents)  ",
        f"**Period:** {backtest_start} to {backtest_end}  ",
        f"**Generated:** {run_date}  ",
        f"**Research question:** Does overlaying a Low-Vol filter improve Momentum's risk-adjusted returns?",
        "",
        "> **Survivorship bias caveat:** Current constituent list is used. Historical delistings",
        "> are excluded. Results are directionally informative but overstate live performance.",
        "",
        "---",
        "",
        "## Method Definitions",
        "",
        "| Label | Description |",
        "|-------|-------------|",
    ]
    for m in metrics_list:
        lbl = m["Label"]
        desc = method_desc.get(lbl, "")
        lines.append(f"| **{lbl}** | {desc} |")

    lines += [
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        "| Portfolio | CAGR (%) | Ann Ret (%) | Ann Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) | Turnover (%/mo) |",
        "|-----------|----------|-------------|-------------|--------|---------|--------|------------|-----------------|",
    ]

    for m in metrics_list:
        lines.append(
            f"| **{m['Label']}** | "
            f"{_fmt(m.get('CAGR (%)'), '.2f')} | "
            f"{_fmt(m.get('Ann. Return (%)'), '.2f')} | "
            f"{_fmt(m.get('Ann. Vol (%)'), '.2f')} | "
            f"{_fmt(m.get('Sharpe'), '.3f')} | "
            f"{_fmt(m.get('Sortino'), '.3f')} | "
            f"{_fmt(m.get('Calmar'), '.3f')} | "
            f"{_fmt(m.get('Max Drawdown (%)'), '.2f')} | "
            f"{_fmt(m.get('Turnover (% /mo)'), '.1f')} |"
        )

    # Find best portfolio by Sharpe among the 5 methods
    method_labels = ["A_Sequential", "B_Combo_7030", "C_Combo_5050", "D_Mom_Only", "E_LowVol_Only"]
    method_metrics = {m["Label"]: m for m in metrics_list if m["Label"] in method_labels}
    best_sharpe_label = max(method_metrics, key=lambda l: method_metrics[l].get("Sharpe", -np.inf))
    best_dd_label = max(method_metrics, key=lambda l: -abs(method_metrics[l].get("Max Drawdown (%)", -100)))

    lines += [
        "",
        "---",
        "",
        "## Analysis",
        "",
        "### Does Low-Vol improve Momentum?",
        "",
    ]

    d_sharpe = method_metrics.get("D_Mom_Only", {}).get("Sharpe", np.nan)
    e_sharpe = method_metrics.get("E_LowVol_Only", {}).get("Sharpe", np.nan)
    a_sharpe = method_metrics.get("A_Sequential", {}).get("Sharpe", np.nan)
    b_sharpe = method_metrics.get("B_Combo_7030", {}).get("Sharpe", np.nan)
    c_sharpe = method_metrics.get("C_Combo_5050", {}).get("Sharpe", np.nan)

    d_cagr = method_metrics.get("D_Mom_Only", {}).get("CAGR (%)", np.nan)
    a_cagr = method_metrics.get("A_Sequential", {}).get("CAGR (%)", np.nan)
    b_cagr = method_metrics.get("B_Combo_7030", {}).get("CAGR (%)", np.nan)
    c_cagr = method_metrics.get("C_Combo_5050", {}).get("CAGR (%)", np.nan)

    d_dd = method_metrics.get("D_Mom_Only", {}).get("Max Drawdown (%)", np.nan)
    a_dd = method_metrics.get("A_Sequential", {}).get("Max Drawdown (%)", np.nan)
    b_dd = method_metrics.get("B_Combo_7030", {}).get("Max Drawdown (%)", np.nan)

    lines += [
        f"**Pure Momentum baseline (D):** CAGR = {_fmt(d_cagr, '.1f', '%')}, Sharpe = {_fmt(d_sharpe, '.3f')}, Max DD = {_fmt(d_dd, '.1f', '%')}",
        "",
        "**Combination methods vs pure momentum:**",
        "",
        f"- Method A (Sequential): Sharpe {_fmt(a_sharpe, '.3f')} vs {_fmt(d_sharpe, '.3f')} baseline. CAGR {_fmt(a_cagr, '.1f', '%')}. Max DD {_fmt(a_dd, '.1f', '%')}.",
        f"- Method B (70/30): Sharpe {_fmt(b_sharpe, '.3f')}. CAGR {_fmt(b_cagr, '.1f', '%')}. Max DD {_fmt(b_dd, '.1f', '%')}.",
        f"- Method C (50/50): Sharpe {_fmt(c_sharpe, '.3f')}. CAGR {_fmt(c_cagr, '.1f', '%')}.",
        "",
        f"**Best overall Sharpe:** {best_sharpe_label} ({_fmt(method_metrics[best_sharpe_label].get('Sharpe'), '.3f')})",
        f"**Smallest Max Drawdown:** {best_dd_label} ({_fmt(method_metrics[best_dd_label].get('Max Drawdown (%)'), '.1f', '%')})",
        "",
    ]

    # Assess improvement
    combo_sharpes = [a_sharpe, b_sharpe, c_sharpe]
    combos_above_d = sum(1 for s in combo_sharpes if not np.isnan(s) and not np.isnan(d_sharpe) and s > d_sharpe)
    combos_above_d_label = f"{combos_above_d}/3"

    if combos_above_d >= 2:
        verdict = (
            f"**VERDICT: LOW-VOL IMPROVES MOMENTUM.** {combos_above_d_label} combination methods "
            f"beat pure momentum on Sharpe ratio. Overlaying a volatility filter reduces drawdowns "
            f"without sacrificing enough return to damage risk-adjusted performance."
        )
    elif combos_above_d == 1:
        verdict = (
            f"**VERDICT: MIXED.** Only {combos_above_d_label} combination method beats pure momentum "
            f"on Sharpe. The benefit of Low-Vol is partial — depends on the blending method."
        )
    else:
        verdict = (
            f"**VERDICT: LOW-VOL DOES NOT IMPROVE MOMENTUM.** {combos_above_d_label} combination "
            f"methods beat pure momentum on Sharpe. Volatility filtering dilutes the momentum signal "
            f"more than it reduces risk."
        )

    lines += [verdict, "", "### Turnover", ""]

    for m in metrics_list:
        if m["Label"] in method_labels:
            to = m.get("Turnover (% /mo)", np.nan)
            lines.append(f"- **{m['Label']}**: {_fmt(to, '.1f', '%/month')} one-sided turnover")

    lines += [
        "",
        "Higher Low-Vol weight -> lower turnover (volatility changes slowly).",
        "Sequential filter tends to have highest turnover because momentum filter changes rapidly.",
        "",
        "---",
        "",
        "## Charts",
        "",
        "All charts saved to `research/charts/combo_mom_lowvol/`:",
        "",
        "- `dashboard.png` — 4-panel summary",
        "- `equity_curves.png` — All methods + benchmarks (log scale)",
        "- `drawdown.png` — Drawdown comparison",
        "- `rolling_sharpe.png` — Rolling 1-year Sharpe",
        "- `metrics_comparison.png` — Bar charts: CAGR, Sharpe, Sortino, Max DD",
        "- `turnover.png` — Monthly turnover time series",
        "",
        "---",
        "*Generated by `research/combo_mom_lowvol.py`. Not investment advice.*",
    ]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", REPORT_PATH)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum x Low-Vol Combination Study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--min-adtv", type=float, default=1.0)
    p.add_argument("--top-pct", type=float, default=0.20,
                   help="Top fraction to select for B/C/D/E methods (default 0.20 = top quintile)")
    p.add_argument("--no-charts", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research/combo_mom_lowvol.log", mode="w", encoding="utf-8"),
        ],
    )

    print("\n" + "=" * 70)
    print("  MOMENTUM x LOW-VOL COMBINATION STUDY – INDIA EQUITIES")
    print("=" * 70)
    print(f"  Period   : {args.start} -> {args.end or 'today'}")
    print(f"  Universe : NIFTY 500")
    print(f"  Methods  : A (Sequential) | B (70/30) | C (50/50) | D (MOM only) | E (LowVol only)")
    print("=" * 70 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Loading cached price data ...")
    tickers = get_nifty500_tickers(use_cache=True)
    data_start = str(int(args.start[:4]) - 1) + args.start[4:]
    adj_close, _, volume = download_price_data(tickers, start=data_start, end=args.end, use_cache=True)
    benchmarks_prices = load_benchmark_data(start=data_start, end=args.end)
    daily_returns = compute_daily_returns(adj_close)
    rebalance_dates = get_rebalance_dates(adj_close, start=args.start, end=args.end)
    log.info("Data: %d tickers, %d days, %d rebalance periods", adj_close.shape[1], adj_close.shape[0], len(rebalance_dates))

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end) if args.end else pd.Timestamp.today()

    def trim(s: pd.Series) -> pd.Series:
        return s.loc[start_dt:end_dt].dropna()

    # ── 2. Compute factor scores ──────────────────────────────────────────────
    log.info("Computing MOMENTUM_12M factor ...")
    mom_df = compute_momentum_12m(adj_close, rebalance_dates)

    log.info("Computing LOW_VOL factor ...")
    lv_df = compute_low_vol(adj_close, rebalance_dates)

    # For benchmarks: MAX factor (to get D1) and MOM D10
    log.info("Computing MAX factor (for D1 benchmark) ...")
    vol_value = adj_close * volume
    max_df = compute_max_scores(adj_close, volume, rebalance_dates, min_adtv_crore=args.min_adtv)

    # ── 3. Build method portfolios ────────────────────────────────────────────
    method_specs = [
        ("A_Sequential", "A"),
        ("B_Combo_7030", "B"),
        ("C_Combo_5050", "C"),
        ("D_Mom_Only",   "D"),
        ("E_LowVol_Only","E"),
    ]

    method_returns: dict[str, pd.Series] = {}
    method_constituents: dict[str, dict] = {}

    for label, method_code in method_specs:
        log.info("Building portfolio: %s ...", label)
        constituents, sizes = build_method_constituents(
            mom_df, lv_df, rebalance_dates,
            method=method_code, top_pct=args.top_pct,
        )
        r = trim(build_portfolio_returns(daily_returns, constituents, rebalance_dates, label=label))
        method_returns[label] = r
        method_constituents[label] = constituents
        avg_size = np.mean([v for v in sizes.values() if v > 0])
        log.info("  %s: avg portfolio size = %.0f stocks", label, avg_size)

    # ── 4. Reference benchmarks ───────────────────────────────────────────────
    log.info("Building benchmark portfolios ...")

    # MOM_12M D10 — top decile pure momentum
    mom_constituents = get_all_decile_constituents(mom_df, n_deciles=10)
    mom_deciles = build_all_decile_returns(daily_returns, mom_constituents, rebalance_dates)
    mom_d10 = trim(mom_deciles[10])

    # MAX D1 — bottom decile MAX (lowest lottery stocks)
    max_constituents = get_all_decile_constituents(max_df, n_deciles=10)
    max_deciles = build_all_decile_returns(daily_returns, max_constituents, rebalance_dates)
    max_d1 = trim(max_deciles[1])

    # NIFTY 50
    nifty50_ret = pd.Series(dtype=float)
    if "NIFTY50" in benchmarks_prices:
        nifty50_ret = trim(benchmarks_prices["NIFTY50"].pct_change().dropna())
        nifty50_ret.name = "NIFTY50"

    # NIFTY 500
    nifty500_ret = pd.Series(dtype=float)
    if "NIFTY500" in benchmarks_prices:
        nifty500_ret = trim(benchmarks_prices["NIFTY500"].pct_change().dropna())
        nifty500_ret.name = "NIFTY500"

    # ── 5. Compile all returns ────────────────────────────────────────────────
    all_returns: dict[str, pd.Series] = {
        **method_returns,
        "MOM12M_D10": mom_d10,
        "MAX_D1": max_d1,
    }
    if not nifty50_ret.empty:
        all_returns["NIFTY50"] = nifty50_ret
    if not nifty500_ret.empty:
        all_returns["NIFTY500"] = nifty500_ret

    # ── 6. Compute metrics ────────────────────────────────────────────────────
    log.info("Computing performance metrics ...")
    metrics_list: list[dict] = []

    for label, r in all_returns.items():
        turnover_pct = (compute_turnover(method_constituents[label])
                        if label in method_constituents else np.nan)
        m = full_metrics(r, label=label, turnover_pct=turnover_pct)
        metrics_list.append(m)

    # ── 7. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"  {'Portfolio':<18} {'CAGR':>7} {'Vol':>7} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD':>8} {'TO%/mo':>8}")
    print("  " + "-" * 80)
    for m in metrics_list:
        def f(k, fmt=".1f"):
            v = m.get(k, float("nan"))
            return f"{v:{fmt}}" if v == v else "  —"
        print(f"  {m['Label']:<18} {f('CAGR (%)')+' ':>8} {f('Ann. Vol (%)')+' ':>8} "
              f"{f('Sharpe','.3f'):>8} {f('Sortino','.3f'):>8} {f('Calmar','.3f'):>8} "
              f"{f('Max Drawdown (%)')+' ':>9} {f('Turnover (% /mo)'):>8}")
    print("=" * 95 + "\n")

    # ── 8. Generate charts ────────────────────────────────────────────────────
    if not args.no_charts:
        log.info("Generating charts ...")
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        try:
            plot_equity_curves(all_returns)
            plot_drawdown(all_returns)
            plot_rolling_sharpe(all_returns)
            plot_metrics_bars(metrics_list, ["CAGR (%)", "Sharpe", "Sortino", "Max Drawdown (%)"])
            plot_combined_dashboard(all_returns, metrics_list)

            turnover_series_dict = {
                label: compute_turnover_series(method_constituents[label])
                for label in method_constituents
            }
            plot_turnover(turnover_series_dict)
            log.info("Charts saved to %s", CHART_DIR)
        except Exception as e:
            log.warning("Chart error: %s", e)

    # ── 9. Generate report ────────────────────────────────────────────────────
    log.info("Generating report ...")
    report = generate_report(metrics_list, args.start, args.end or str(end_dt.date()))
    preview_lines = report.split("\n")[:60]
    safe_preview = "\n".join(l.encode("ascii", "replace").decode("ascii") for l in preview_lines)
    print(safe_preview)
    extra = len(report.split("\n")) - 60
    if extra > 0:
        print(f"\n... [{extra} more lines in {REPORT_PATH}]")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print(f"  Report : {REPORT_PATH}")
    print(f"  Charts : {CHART_DIR}/")
    print(f"  Log    : research/combo_mom_lowvol.log")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
