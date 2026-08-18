"""
research/survivorship_assessment.py
-------------------------------------
Survivorship-bias / return-concentration assessment.

Questions answered
------------------
1. What fraction of total strategy return comes from the top 10 / top 20 stocks?
2. How does performance degrade when we exclude the top 5 / 10 / 20 contributors?
3. Is the return distribution broad (many contributors) or narrow (few extreme winners)?

Methodology
-----------
Contribution metric: each stock's total additive contribution to portfolio daily returns.
  contrib[stock] = sum over all days held of (1 / n_held_t) * daily_return[stock, t]

This is the exact per-day weight * return, summed over the backtest.
Cumulative product (CAGR) is inherently non-additive, so we use additive daily
contributions as a decomposition and check CAGR via geometric re-run on sub-portfolios.

Exclusion test: re-run strategy with excluded stocks removed from the selectable universe.
The portfolio still selects top-20 from the remaining stocks each period.
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_nifty500_tickers, download_price_data, load_benchmark_data
from factor_engine import compute_daily_returns
from performance import compute_metrics, TRADING_DAYS, RISK_FREE_RATE
from momentum_study import (
    compute_mom_scores, _select_top_n, _build_port,
    get_rebal_dates, BASE,
)

log = logging.getLogger(__name__)

CHART_DIR   = Path("research/charts/survivorship")
REPORT_PATH = Path("research/survivorship_report.md")

OPT = dict(lookback=252, skip=21, top_n=20, freq="monthly")
START = "2010-01-01"
END   = None   # through latest available data


# ===============================================================================
# Core: per-stock contribution
# ===============================================================================

def compute_stock_contributions(
    daily_returns: pd.DataFrame,
    constituents: dict,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    For each stock ever held, compute:
      - total_days_held    : calendar days in the portfolio
      - total_contrib      : sum of (1/n_t * r_stock_t) over all days held
      - mean_daily_ret     : average daily return while in the portfolio
      - n_periods_held     : number of rebalance periods held
      - total_return_held  : compounded return of the stock during periods held (geometric)
      - ticker             : stock symbol

    Returns a DataFrame sorted by total_contrib descending.
    """
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else pd.Timestamp.today()

    rebal_dates = sorted(constituents.keys())
    rows = []
    # Track per-stock daily contribution series for geometric calculation
    stock_contrib_log = {}   # ticker -> list of log(1+weighted_r) per day

    for i, t in enumerate(rebal_dates):
        t_next = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
        hold = daily_returns.loc[t:t_next] if t_next else daily_returns.loc[t:]
        if t_next:
            hold = hold.iloc[:-1]
        hold = hold.loc[start_dt:end_dt]
        if hold.empty:
            continue

        tickers = [tk for tk in constituents.get(t, []) if tk in daily_returns.columns]
        if not tickers:
            continue
        n = len(tickers)
        sub = daily_returns.loc[hold.index, tickers]

        for tk in tickers:
            r_series = sub[tk].fillna(0)
            contrib_series = r_series / n      # equal-weight contribution per day
            if tk not in stock_contrib_log:
                stock_contrib_log[tk] = {"additive": [], "log": [], "n_periods": 0}
            stock_contrib_log[tk]["additive"].extend(contrib_series.tolist())
            stock_contrib_log[tk]["log"].extend(np.log1p(contrib_series).tolist())
            stock_contrib_log[tk]["n_periods"] += 1

    for tk, data in stock_contrib_log.items():
        additive = np.array(data["additive"])
        log_vals = np.array(data["log"])
        n_days = len(additive)
        total_contrib = additive.sum()
        mean_daily = additive.mean() if n_days > 0 else np.nan
        # Geometric return of this stock's contribution stream
        geo_total = np.expm1(log_vals.sum())
        rows.append({
            "ticker": tk,
            "total_contrib": total_contrib,
            "total_contrib_%": total_contrib * 100,
            "geo_contribution": geo_total,
            "mean_daily_ret_%": mean_daily * 100,
            "total_days_held": n_days,
            "n_periods_held": data["n_periods"],
        })

    df = pd.DataFrame(rows).sort_values("total_contrib", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["cum_contrib_%"] = df["total_contrib_%"].cumsum()
    df["pct_of_total"] = df["total_contrib"] / df["total_contrib"].sum() * 100
    df["cum_pct_of_total"] = df["pct_of_total"].cumsum()
    return df


# ===============================================================================
# Exclusion re-run
# ===============================================================================

def run_excluding(
    mom_scores: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebal: pd.DatetimeIndex,
    exclude_tickers: set,
    top_n: int,
    start: str,
    end: str,
    label: str,
) -> dict:
    """Re-run the strategy with a set of stocks blacklisted from selection."""
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else pd.Timestamp.today()

    # Build exclusion mask: False for excluded stocks, True for all others
    exclude_mask = pd.DataFrame(
        True, index=mom_scores.index, columns=mom_scores.columns
    )
    for tk in exclude_tickers:
        if tk in exclude_mask.columns:
            exclude_mask[tk] = False

    constituents = _select_top_n(mom_scores, rebal, top_n, filter_mask=exclude_mask)
    port = _build_port(daily_returns, constituents, rebal)
    port = port.loc[start_dt:end_dt].dropna()

    m = compute_metrics(port, label=label)
    m["_returns"] = port
    m["_constituents"] = constituents
    m["n_excluded"] = len(exclude_tickers)
    m["label"] = label
    log.info("  Excl/%s: CAGR=%.1f%% Sharpe=%.3f MaxDD=%.1f%%",
             label, m.get("CAGR (%)", 0), m.get("Sharpe", 0), m.get("Max Drawdown (%)", 0))
    return m


# ===============================================================================
# Concentration analysis
# ===============================================================================

def lorenz_curve(contrib_df: pd.DataFrame) -> tuple:
    """
    Compute Lorenz curve and Gini coefficient from contribution data.
    Uses only positive contributors (convention: concentration of gains).
    Returns (x_vals, y_vals, gini).
    """
    pos = contrib_df[contrib_df["total_contrib"] > 0]["total_contrib"].sort_values().values
    if len(pos) == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.0

    n = len(pos)
    cum = np.cumsum(pos)
    total = cum[-1]

    x = np.concatenate([[0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0], cum / total])

    # Gini = 1 - 2 * area under Lorenz curve
    area = np.trapz(y, x)
    gini = 1 - 2 * area
    return x, y, round(gini, 4)


def gini_interpretation(g: float) -> str:
    if g < 0.3:
        return "LOW concentration -- broad return distribution"
    if g < 0.5:
        return "MODERATE concentration -- some dependence on winners"
    if g < 0.7:
        return "HIGH concentration -- significant winner dependence"
    return "EXTREME concentration -- dominated by very few winners"


# ===============================================================================
# Charts
# ===============================================================================

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
        "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.6,
        "font.family": "sans-serif", "font.size": 9,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dd(r):
    cum = (1 + r.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def plot_top_contributors(contrib_df: pd.DataFrame, top_n: int = 30) -> None:
    """Bar chart of top-N contributors with cumulative % overlay."""
    _style()
    top = contrib_df.head(top_n).copy()
    fig, ax1 = plt.subplots(figsize=(16, 7))
    fig.suptitle(f"Top {top_n} Stock Contributions to Strategy Return",
                 fontweight="bold", fontsize=13)

    colors = plt.cm.RdYlGn(np.linspace(0.9, 0.3, top_n))
    bars = ax1.bar(range(top_n), top["total_contrib_%"], color=colors, edgecolor="white")
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(
        [t.replace(".NS", "") for t in top["ticker"]],
        rotation=45, ha="right", fontsize=7.5
    )
    ax1.set_ylabel("Additive Return Contribution (%)")
    ax1.set_title(
        f"Each bar = sum of (weight x daily return) over all days held\n"
        f"Top {top_n} stocks account for {top['pct_of_total'].sum():.1f}% of total strategy gain",
        fontsize=9
    )

    # Cumulative % on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(top_n), top["cum_pct_of_total"].values,
             color="navy", lw=2, marker="o", ms=3, label="Cumulative % of total")
    ax2.axhline(50, color="orange", lw=1, ls="--", alpha=0.7, label="50% of total return")
    ax2.axhline(80, color="red",    lw=1, ls="--", alpha=0.7, label="80% of total return")
    ax2.set_ylabel("Cumulative % of Total Strategy Return", color="navy")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, CHART_DIR / "top_contributors.png")


def plot_lorenz(contrib_df: pd.DataFrame, x, y, gini: float) -> None:
    """Lorenz curve with Gini coefficient annotation."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Return Concentration Analysis", fontweight="bold", fontsize=13)

    # Lorenz curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Perfect equality")
    ax1.fill_between(x, y, x, alpha=0.2, color="crimson", label="Inequality area")
    ax1.plot(x, y, color="navy", lw=2, label=f"Lorenz curve (Gini={gini:.3f})")

    # Mark key thresholds
    for pct, color in [(0.5, "orange"), (0.8, "red")]:
        idx = np.searchsorted(y, pct)
        if idx < len(x):
            ax1.axhline(pct, color=color, lw=0.8, ls=":", alpha=0.7)
            ax1.axvline(x[idx], color=color, lw=0.8, ls=":", alpha=0.7)
            n_stocks_total = len(contrib_df[contrib_df["total_contrib"] > 0])
            n_stocks_needed = int(x[idx] * n_stocks_total)
            ax1.annotate(
                f"{int(pct*100)}% of return\nfrom {int(x[idx]*100)}% of stocks\n({n_stocks_needed} stocks)",
                xy=(x[idx], pct), xytext=(x[idx] + 0.08, pct - 0.12),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax1.set_xlabel("Cumulative fraction of stocks (ranked by contribution)")
    ax1.set_ylabel("Cumulative fraction of total return")
    ax1.set_title(f"Lorenz Curve -- Gini = {gini:.3f}\n{gini_interpretation(gini)}", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # Contribution histogram
    ax2 = axes[1]
    contribs = contrib_df["total_contrib_%"].values
    pos_mask = contribs > 0
    neg_mask = contribs < 0

    bins = np.linspace(contribs.min() - 0.5, contribs.max() + 0.5, 50)
    ax2.hist(contribs[pos_mask], bins=bins, color="steelblue",
             alpha=0.8, label=f"Positive ({pos_mask.sum()} stocks)")
    ax2.hist(contribs[neg_mask], bins=bins, color="crimson",
             alpha=0.8, label=f"Negative ({neg_mask.sum()} stocks)")

    # Mark top-10 and top-20 threshold
    if len(contrib_df) >= 20:
        thr_10 = contrib_df.iloc[9]["total_contrib_%"]
        thr_20 = contrib_df.iloc[19]["total_contrib_%"]
        ax2.axvline(thr_10, color="darkred",   lw=1.5, ls="--", label=f"Top 10 threshold ({thr_10:.1f}%)")
        ax2.axvline(thr_20, color="darkorange", lw=1.5, ls="--", label=f"Top 20 threshold ({thr_20:.1f}%)")

    ax2.set_xlabel("Total Additive Contribution to Strategy Return (%)")
    ax2.set_ylabel("Number of Stocks")
    ax2.set_title("Distribution of Per-Stock Contributions", fontweight="bold")
    ax2.legend(fontsize=9)

    # Stats box
    med = np.median(contribs)
    mean_ = contribs.mean()
    ax2.text(0.97, 0.97,
             f"n total: {len(contribs)}\n"
             f"n positive: {pos_mask.sum()}\n"
             f"n negative: {neg_mask.sum()}\n"
             f"mean: {mean_:.2f}%\n"
             f"median: {med:.2f}%\n"
             f"max: {contribs.max():.2f}%\n"
             f"min: {contribs.min():.2f}%",
             transform=ax2.transAxes, va="top", ha="right", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="lightyellow", ec="navy", alpha=0.9))

    fig.tight_layout()
    _save(fig, CHART_DIR / "lorenz_concentration.png")


def plot_exclusion_impact(baseline: dict, exclusion_results: list,
                          contrib_df: pd.DataFrame) -> None:
    """4-panel: equity curves, CAGR bars, Sharpe bars, MaxDD bars."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Strategy Performance After Excluding Top Contributors",
                 fontweight="bold", fontsize=13)

    all_results = [baseline] + exclusion_results
    labels = [r["label"] for r in all_results]
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd"]

    # Equity curves
    ax1 = axes[0, 0]
    for r, col in zip(all_results, colors):
        rets = r.get("_returns", pd.Series())
        if not rets.empty:
            cum = (1 + rets.fillna(0)).cumprod()
            ax1.plot(cum.index, cum, color=col, lw=1.8, label=r["label"])
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}x"))
    ax1.set_title("Equity Curves", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(2))
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

    # CAGR bar
    ax2 = axes[0, 1]
    cagrs = [r.get("CAGR (%)", np.nan) for r in all_results]
    bar2 = ax2.bar(range(len(labels)), cagrs, color=colors, edgecolor="white", width=0.6)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("CAGR (%)")
    ax2.set_title("CAGR by Exclusion Scenario", fontweight="bold")
    for b, v in zip(bar2, cagrs):
        if not np.isnan(v):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.axhline(cagrs[0], color="navy", lw=1, ls="--", alpha=0.5)

    # Sharpe bar
    ax3 = axes[1, 0]
    sharpes = [r.get("Sharpe", np.nan) for r in all_results]
    bar3 = ax3.bar(range(len(labels)), sharpes, color=colors, edgecolor="white", width=0.6)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_title("Sharpe Ratio by Exclusion Scenario", fontweight="bold")
    ax3.axhline(1.0, color="green", lw=1, ls=":", alpha=0.7, label="Sharpe=1.0")
    for b, v in zip(bar3, sharpes):
        if not np.isnan(v):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax3.legend(fontsize=8)

    # MaxDD bar
    ax4 = axes[1, 1]
    maxdds = [r.get("Max Drawdown (%)", np.nan) for r in all_results]
    bar4 = ax4.bar(range(len(labels)), [abs(v) for v in maxdds],
                   color=colors, edgecolor="white", width=0.6)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax4.set_ylabel("Max Drawdown (%, absolute)")
    ax4.set_title("Max Drawdown by Exclusion Scenario", fontweight="bold")
    ax4.axhline(abs(maxdds[0]), color="navy", lw=1, ls="--", alpha=0.5)
    for b, v in zip(bar4, maxdds):
        if not np.isnan(v):
            ax4.text(b.get_x() + b.get_width()/2, abs(v) + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    _save(fig, CHART_DIR / "exclusion_impact.png")


def plot_contribution_treemap(contrib_df: pd.DataFrame, top_n: int = 40) -> None:
    """Horizontal stacked bar: each row is a 'bucket', showing contribution share."""
    _style()
    top = contrib_df.head(top_n).copy()
    rest = contrib_df.iloc[top_n:]

    # Buckets
    buckets = [
        ("Top 1",    contrib_df.iloc[0:1]["total_contrib"].sum()),
        ("Top 2-5",  contrib_df.iloc[1:5]["total_contrib"].sum()),
        ("Top 6-10", contrib_df.iloc[5:10]["total_contrib"].sum()),
        ("Top 11-20",contrib_df.iloc[10:20]["total_contrib"].sum()),
        ("Top 21-50",contrib_df.iloc[20:50]["total_contrib"].sum()),
        ("Rest",     contrib_df.iloc[50:]["total_contrib"].sum()),
    ]
    total_pos = contrib_df[contrib_df["total_contrib"] > 0]["total_contrib"].sum()
    buckets_pct = [(k, v / total_pos * 100) for k, v in buckets if v > 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Return Attribution -- Contribution Buckets", fontweight="bold", fontsize=13)

    # Stacked horizontal bar
    ax1 = axes[0]
    cmap = ["#1a237e", "#1565c0", "#0288d1", "#26c6da", "#a5d6a7", "#e8f5e9"]
    left = 0
    for (k, pct), col in zip(buckets_pct, cmap):
        ax1.barh(0, pct, left=left, color=col, edgecolor="white", height=0.5)
        if pct > 3:
            ax1.text(left + pct/2, 0, f"{k}\n{pct:.1f}%",
                     ha="center", va="center", fontsize=8.5,
                     color="white", fontweight="bold")
        left += pct
    ax1.set_xlim(0, 100)
    ax1.set_yticks([])
    ax1.set_xlabel("Share of Total Positive Return (%)")
    ax1.set_title("Return Attribution by Stock Group", fontweight="bold")
    patches = [mpatches.Patch(color=c, label=k) for (k, _), c in zip(buckets_pct, cmap)]
    ax1.legend(handles=patches, fontsize=8, loc="lower right")
    ax1.set_ylim(-1, 1)

    # Waterfall: top-20 stocks
    ax2 = axes[1]
    top20 = contrib_df.head(20).copy()
    top20["color"] = ["#1a237e" if i < 1 else "#0288d1" if i < 5
                      else "#26c6da" if i < 10 else "#a5d6a7"
                      for i in range(20)]
    tickers_short = [t.replace(".NS", "") for t in top20["ticker"]]
    bars = ax2.barh(range(19, -1, -1), top20["pct_of_total"].values,
                    color=top20["color"].values, edgecolor="white")
    ax2.set_yticks(range(19, -1, -1))
    ax2.set_yticklabels(tickers_short, fontsize=7.5)
    ax2.set_xlabel("Share of Total Positive Return (%)")
    ax2.set_title("Top 20 Stocks -- Individual Share of Total Return", fontweight="bold")
    for b, v in zip(bars, top20["pct_of_total"].values):
        ax2.text(v + 0.1, b.get_y() + b.get_height()/2,
                 f"{v:.1f}%", va="center", fontsize=7.5)
    legend_items = [
        mpatches.Patch(color="#1a237e", label="#1"),
        mpatches.Patch(color="#0288d1", label="#2-5"),
        mpatches.Patch(color="#26c6da", label="#6-10"),
        mpatches.Patch(color="#a5d6a7", label="#11-20"),
    ]
    ax2.legend(handles=legend_items, fontsize=8)

    fig.tight_layout()
    _save(fig, CHART_DIR / "contribution_treemap.png")


# ===============================================================================
# Report
# ===============================================================================

def _fmt(v, fmt=".2f", suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:{fmt}}{suf}"


def generate_report(
    contrib_df: pd.DataFrame,
    baseline: dict,
    exclusion_results: list,
    gini: float,
    x_lorenz, y_lorenz,
    start: str, end: str,
) -> str:
    import datetime
    run_date = datetime.date.today().isoformat()

    total_pos = contrib_df[contrib_df["total_contrib"] > 0]["total_contrib"].sum()
    n_pos = (contrib_df["total_contrib"] > 0).sum()
    n_neg = (contrib_df["total_contrib"] < 0).sum()
    n_total = len(contrib_df)

    top10_pct  = contrib_df.head(10)["pct_of_total"].sum()
    top20_pct  = contrib_df.head(20)["pct_of_total"].sum()
    top50_pct  = contrib_df.head(50)["pct_of_total"].sum() if len(contrib_df) >= 50 else contrib_df["pct_of_total"].sum()

    # Find stocks at 50% and 80% of cumulative return
    idx_50 = (contrib_df["cum_pct_of_total"] >= 50).idxmax()
    idx_80 = (contrib_df["cum_pct_of_total"] >= 80).idxmax()
    n_stocks_50 = idx_50 + 1
    n_stocks_80 = idx_80 + 1

    lines = [
        "# Survivorship Bias & Return Concentration Assessment",
        "## NIFTY 500 Momentum Strategy -- 12M, Top 20, Monthly",
        "",
        f"**Period:** {start} to {end}  ",
        f"**Generated:** {run_date}  ",
        "",
        "> This study measures how concentrated the strategy's returns are.",
        "> High concentration suggests dependence on a small number of winners",
        "> that may not persist in live trading.",
        "",
        "---",
        "",
        "## 1. Universe Coverage",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total stocks ever held | {n_total} |",
        f"| Stocks with positive contribution | {n_pos} ({n_pos/n_total*100:.1f}%) |",
        f"| Stocks with negative contribution | {n_neg} ({n_neg/n_total*100:.1f}%) |",
        f"| Stocks generating 50% of total return | {n_stocks_50} |",
        f"| Stocks generating 80% of total return | {n_stocks_80} |",
        "",
        "---",
        "",
        "## 2. Top Contributors",
        "",
        "### Top 10 Contributors",
        "",
        "| Rank | Ticker | Total Contrib (%) | % of Total Return | Cum % | Periods Held |",
        "|------|--------|------------------|------------------|-------|--------------|",
    ]

    for _, row in contrib_df.head(10).iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['ticker'].replace('.NS','')} | "
            f"{row['total_contrib_%']:.2f}% | {row['pct_of_total']:.2f}% | "
            f"{row['cum_pct_of_total']:.1f}% | {int(row['n_periods_held'])} |"
        )

    lines += [
        "",
        f"**Top 10 stocks account for {top10_pct:.1f}% of total strategy return.**",
        "",
        "### Top 11-20 Contributors",
        "",
        "| Rank | Ticker | Total Contrib (%) | % of Total Return | Cum % | Periods Held |",
        "|------|--------|------------------|------------------|-------|--------------|",
    ]

    for _, row in contrib_df.iloc[10:20].iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['ticker'].replace('.NS','')} | "
            f"{row['total_contrib_%']:.2f}% | {row['pct_of_total']:.2f}% | "
            f"{row['cum_pct_of_total']:.1f}% | {int(row['n_periods_held'])} |"
        )

    lines += [
        "",
        f"**Top 20 stocks account for {top20_pct:.1f}% of total strategy return.**",
        "",
        "---",
        "",
        "## 3. Exclusion Sensitivity Test",
        "",
        f"| Scenario | CAGR | Sharpe | Sortino | Max DD | vs Baseline (CAGR) |",
        f"|----------|------|--------|---------|--------|-------------------|",
    ]

    baseline_cagr = baseline.get("CAGR (%)", np.nan)
    lines.append(
        f"| Baseline (all stocks) | {_fmt(baseline_cagr, '.1f', '%')} | "
        f"{_fmt(baseline.get('Sharpe'), '.3f')} | "
        f"{_fmt(baseline.get('Sortino'), '.3f')} | "
        f"{_fmt(baseline.get('Max Drawdown (%)'), '.1f', '%')} | -- |"
    )
    for r in exclusion_results:
        delta = r.get("CAGR (%)", np.nan) - baseline_cagr
        lines.append(
            f"| {r['label']} | {_fmt(r.get('CAGR (%)'), '.1f', '%')} | "
            f"{_fmt(r.get('Sharpe'), '.3f')} | "
            f"{_fmt(r.get('Sortino'), '.3f')} | "
            f"{_fmt(r.get('Max Drawdown (%)'), '.1f', '%')} | "
            f"{delta:+.1f}% |"
        )

    # Concentration summary
    lines += [
        "",
        "---",
        "",
        "## 4. Return Concentration Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Gini coefficient | **{gini:.3f}** |",
        f"| Interpretation | {gini_interpretation(gini)} |",
        f"| Top 10 stocks share | {top10_pct:.1f}% of total return |",
        f"| Top 20 stocks share | {top20_pct:.1f}% of total return |",
        f"| Top 50 stocks share | {top50_pct:.1f}% of total return |",
        f"| Stocks for 50% of return | {n_stocks_50} (of {n_total} ever held) |",
        f"| Stocks for 80% of return | {n_stocks_80} (of {n_total} ever held) |",
        "",
    ]

    # Verdict
    excl5_cagr  = next((r.get("CAGR (%)") for r in exclusion_results if "Top 5"  in r["label"]), np.nan)
    excl10_cagr = next((r.get("CAGR (%)") for r in exclusion_results if "Top 10" in r["label"]), np.nan)
    excl20_cagr = next((r.get("CAGR (%)") for r in exclusion_results if "Top 20" in r["label"]), np.nan)
    excl10_sh   = next((r.get("Sharpe")   for r in exclusion_results if "Top 10" in r["label"]), np.nan)
    excl20_sh   = next((r.get("Sharpe")   for r in exclusion_results if "Top 20" in r["label"]), np.nan)

    lines += [
        "---",
        "",
        "## 5. Assessment & Verdict",
        "",
        "### Is the strategy broadly distributed or winner-dependent?",
        "",
    ]

    if gini > 0.6:
        lines.append(
            f"The Gini coefficient of {gini:.3f} indicates HIGH-to-EXTREME return concentration. "
            f"A small number of stocks contribute disproportionately to total strategy gains."
        )
    elif gini > 0.4:
        lines.append(
            f"The Gini coefficient of {gini:.3f} indicates MODERATE return concentration. "
            f"The strategy has meaningful dependence on its top winners, but returns are not "
            f"entirely dominated by a handful of stocks."
        )
    else:
        lines.append(
            f"The Gini coefficient of {gini:.3f} indicates LOW concentration. "
            f"Returns are broadly distributed across many stocks."
        )

    lines += [
        "",
        f"Only **{n_stocks_50} stocks** (of {n_total} ever held) are needed to generate 50% of total return.",
        f"**{n_stocks_80} stocks** are needed to generate 80% of total return.",
        "",
        "### Robustness after winner exclusion",
        "",
    ]

    if not np.isnan(excl10_cagr) and excl10_cagr > 25 and not np.isnan(excl10_sh) and excl10_sh > 1.0:
        lines.append(
            f"After removing the top 10 contributors, the strategy still delivers "
            f"{excl10_cagr:.1f}% CAGR with Sharpe {excl10_sh:.3f}. "
            f"**The strategy is ROBUST to winner exclusion.**"
        )
    elif not np.isnan(excl10_cagr) and excl10_cagr > 15:
        lines.append(
            f"After removing the top 10 contributors, CAGR falls to {excl10_cagr:.1f}% "
            f"(from {baseline_cagr:.1f}%). Performance degrades but the strategy "
            f"remains **viable**."
        )
    else:
        lines.append(
            f"After removing the top 10 contributors, CAGR falls to {excl10_cagr:.1f}%. "
            f"The strategy shows **significant winner dependence**."
        )

    if not np.isnan(excl20_cagr) and excl20_cagr > 20:
        lines.append(
            f"Even after removing the top 20 contributors, CAGR is {excl20_cagr:.1f}% "
            f"(Sharpe {excl20_sh:.3f}). The factor continues to generate alpha "
            f"from the broader universe."
        )

    lines += [
        "",
        "### Survivorship bias implication",
        "",
        "The top contributors in this backtest are stocks that survived AND thrived over 2010-2026. "
        "In live trading:",
        "- Some future top-20 momentum stocks will delist, merge, or enter bear markets.",
        "- The realized concentration is likely HIGHER than what is observable ex-ante.",
        "- This makes the strategy more fragile than the backtest concentration implies.",
        "",
        "**Practical recommendation:** Monitor rolling 6-month Sharpe per stock. "
        "If a single stock accounts for >15% of month's portfolio return for 3+ months, "
        "consider a concentration cap (e.g., max 10% per stock) to reduce dependence on any single winner.",
        "",
        "---",
        "## Charts",
        "",
        "Saved to `research/charts/survivorship/`:",
        "",
        "- `top_contributors.png` -- bar chart of top-30 contributors with cumulative % overlay",
        "- `lorenz_concentration.png` -- Lorenz curve and contribution histogram",
        "- `exclusion_impact.png` -- equity curves and CAGR/Sharpe/MaxDD after exclusion",
        "- `contribution_treemap.png` -- return attribution by stock group + top-20 waterfall",
        "",
        "---",
        "*Generated by `research/survivorship_assessment.py`. Not investment advice.*",
    ]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", REPORT_PATH)
    return report


# ===============================================================================
# Main
# ===============================================================================

def main():
    Path("research").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research/survivorship_assessment.log",
                                mode="w", encoding="utf-8"),
        ],
    )

    end = str(pd.Timestamp.today().date())
    data_start = str(int(START[:4]) - 1) + START[4:]

    print("\n" + "=" * 70)
    print("  SURVIVORSHIP BIAS & RETURN CONCENTRATION ASSESSMENT")
    print("=" * 70)
    print(f"  Strategy : 12M lookback | skip 1M | Top 20 | Monthly")
    print(f"  Period   : {START} to {end}")
    print("=" * 70 + "\n")

    # ?? Load ??????????????????????????????????????????????????????????????????
    log.info("Loading data ...")
    tickers = get_nifty500_tickers(use_cache=True)
    adj_close, _, volume = download_price_data(tickers, start=data_start, end=END, use_cache=True)
    daily_returns = compute_daily_returns(adj_close)
    log.info("Data: %d tickers x %d days", adj_close.shape[1], adj_close.shape[0])

    # ?? Build baseline portfolio ???????????????????????????????????????????????
    log.info("Building baseline portfolio ...")
    mom = compute_mom_scores(adj_close, OPT["lookback"], OPT["skip"])
    rebal = get_rebal_dates(adj_close, START, END, OPT["freq"])
    constituents = _select_top_n(mom, rebal, OPT["top_n"])
    port_rets = _build_port(daily_returns, constituents, rebal)
    port_rets = port_rets.loc[START:end].dropna()

    baseline_metrics = compute_metrics(port_rets, label="Baseline")
    baseline_metrics["label"] = "Baseline"
    baseline_metrics["_returns"] = port_rets
    log.info("Baseline: CAGR=%.1f%% Sharpe=%.3f MaxDD=%.1f%%",
             baseline_metrics.get("CAGR (%)", 0),
             baseline_metrics.get("Sharpe", 0),
             baseline_metrics.get("Max Drawdown (%)", 0))

    # ?? Contribution analysis ??????????????????????????????????????????????????
    log.info("Computing per-stock contributions ...")
    contrib_df = compute_stock_contributions(daily_returns, constituents, START, end)
    log.info("Contribution analysis: %d stocks, %d positive, %d negative",
             len(contrib_df),
             (contrib_df["total_contrib"] > 0).sum(),
             (contrib_df["total_contrib"] < 0).sum())

    # Summary prints
    top10_pct = contrib_df.head(10)["pct_of_total"].sum()
    top20_pct = contrib_df.head(20)["pct_of_total"].sum()
    print(f"\n  TOP CONTRIBUTORS")
    print("  " + "-" * 70)
    print(f"  {'Rank':<5} {'Ticker':<16} {'Contrib%':>10} {'%ofTotal':>10} {'Cum%':>8} {'Periods':>8}")
    for _, row in contrib_df.head(20).iterrows():
        print(f"  {int(row['rank']):<5} {row['ticker'].replace('.NS',''):<16} "
              f"{row['total_contrib_%']:>10.2f}% {row['pct_of_total']:>10.2f}% "
              f"{row['cum_pct_of_total']:>8.1f}% {int(row['n_periods_held']):>8}")
    print(f"\n  Top 10 stocks = {top10_pct:.1f}% of total return")
    print(f"  Top 20 stocks = {top20_pct:.1f}% of total return")

    # ?? Exclusion tests ????????????????????????????????????????????????????????
    log.info("Running exclusion scenarios ...")
    excl_scenarios = [
        ("Excl Top 5",  5),
        ("Excl Top 10", 10),
        ("Excl Top 20", 20),
    ]
    exclusion_results = []
    for label, n in excl_scenarios:
        exclude_set = set(contrib_df.head(n)["ticker"].tolist())
        r = run_excluding(mom, daily_returns, rebal, exclude_set,
                          OPT["top_n"], START, end, label)
        exclusion_results.append(r)

    print(f"\n  EXCLUSION SENSITIVITY")
    print("  " + "-" * 65)
    print(f"  {'Scenario':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>9} {'Delta CAGR':>12}")
    base_c = baseline_metrics.get("CAGR (%)", 0)
    print(f"  {'Baseline':<18} {base_c:>7.1f}% "
          f"{baseline_metrics.get('Sharpe', 0):>8.3f} "
          f"{baseline_metrics.get('Max Drawdown (%)', 0):>8.1f}%  {'--':>11}")
    for r in exclusion_results:
        delta = r.get("CAGR (%)", 0) - base_c
        print(f"  {r['label']:<18} {r.get('CAGR (%)', 0):>7.1f}% "
              f"{r.get('Sharpe', 0):>8.3f} "
              f"{r.get('Max Drawdown (%)', 0):>8.1f}% {delta:>+11.1f}%")

    # ?? Lorenz + Gini ??????????????????????????????????????????????????????????
    log.info("Computing Lorenz curve and Gini coefficient ...")
    x_l, y_l, gini = lorenz_curve(contrib_df)
    print(f"\n  CONCENTRATION ANALYSIS")
    print("  " + "-" * 50)
    print(f"  Gini coefficient : {gini:.4f}")
    print(f"  Interpretation   : {gini_interpretation(gini)}")

    idx_50 = (contrib_df["cum_pct_of_total"] >= 50).idxmax()
    idx_80 = (contrib_df["cum_pct_of_total"] >= 80).idxmax()
    print(f"  Stocks for 50%% return : {idx_50 + 1} of {len(contrib_df)}")
    print(f"  Stocks for 80%% return : {idx_80 + 1} of {len(contrib_df)}")

    # ?? Charts ?????????????????????????????????????????????????????????????????
    log.info("Generating charts ...")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    try:
        plot_top_contributors(contrib_df, top_n=30)
        plot_lorenz(contrib_df, x_l, y_l, gini)
        plot_exclusion_impact(baseline_metrics, exclusion_results, contrib_df)
        plot_contribution_treemap(contrib_df, top_n=40)
        log.info("Charts saved to %s", CHART_DIR)
    except Exception as e:
        log.warning("Chart error: %s", e, exc_info=True)

    # ?? Report ?????????????????????????????????????????????????????????????????
    log.info("Generating report ...")
    report = generate_report(
        contrib_df, baseline_metrics, exclusion_results,
        gini, x_l, y_l, START, end
    )

    print("\n" + "=" * 70)
    print(f"  COMPLETE")
    print(f"  Report : {REPORT_PATH}")
    print(f"  Charts : {CHART_DIR}/")
    print("=" * 70 + "\n")

    lines = report.split("\n")
    safe = "\n".join(l.encode("ascii", "replace").decode("ascii") for l in lines[:80])
    print(safe)
    extra = len(lines) - 80
    if extra > 0:
        print(f"\n... [{extra} more lines in {REPORT_PATH}]")


if __name__ == "__main__":
    main()
