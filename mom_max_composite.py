"""
research/mom_max_composite.py
------------------------------
Momentum + MAX Composite Strategy Study – NIFTY 500

Hypothesis:
  Momentum_12M finds winners.
  MAX (negative) removes lottery-like stocks from the winner set.
  Combining them should improve the quality of the momentum portfolio.

Composite score formula:
  composite = w_mom * pct_rank(MOM_12M) + w_max * (1 - pct_rank(MAX))

  pct_rank(MOM)         : 0=worst, 1=best momentum
  1 - pct_rank(MAX)     : 0=highest lottery, 1=lowest lottery (inverted)
  w_mom + w_max = 1.0

Weightings tested:
  W1  100% / 0%    pure momentum
  W2   80% / 20%   momentum-dominant
  W3   70% / 30%   moderate MAX overlay
  W4   50% / 50%   equal blend

Portfolio: top 20 stocks, equal weight, monthly rebalance.

Parameter heatmap: sweeps w_mom across [0.0, 0.1, ..., 1.0] x top-N [10, 15, 20, 25, 30].

Usage:
    python research/mom_max_composite.py
    python research/mom_max_composite.py --start 2012-01-01
    python research/mom_max_composite.py --no-charts
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
from factor_engine import compute_daily_returns, compute_max_factor
from portfolio import get_rebalance_dates, build_portfolio_returns
from performance import compute_metrics, TRADING_DAYS, RISK_FREE_RATE

from factors import compute_momentum_12m, compute_max_scores

log = logging.getLogger(__name__)

CHART_DIR = Path("research/charts/mom_max_composite")
REPORT_PATH = Path("research/mom_max_composite_report.md")

# The 4 specified weight combos (mom_weight, max_weight, label)
WEIGHT_SPECS = [
    (1.00, 0.00, "W1_Mom100"),
    (0.80, 0.20, "W2_Mom80_MAX20"),
    (0.70, 0.30, "W3_Mom70_MAX30"),
    (0.50, 0.50, "W4_Mom50_MAX50"),
]

COLORS = {
    "W1_Mom100":      "#1f77b4",   # blue
    "W2_Mom80_MAX20": "#ff7f0e",   # orange
    "W3_Mom70_MAX30": "#2ca02c",   # green
    "W4_Mom50_MAX50": "#d62728",   # red
    "NIFTY50":        "#555555",   # grey
    "NIFTY500":       "#aaaaaa",   # light grey
}


# ═══════════════════════════════════════════════════════════════════════════════
# Factor score helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank: 0 (worst) to 1 (best). NaN-safe."""
    valid = s.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    ranked = valid.rank(method="average") / len(valid)
    return ranked.reindex(s.index)


def compute_composite(
    mom_df: pd.DataFrame,
    max_df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    mom_weight: float,
) -> pd.DataFrame:
    """
    Compute composite score at each rebalance date.

    composite = mom_weight * pct_rank(MOM) + (1 - mom_weight) * (1 - pct_rank(MAX))

    Higher composite = stronger momentum AND/OR lower lottery exposure.
    When mom_weight=1: pure momentum rank.
    When mom_weight=0: pure inverted-MAX rank (lowest lottery stocks).
    """
    max_weight = 1.0 - mom_weight
    scores: dict = {}

    for date in rebalance_dates:
        mom_row = mom_df.loc[date] if date in mom_df.index else pd.Series(dtype=float)
        max_row = max_df.loc[date] if date in max_df.index else pd.Series(dtype=float)

        if mom_weight == 1.0:
            # Pure momentum — only MOM ranks, MAX not required
            rank_m = _pct_rank(mom_row)
            scores[date] = rank_m
            continue

        if mom_weight == 0.0:
            # Pure inverted MAX — only MAX ranks, MOM not required
            rank_x = _pct_rank(max_row)
            scores[date] = 1.0 - rank_x
            continue

        # Both required — only score stocks present in both
        common = mom_row.dropna().index.intersection(max_row.dropna().index)
        if len(common) < 5:
            scores[date] = pd.Series(dtype=float)
            continue

        rank_m = _pct_rank(mom_row.loc[common])
        rank_x = _pct_rank(max_row.loc[common])
        composite = mom_weight * rank_m + max_weight * (1.0 - rank_x)
        scores[date] = composite

    df = pd.DataFrame(scores).T
    df.index.name = "date"
    return df


def build_top_n_constituents(
    composite_df: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    top_n: int,
) -> dict:
    """Select top N stocks by composite score at each rebalance date."""
    constituents: dict = {}
    for date in rebalance_dates:
        if date not in composite_df.index:
            constituents[date] = []
            continue
        row = composite_df.loc[date].dropna()
        sel = list(row.nlargest(top_n).index) if len(row) >= top_n else list(row.index)
        constituents[date] = sel
    return constituents


def compute_turnover(constituents: dict) -> float:
    dates = sorted(constituents.keys())
    monthly = []
    for i in range(1, len(dates)):
        prev = set(constituents[dates[i - 1]])
        curr = set(constituents[dates[i]])
        if curr:
            monthly.append(len(curr - prev) / len(curr))
    return round(np.mean(monthly) * 100, 1) if monthly else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
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


def _dd_series(r: pd.Series) -> pd.Series:
    cum = (1 + r.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def _rolling_sharpe(r: pd.Series, window: int = 252) -> pd.Series:
    rf_d = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    exc = r - rf_d
    return (exc.rolling(window).mean() / exc.rolling(window).std()) * np.sqrt(TRADING_DAYS)


def plot_equity_curves(returns_dict: dict, title: str = "Equity Curves") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(15, 7))
    for label, r in returns_dict.items():
        cum = (1 + r.fillna(0)).cumprod()
        color = COLORS.get(label, "#888888")
        lw = 2.2 if label.startswith("W") else 1.5
        ls = "--" if label in ("NIFTY50", "NIFTY500") else "-"
        ax.plot(cum.index, cum.values, label=label, color=color, lw=lw, ls=ls)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylabel("Wealth (log scale, starting at 1x)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "equity_curves.png")


def plot_drawdown(returns_dict: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(15, 6))
    for label, r in returns_dict.items():
        dd = _dd_series(r)
        color = COLORS.get(label, "#888888")
        lw = 2.0 if label.startswith("W") else 1.4
        ls = "--" if label in ("NIFTY50", "NIFTY500") else "-"
        ax.plot(dd.index, dd.values, label=label, color=color, lw=lw, ls=ls)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Drawdown Comparison", fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncol=2, framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "drawdown.png")


def plot_rolling_sharpe(returns_dict: dict, window: int = 252) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(15, 6))
    for label, r in returns_dict.items():
        rs = _rolling_sharpe(r, window)
        color = COLORS.get(label, "#888888")
        lw = 2.0 if label.startswith("W") else 1.4
        ls = "--" if label in ("NIFTY50", "NIFTY500") else "-"
        ax.plot(rs.index, rs.values, label=label, color=color, lw=lw, ls=ls)
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.axhline(1, color="#2ca02c", lw=0.6, ls=":", alpha=0.7)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontweight="bold", pad=12)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    _year_ax(ax)
    fig.tight_layout()
    _save(fig, CHART_DIR / "rolling_sharpe.png")


def plot_parameter_heatmaps(grid_results: pd.DataFrame) -> None:
    """
    5 heatmaps in one figure.
    Rows = mom_weight (0.0 to 1.0), Cols = top_n (10..30).
    One panel per metric: CAGR, Sharpe, Sortino, Calmar, MaxDD.
    """
    _style()

    metrics_config = [
        ("CAGR (%)",          "CAGR (%)",      "RdYlGn",  False),
        ("Sharpe",            "Sharpe",         "RdYlGn",  False),
        ("Sortino",           "Sortino",        "RdYlGn",  False),
        ("Calmar",            "Calmar",         "RdYlGn",  False),
        ("Max Drawdown (%)",  "Max DD (%)",     "RdYlGn_r", False),
    ]

    mom_weights = sorted(grid_results["mom_weight"].unique())
    top_ns = sorted(grid_results["top_n"].unique())

    fig, axes = plt.subplots(1, 5, figsize=(24, 7))
    fig.suptitle(
        "Parameter Heatmap: Momentum Weight x Portfolio Size\n"
        "Composite = w * pct_rank(MOM12M) + (1-w) * (1 - pct_rank(MAX))",
        fontweight="bold", fontsize=13,
    )

    for ax, (metric_col, title, cmap, invert) in zip(axes, metrics_config):
        mat = np.full((len(mom_weights), len(top_ns)), np.nan)
        for i, mw in enumerate(mom_weights):
            for j, tn in enumerate(top_ns):
                row = grid_results[
                    (grid_results["mom_weight"] == mw) & (grid_results["top_n"] == tn)
                ]
                if not row.empty and metric_col in row.columns:
                    mat[i, j] = row[metric_col].values[0]

        # Robust color scaling
        valid = mat[~np.isnan(mat)]
        vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
        if vmin == vmax:
            vmin -= 0.01; vmax += 0.01

        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                       origin="upper")

        ax.set_xticks(range(len(top_ns)))
        ax.set_xticklabels([f"N={n}" for n in top_ns], fontsize=8)
        ax.set_yticks(range(len(mom_weights)))
        ax.set_yticklabels([f"{int(mw*100)}%" for mw in mom_weights], fontsize=8)
        ax.set_xlabel("Portfolio Size (top-N)", fontsize=9)
        ax.set_ylabel("Momentum Weight", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11)

        # Annotate cells
        for i in range(len(mom_weights)):
            for j in range(len(top_ns)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7.5,
                            color="black" if abs(v - (vmin + vmax) / 2) < (vmax - vmin) * 0.4
                            else "white")

        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

        # Mark the 4 specified combos with a border
        specified_mw = [1.00, 0.80, 0.70, 0.50]
        for mw in specified_mw:
            if mw in mom_weights:
                i = mom_weights.index(mw)
                # find top_n=20
                if 20 in top_ns:
                    j = top_ns.index(20)
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor="white", lw=2.0, zorder=5)
                    ax.add_patch(rect)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, CHART_DIR / "parameter_heatmaps.png")


def plot_sharpe_heatmap_focused(grid_results: pd.DataFrame) -> None:
    """
    A large, readable Sharpe heatmap as a standalone chart.
    Highlights the 4 specified weight combinations at top-N=20.
    """
    _style()
    mom_weights = sorted(grid_results["mom_weight"].unique())
    top_ns = sorted(grid_results["top_n"].unique())

    mat = np.full((len(mom_weights), len(top_ns)), np.nan)
    for i, mw in enumerate(mom_weights):
        for j, tn in enumerate(top_ns):
            row = grid_results[
                (grid_results["mom_weight"] == mw) & (grid_results["top_n"] == tn)
            ]
            if not row.empty and "Sharpe" in row.columns:
                mat[i, j] = row["Sharpe"].values[0]

    valid = mat[~np.isnan(mat)]
    vmin, vmax = valid.min() - 0.01, valid.max() + 0.01

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax, origin="upper")

    ax.set_xticks(range(len(top_ns)))
    ax.set_xticklabels([f"N={n}" for n in top_ns], fontsize=10)
    ax.set_yticks(range(len(mom_weights)))
    ax.set_yticklabels([f"{int(mw*100)}% MOM / {int((1-mw)*100)}% MAX"
                        for mw in mom_weights], fontsize=10)
    ax.set_xlabel("Portfolio Size (top-N stocks)", fontsize=11)
    ax.set_title(
        "Sharpe Ratio — Momentum + MAX Composite\n"
        "composite = w*pct_rank(MOM) + (1-w)*(1-pct_rank(MAX))",
        fontweight="bold", fontsize=12, pad=12,
    )

    specified_mw = [1.00, 0.80, 0.70, 0.50]
    for i, mw in enumerate(mom_weights):
        for j, tn in enumerate(top_ns):
            v = mat[i, j]
            if not np.isnan(v):
                text_col = "black" if (v - vmin) / (vmax - vmin) < 0.7 else "white"
                weight = 2.5 if (mw in specified_mw and tn == 20) else 1.0
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if weight > 1 else "normal",
                        color=text_col)

    # Box around specified combos at N=20
    if 20 in top_ns:
        j20 = top_ns.index(20)
        for mw in specified_mw:
            if mw in mom_weights:
                i_mw = mom_weights.index(mw)
                rect = plt.Rectangle((j20 - 0.5, i_mw - 0.5), 1, 1,
                                     fill=False, edgecolor="navy", lw=2.5, zorder=5)
                ax.add_patch(rect)
        ax.text(j20, -0.7, "Specified\ncombos", ha="center", va="bottom",
                fontsize=8.5, color="navy", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.8)
    fig.tight_layout()
    _save(fig, CHART_DIR / "sharpe_heatmap.png")


def plot_weight_sensitivity(grid_results: pd.DataFrame, top_n: int = 20) -> None:
    """Line charts: metric vs momentum weight at fixed top_n=20."""
    _style()
    subset = grid_results[grid_results["top_n"] == top_n].sort_values("mom_weight")

    metrics = [("CAGR (%)", "CAGR (%)"), ("Sharpe", "Sharpe Ratio"),
               ("Sortino", "Sortino Ratio"), ("Calmar", "Calmar Ratio"),
               ("Max Drawdown (%)", "Max Drawdown (%)")]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Metric Sensitivity to Momentum Weight (Portfolio Size = Top {top_n})",
                 fontweight="bold", fontsize=13)

    # mark specified combos
    specified_x = [1.00, 0.80, 0.70, 0.50]

    for ax, (col, ylabel) in zip(axes, metrics):
        x = subset["mom_weight"].values
        y = subset[col].values
        ax.plot(x * 100, y, "o-", color="#1f77b4", lw=2, ms=5, zorder=3)

        # Highlight the 4 specified combos
        for mw in specified_x:
            row = subset[subset["mom_weight"] == mw]
            if not row.empty:
                yv = row[col].values[0]
                ax.scatter([mw * 100], [yv], s=80, color="#d62728", zorder=5)
                ax.annotate(f"{yv:.2f}", (mw * 100, yv),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color="#d62728", fontweight="bold")

        ax.set_xlabel("Momentum Weight (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.set_xlim(-5, 105)

        # Add vertical lines for the 4 combos
        for mw in specified_x:
            ax.axvline(mw * 100, color="#d62728", lw=0.7, ls="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, CHART_DIR / "weight_sensitivity.png")


def plot_dashboard(returns_dict: dict, metrics_list: list) -> None:
    """4-panel summary dashboard."""
    _style()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle("Momentum + MAX Composite – Summary Dashboard",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    for label, r in returns_dict.items():
        cum = (1 + r.fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, color=COLORS.get(label, "#888"),
                 lw=2.0 if label.startswith("W") else 1.3,
                 ls="--" if label in ("NIFTY50", "NIFTY500") else "-",
                 label=label)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_title("Equity Curves (log scale)", fontweight="bold")
    ax1.legend(ncol=2, fontsize=7.5)
    _year_ax(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = [m["Label"] for m in metrics_list]
    cagrs = [m.get("CAGR (%)", np.nan) for m in metrics_list]
    colors_ = [COLORS.get(l, "#555") for l in labels]
    bars = ax2.bar(range(len(labels)), cagrs, color=colors_, edgecolor="white")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_title("CAGR (%)", fontweight="bold")
    for b, v in zip(bars, cagrs):
        if not np.isnan(v):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                     f"{v:.1f}%", ha="center", fontsize=7.5)

    ax3 = fig.add_subplot(gs[1, 0])
    for label, r in returns_dict.items():
        dd = _dd_series(r)
        ax3.plot(dd.index, dd.values, color=COLORS.get(label, "#888"),
                 lw=1.8 if label.startswith("W") else 1.2,
                 ls="--" if label in ("NIFTY50", "NIFTY500") else "-",
                 label=label)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax3.set_title("Drawdown", fontweight="bold")
    ax3.legend(ncol=2, fontsize=7.5, loc="lower left")
    _year_ax(ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    sharpes = [m.get("Sharpe", np.nan) for m in metrics_list]
    ax4.bar(range(len(labels)), sharpes, color=colors_, edgecolor="white")
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax4.set_title("Sharpe Ratio", fontweight="bold")
    ax4.axhline(0, color="black", lw=0.5)

    _save(fig, CHART_DIR / "dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt=".2f", suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:{fmt}}{suf}"


def generate_report(
    metrics_list: list,
    grid_results: pd.DataFrame,
    backtest_start: str,
    backtest_end: str,
    top_n: int,
) -> str:
    import datetime
    run_date = datetime.date.today().isoformat()

    # Best cell in heatmap
    if not grid_results.empty:
        best_sharpe_row = grid_results.loc[grid_results["Sharpe"].idxmax()]
        best_cagr_row = grid_results.loc[grid_results["CAGR (%)"].idxmax()]
        best_dd_row = grid_results.loc[grid_results["Max Drawdown (%)"].idxmax()]
    else:
        best_sharpe_row = best_cagr_row = best_dd_row = {}

    # Extract metrics for the 4 combos
    method_metrics = {m["Label"]: m for m in metrics_list if m["Label"].startswith("W")}

    lines = [
        "# Momentum + MAX Composite Strategy Study",
        "",
        f"**Universe:** NIFTY 500 (current constituents)  ",
        f"**Period:** {backtest_start} to {backtest_end}  ",
        f"**Generated:** {run_date}  ",
        f"**Portfolio:** Top {top_n} stocks, equal weight, monthly rebalance  ",
        "",
        "**Composite score formula:**",
        "```",
        "composite = w * pct_rank(MOM_12M) + (1-w) * (1 - pct_rank(MAX))",
        "",
        "pct_rank(MOM_12M)    : 0=worst 12M momentum, 1=best",
        "1 - pct_rank(MAX)    : 0=highest lottery exposure, 1=lowest",
        "Select top N stocks by composite score.",
        "```",
        "",
        "**Hypothesis:** MAX identifies lottery-like behavior. Removing high-MAX stocks",
        "from a momentum portfolio should reduce crash risk and smooth returns.",
        "",
        "---",
        "",
        "## Performance Summary (Top 20 Stocks)",
        "",
        "| Portfolio | CAGR | Ann Vol | Sharpe | Sortino | Calmar | Max DD | Turnover |",
        "|-----------|------|---------|--------|---------|--------|--------|----------|",
    ]

    for m in metrics_list:
        lines.append(
            f"| **{m['Label']}** | "
            f"{_fmt(m.get('CAGR (%)'), '.1f', '%')} | "
            f"{_fmt(m.get('Ann. Vol (%)'), '.1f', '%')} | "
            f"{_fmt(m.get('Sharpe'), '.3f')} | "
            f"{_fmt(m.get('Sortino'), '.3f')} | "
            f"{_fmt(m.get('Calmar'), '.3f')} | "
            f"{_fmt(m.get('Max Drawdown (%)'), '.1f', '%')} | "
            f"{_fmt(m.get('Turnover (% /mo)'), '.1f', '%/mo')} |"
        )

    # Analysis section
    w1 = method_metrics.get("W1_Mom100", {})
    w2 = method_metrics.get("W2_Mom80_MAX20", {})
    w3 = method_metrics.get("W3_Mom70_MAX30", {})
    w4 = method_metrics.get("W4_Mom50_MAX50", {})

    w1_sh = w1.get("Sharpe", np.nan)
    w2_sh = w2.get("Sharpe", np.nan)
    w3_sh = w3.get("Sharpe", np.nan)
    w4_sh = w4.get("Sharpe", np.nan)

    # Count how many MAX-overlaid combos improve Sharpe
    baseline_sh = w1_sh
    improved = sum(1 for sh in [w2_sh, w3_sh, w4_sh]
                   if not np.isnan(sh) and not np.isnan(baseline_sh) and sh > baseline_sh)

    lines += [
        "",
        "---",
        "",
        "## Analysis: Does MAX Improve Momentum?",
        "",
        "### Sharpe Ratio Progression",
        "",
        f"| Weight | Sharpe | vs Pure Momentum |",
        f"|--------|--------|-----------------|",
        f"| W1: 100% MOM | {_fmt(w1_sh, '.3f')} | baseline |",
        f"| W2: 80/20 | {_fmt(w2_sh, '.3f')} | {_fmt(w2_sh - w1_sh if not np.isnan(w2_sh) and not np.isnan(w1_sh) else np.nan, '+.3f')} |",
        f"| W3: 70/30 | {_fmt(w3_sh, '.3f')} | {_fmt(w3_sh - w1_sh if not np.isnan(w3_sh) and not np.isnan(w1_sh) else np.nan, '+.3f')} |",
        f"| W4: 50/50 | {_fmt(w4_sh, '.3f')} | {_fmt(w4_sh - w1_sh if not np.isnan(w4_sh) and not np.isnan(w1_sh) else np.nan, '+.3f')} |",
        "",
    ]

    if improved == 3:
        verdict = (
            "**VERDICT: MAX CONSISTENTLY IMPROVES MOMENTUM.** All 3 MAX-overlaid combinations "
            "beat pure momentum on Sharpe ratio. Removing high-lottery stocks from momentum "
            "portfolios reduces volatility faster than it reduces returns."
        )
    elif improved == 2:
        verdict = (
            "**VERDICT: MAX PARTIALLY IMPROVES MOMENTUM.** 2 of 3 MAX-overlaid combinations "
            "beat pure momentum on Sharpe. Moderate MAX weighting (20-30%) appears beneficial; "
            "heavy MAX weighting (50%) dilutes the momentum signal too much."
        )
    elif improved == 1:
        verdict = (
            "**VERDICT: MAX MARGINALLY IMPROVES MOMENTUM.** Only 1 of 3 MAX-overlaid "
            "combinations beats pure momentum on Sharpe. The optimal MAX weight is narrow."
        )
    else:
        verdict = (
            "**VERDICT: MAX DOES NOT IMPROVE MOMENTUM.** None of the MAX-overlaid combinations "
            "beat pure momentum on Sharpe. Removing high-MAX stocks costs more in momentum "
            "return than it saves in volatility reduction."
        )

    lines += [verdict, ""]

    # Heatmap best cell
    if not grid_results.empty:
        bsr = best_sharpe_row
        lines += [
            "### Parameter Heatmap Findings",
            "",
            f"- **Best Sharpe in full grid:** {_fmt(bsr.get('Sharpe', np.nan), '.3f')} "
            f"at MOM weight = {_fmt(bsr.get('mom_weight', np.nan)*100 if isinstance(bsr, dict) or hasattr(bsr, 'get') else np.nan, '.0f', '%')}, "
            f"top-N = {int(bsr.get('top_n', 0)) if hasattr(bsr, 'get') else '--'}",
            f"- **Best CAGR in full grid:** {_fmt(best_cagr_row.get('CAGR (%)', np.nan), '.1f', '%')} "
            f"at MOM weight = {_fmt(best_cagr_row.get('mom_weight', np.nan)*100 if hasattr(best_cagr_row, 'get') else np.nan, '.0f', '%')}, "
            f"top-N = {int(best_cagr_row.get('top_n', 0)) if hasattr(best_cagr_row, 'get') else '--'}",
            f"- **Smallest Max DD:** {_fmt(best_dd_row.get('Max Drawdown (%)', np.nan), '.1f', '%')} "
            f"at MOM weight = {_fmt(best_dd_row.get('mom_weight', np.nan)*100 if hasattr(best_dd_row, 'get') else np.nan, '.0f', '%')}, "
            f"top-N = {int(best_dd_row.get('top_n', 0)) if hasattr(best_dd_row, 'get') else '--'}",
            "",
        ]

    lines += [
        "### Interpretation",
        "",
        "- **Momentum** is the dominant signal. The 12-month return captures stocks in "
        "persistent uptrends — a well-documented effect in Indian equities.",
        "- **MAX as a negative screen** acts as a quality/volatility filter. It removes stocks "
        "whose recent gains came from a few explosive days (lottery-like behavior). These stocks "
        "tend to mean-revert sharply.",
        "- **The combo works if** MAX is removing momentum stocks that are driven by one-off "
        "spikes rather than sustained price appreciation. In that case, drawdowns narrow without "
        "losing the genuine momentum leaders.",
        "- **The combo fails if** high-MAX stocks are genuinely the best momentum stocks "
        "(the large price jump IS the momentum). In India's bull market, this is often true for "
        "high-growth mid/small caps.",
        "- **Turnover**: the MAX component tends to increase turnover because MAX scores are "
        "volatile month-to-month. Expect ~5-10% higher turnover for W2-W4 vs W1.",
        "",
        "---",
        "",
        "## Charts",
        "",
        "Saved to `research/charts/mom_max_composite/`:",
        "",
        "- `dashboard.png` -- 4-panel summary",
        "- `equity_curves.png` -- All methods + benchmarks",
        "- `drawdown.png` -- Drawdown comparison",
        "- `rolling_sharpe.png` -- Rolling 1-year Sharpe",
        "- `sharpe_heatmap.png` -- Large readable Sharpe heatmap with combo highlights",
        "- `parameter_heatmaps.png` -- 5-metric heatmaps side-by-side",
        "- `weight_sensitivity.png` -- Metric vs momentum weight at top-N=20",
        "",
        "---",
        "*Generated by `research/mom_max_composite.py`. Not investment advice.*",
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
    p = argparse.ArgumentParser(description="Momentum + MAX Composite Study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=20, help="Primary portfolio size")
    p.add_argument("--min-adtv", type=float, default=1.0)
    p.add_argument("--no-charts", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path("research").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research/mom_max_composite.log", mode="w", encoding="utf-8"),
        ],
    )

    print("\n" + "=" * 70)
    print("  MOMENTUM + MAX COMPOSITE STRATEGY STUDY")
    print("=" * 70)
    print(f"  Period   : {args.start} -> {args.end or 'today'}")
    print(f"  Universe : NIFTY 500")
    print(f"  Portfolio: Top {args.top_n} stocks, equal weight")
    print(f"  Combos   : 100/0 | 80/20 | 70/30 | 50/50  (MOM/MAX)")
    print("=" * 70 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Loading cached price data ...")
    tickers = get_nifty500_tickers(use_cache=True)
    data_start = str(int(args.start[:4]) - 1) + args.start[4:]
    adj_close, _, volume = download_price_data(
        tickers, start=data_start, end=args.end, use_cache=True
    )
    benchmarks_prices = load_benchmark_data(start=data_start, end=args.end)
    daily_returns = compute_daily_returns(adj_close)
    rebalance_dates = get_rebalance_dates(adj_close, start=args.start, end=args.end)
    log.info("Data: %d tickers, %d days, %d rebalance dates",
             adj_close.shape[1], adj_close.shape[0], len(rebalance_dates))

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end) if args.end else pd.Timestamp.today()

    def trim(s: pd.Series) -> pd.Series:
        return s.loc[start_dt:end_dt].dropna()

    # ── 2. Compute factor scores ──────────────────────────────────────────────
    log.info("Computing MOM_12M factor scores ...")
    mom_df = compute_momentum_12m(adj_close, rebalance_dates)

    log.info("Computing MAX factor scores ...")
    max_df = compute_max_scores(adj_close, volume, rebalance_dates,
                                min_adtv_crore=args.min_adtv)

    # ── 3. Build the 4 specified portfolios ───────────────────────────────────
    portfolio_returns: dict[str, pd.Series] = {}
    portfolio_constituents: dict[str, dict] = {}
    metrics_list: list = []

    for mom_w, max_w, label in WEIGHT_SPECS:
        log.info("Building portfolio: %s (MOM=%.0f%%, MAX=%.0f%%) ...",
                 label, mom_w * 100, max_w * 100)
        composite_df = compute_composite(mom_df, max_df, rebalance_dates, mom_weight=mom_w)
        constituents = build_top_n_constituents(composite_df, rebalance_dates, top_n=args.top_n)
        r = trim(build_portfolio_returns(daily_returns, constituents, rebalance_dates, label=label))
        portfolio_returns[label] = r
        portfolio_constituents[label] = constituents
        turnover = compute_turnover(constituents)
        m = compute_metrics(r, label=label)
        m["Turnover (% /mo)"] = turnover
        metrics_list.append(m)

    # ── 4. Benchmarks ─────────────────────────────────────────────────────────
    all_returns = dict(portfolio_returns)
    bench_labels = []
    for bname in ("NIFTY50", "NIFTY500"):
        if bname in benchmarks_prices:
            br = trim(benchmarks_prices[bname].pct_change().dropna())
            br.name = bname
            all_returns[bname] = br
            bm = compute_metrics(br, label=bname)
            bm["Turnover (% /mo)"] = np.nan
            metrics_list.append(bm)
            bench_labels.append(bname)

    # ── 5. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    hdr = f"  {'Portfolio':<20} {'CAGR':>7} {'Vol':>6} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD':>8} {'TO%/mo':>8}"
    print(hdr)
    print("  " + "-" * 77)
    for m in metrics_list:
        def f(k, fmt=".1f"):
            v = m.get(k, float("nan"))
            return f"{v:{fmt}}" if v == v else "--"
        print(f"  {m['Label']:<20} {f('CAGR (%)')+' ':>8} {f('Ann. Vol (%)')+' ':>7} "
              f"{f('Sharpe','.3f'):>8} {f('Sortino','.3f'):>8} {f('Calmar','.3f'):>8} "
              f"{f('Max Drawdown (%)')+' ':>9} {f('Turnover (% /mo)'):>8}")
    print("=" * 90 + "\n")

    # ── 6. Parameter grid sweep ───────────────────────────────────────────────
    mom_weight_grid = [round(w / 10, 1) for w in range(0, 11)]  # 0.0 to 1.0
    top_n_grid = [10, 15, 20, 25, 30]

    log.info("Running parameter grid: %d mom_weights x %d top_n = %d cells ...",
             len(mom_weight_grid), len(top_n_grid), len(mom_weight_grid) * len(top_n_grid))

    grid_rows = []
    total_cells = len(mom_weight_grid) * len(top_n_grid)
    done = 0

    # Cache composite DFs per mom_weight to avoid recomputing
    composite_cache: dict[float, pd.DataFrame] = {}
    for mw in mom_weight_grid:
        log.info("  Grid: computing composite for MOM=%.0f%% ...", mw * 100)
        composite_cache[mw] = compute_composite(mom_df, max_df, rebalance_dates, mom_weight=mw)

    for mw in mom_weight_grid:
        comp_df = composite_cache[mw]
        for tn in top_n_grid:
            label_g = f"MOM{int(mw*100)}_N{tn}"
            consts = build_top_n_constituents(comp_df, rebalance_dates, top_n=tn)
            r = trim(build_portfolio_returns(daily_returns, consts, rebalance_dates, label=label_g))
            if r.empty:
                continue
            m = compute_metrics(r, label=label_g)
            m["mom_weight"] = mw
            m["top_n"] = tn
            m["Turnover (% /mo)"] = compute_turnover(consts)
            grid_rows.append(m)
            done += 1
            if done % 10 == 0:
                log.info("  Grid: %d/%d cells done", done, total_cells)

    grid_results = pd.DataFrame(grid_rows)
    grid_results.to_csv("research/mom_max_grid_results.csv", index=False)
    log.info("Grid results saved: research/mom_max_grid_results.csv (%d rows)", len(grid_results))

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    if not args.no_charts:
        log.info("Generating charts ...")
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        try:
            plot_equity_curves(all_returns,
                               title="Momentum + MAX Composite – Equity Curves (log scale)")
            plot_drawdown(all_returns)
            plot_rolling_sharpe(all_returns)
            plot_dashboard(all_returns, metrics_list)
            if not grid_results.empty:
                plot_parameter_heatmaps(grid_results)
                plot_sharpe_heatmap_focused(grid_results)
                plot_weight_sensitivity(grid_results, top_n=args.top_n)
            log.info("Charts saved to %s", CHART_DIR)
        except Exception as e:
            log.warning("Chart error: %s", e)

    # ── 8. Report ─────────────────────────────────────────────────────────────
    log.info("Generating report ...")
    report = generate_report(
        metrics_list, grid_results, args.start,
        args.end or str(end_dt.date()), args.top_n,
    )

    print("=" * 70)
    print("  COMPLETE")
    print(f"  Report  : {REPORT_PATH}")
    print(f"  Charts  : {CHART_DIR}/")
    print(f"  Grid    : research/mom_max_grid_results.csv")
    print(f"  Log     : research/mom_max_composite.log")
    print("=" * 70 + "\n")

    preview = "\n".join(
        l.encode("ascii", "replace").decode("ascii")
        for l in report.split("\n")[:55]
    )
    print(preview)
    extra = len(report.split("\n")) - 55
    if extra > 0:
        print(f"\n... [{extra} more lines in {REPORT_PATH}]")


if __name__ == "__main__":
    main()
