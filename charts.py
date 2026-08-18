"""
research/charts.py
------------------
All visualizations for the factor research framework.

Charts saved to research/charts/<factor_name>/*.png
"""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CHART_ROOT = Path("research/charts")
DPI = 150

# 10-color palette: D1=deep blue → D10=deep red
DECILE_COLORS = [
    "#053061", "#2166ac", "#4393c3", "#74add1", "#abd9e9",
    "#fdae61", "#f46d43", "#d73027", "#b2182b", "#67001f",
]

FACTOR_COLORS = {
    "MAX": "#9013fe",
    "MOMENTUM_6M": "#1a6faf",
    "MOMENTUM_12M": "#2ca02c",
    "LOW_VOL": "#17becf",
    "QUALITY": "#e6820a",
    "VALUE": "#d62728",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    log.info("Saved: %s", path)
    plt.close(fig)


def _year_fmt(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ──────────────────────────────────────────────────────────────────────────────
# Per-factor charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_decile_equity_curves(
    factor_name: str,
    decile_returns: dict[int, pd.Series],
    save_dir: Path = None,
) -> None:
    """All 10 decile equity curves on one log-scale chart."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT / factor_name.lower()
    fig, ax = plt.subplots(figsize=(14, 7))

    for d in sorted(decile_returns):
        r = decile_returns[d]
        cum = (1 + r.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, color=DECILE_COLORS[d - 1], lw=1.5, label=f"D{d}")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_title(f"{factor_name} – All Decile Equity Curves (log scale)", fontweight="bold", pad=12)
    ax.set_ylabel("Wealth (starting at 1x)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    _year_fmt(ax)
    fig.tight_layout()
    _save(fig, save_dir / "decile_equity_curves.png")


def plot_decile_bars(
    factor_name: str,
    metrics_df: pd.DataFrame,
    save_dir: Path = None,
) -> None:
    """3-panel bar chart: CAGR, Sharpe, and Ann. Vol by decile."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT / factor_name.lower()

    panels = [
        ("CAGR (%)", "CAGR (%)"),
        ("Sharpe", "Sharpe Ratio"),
        ("Ann. Vol (%)", "Ann. Vol (%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{factor_name} – Decile Performance", fontweight="bold", fontsize=14, y=1.01)

    for ax, (col, label) in zip(axes, panels):
        if col not in metrics_df.columns:
            ax.set_visible(False)
            continue
        vals = metrics_df[col].sort_index()
        colors = [DECILE_COLORS[int(d) - 1] for d in vals.index]
        bars = ax.bar(
            [f"D{int(d)}" for d in vals.index], vals.values,
            color=colors, edgecolor="white", linewidth=0.5,
        )
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Decile (D1=Low, D10=High)")
        for bar, v in zip(bars, vals.values):
            if not np.isnan(v):
                ypos = bar.get_height() + (0.2 if bar.get_height() >= 0 else -1.2)
                ax.text(
                    bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                )

    fig.tight_layout()
    _save(fig, save_dir / "decile_bars.png")


def plot_monotonicity(
    factor_name: str,
    metrics_df: pd.DataFrame,
    monotonicity: dict,
    save_dir: Path = None,
) -> None:
    """4-panel monotonicity chart with Spearman rho annotations."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT / factor_name.lower()

    panels = [
        ("CAGR (%)", "CAGR (%)"),
        ("Sharpe", "Sharpe Ratio"),
        ("Ann. Vol (%)", "Ann. Volatility (%)"),
        ("Max Drawdown (%)", "Max Drawdown (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{factor_name} – Monotonicity Analysis", fontweight="bold", fontsize=14)

    for ax, (col, label) in zip(axes.flatten(), panels):
        if col not in metrics_df.columns:
            ax.set_visible(False)
            continue
        vals = metrics_df[col].dropna().sort_index()
        ax.plot(vals.index, vals.values, "o-", color="#2166ac", lw=2, ms=7, zorder=3)
        ax.set_xlabel("Decile (1=Lowest factor, 10=Highest)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(sorted(vals.index))

        # OLS trend line
        if len(vals) >= 3:
            z = np.polyfit(vals.index.astype(float), vals.values, 1)
            xfit = np.linspace(vals.index.min(), vals.index.max(), 50)
            ax.plot(xfit, np.polyval(z, xfit), "--", color="#d73027", alpha=0.6, lw=1.5)

        # Spearman annotation
        if col in monotonicity:
            rho = monotonicity[col]["rho"]
            pval = monotonicity[col]["pval"]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            direction = "NORMAL" if rho > 0 else "REVERSED"
            ax.text(
                0.05, 0.95,
                f"rho = {rho:.3f}{stars}\n{direction}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                color="#053061",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#053061", alpha=0.85),
            )

    fig.tight_layout()
    _save(fig, save_dir / "monotonicity.png")


def plot_factor_spread_curve(
    factor_name: str,
    decile_returns: dict[int, pd.Series],
    save_dir: Path = None,
) -> None:
    """D1 / D5 / D10 equity curves + D10-D1 cumulative spread."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT / factor_name.lower()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{factor_name} – Factor Spread Analysis", fontweight="bold", fontsize=14)

    for d, color, label, lw in [
        (1, DECILE_COLORS[0], "D1 (Lowest)", 2.0),
        (5, DECILE_COLORS[4], "D5 (Middle)", 1.5),
        (10, DECILE_COLORS[9], "D10 (Highest)", 2.0),
    ]:
        if d not in decile_returns:
            continue
        cum = (1 + decile_returns[d].fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, color=color, lw=lw, label=label,
                 ls="--" if d == 5 else "-")

    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_ylabel("Wealth (log scale)")
    ax1.set_title("D1 / D5 / D10 Equity Curves")
    ax1.legend(framealpha=0.9)

    # Spread panel
    if 1 in decile_returns and 10 in decile_returns:
        d1 = decile_returns[1]
        d10 = decile_returns[10]
        common = d1.index.intersection(d10.index)
        spread = d10.loc[common] - d1.loc[common]
        cum_spread = (1 + spread.fillna(0)).cumprod()
        final = cum_spread.iloc[-1]
        color = "#2166ac" if final >= 1 else "#d73027"
        ax2.plot(cum_spread.index, cum_spread.values, color=color, lw=2)
        ax2.axhline(1, color="black", lw=0.8, ls="--")
        ax2.fill_between(
            cum_spread.index, 1, cum_spread.values,
            where=cum_spread.values >= 1, alpha=0.1, color="#2166ac",
        )
        ax2.fill_between(
            cum_spread.index, 1, cum_spread.values,
            where=cum_spread.values < 1, alpha=0.1, color="#d73027",
        )
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
        ax2.set_ylabel("D10 minus D1 (wealth)")
        ax2.set_title("D10 – D1 Long/Short Spread (>1x = D10 wins)")

    _year_fmt(ax2)
    fig.tight_layout()
    _save(fig, save_dir / "factor_spread.png")


def plot_factor_dashboard(
    factor_name: str,
    decile_returns: dict[int, pd.Series],
    metrics_df: pd.DataFrame,
    monotonicity: dict,
    save_dir: Path = None,
) -> None:
    """4-panel single-page summary dashboard for one factor."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT / factor_name.lower()

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"{factor_name} Factor – Indian Equities (NIFTY 500)",
        fontsize=16, fontweight="bold", y=0.98,
    )
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # Panel 1: All decile equity curves
    ax1 = fig.add_subplot(gs[0, 0])
    for d in sorted(decile_returns):
        cum = (1 + decile_returns[d].fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, color=DECILE_COLORS[d - 1], lw=1.3, label=f"D{d}")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.set_title("All Decile Equity Curves", fontweight="bold")
    ax1.legend(ncol=2, fontsize=7)
    _year_fmt(ax1)

    # Panel 2: CAGR bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    if "CAGR (%)" in metrics_df.columns:
        vals = metrics_df["CAGR (%)"].sort_index()
        colors = [DECILE_COLORS[int(d) - 1] for d in vals.index]
        ax2.bar([f"D{int(d)}" for d in vals.index], vals.values, color=colors, edgecolor="white")
        ax2.set_title("CAGR (%) by Decile", fontweight="bold")
        ax2.set_xlabel("Decile")
        ax2.set_ylabel("CAGR (%)")

    # Panel 3: Sharpe monotonicity
    ax3 = fig.add_subplot(gs[1, 0])
    if "Sharpe" in metrics_df.columns:
        vals = metrics_df["Sharpe"].dropna().sort_index()
        ax3.plot(vals.index, vals.values, "o-", color="#2166ac", lw=2, ms=7)
        ax3.set_xlabel("Decile")
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_title("Sharpe Ratio Monotonicity", fontweight="bold")
        if "Sharpe" in monotonicity:
            rho = monotonicity["Sharpe"]["rho"]
            pval = monotonicity["Sharpe"]["pval"]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            ax3.text(0.05, 0.95, f"rho={rho:.3f}{stars}",
                     transform=ax3.transAxes, va="top", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#053061", alpha=0.85))

    # Panel 4: Vol monotonicity
    ax4 = fig.add_subplot(gs[1, 1])
    if "Ann. Vol (%)" in metrics_df.columns:
        vals = metrics_df["Ann. Vol (%)"].dropna().sort_index()
        ax4.plot(vals.index, vals.values, "o-", color="#d73027", lw=2, ms=7)
        ax4.set_xlabel("Decile")
        ax4.set_ylabel("Ann. Vol (%)")
        ax4.set_title("Volatility Monotonicity", fontweight="bold")
        if "Ann. Vol (%)" in monotonicity:
            rho = monotonicity["Ann. Vol (%)"]["rho"]
            pval = monotonicity["Ann. Vol (%)"]["pval"]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            ax4.text(0.05, 0.95, f"rho={rho:.3f}{stars}",
                     transform=ax4.transAxes, va="top", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d73027", alpha=0.85))

    _save(fig, save_dir / "dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-factor comparison charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_factor_comparison(
    all_results: dict[str, dict],
    save_dir: Path = None,
) -> None:
    """
    Cross-factor summary: CAGR, Sharpe, and spread CAGR for all factors.
    Saved to research/charts/factor_comparison.png.
    """
    _apply_style()
    save_dir = save_dir or CHART_ROOT

    factor_names = [f for f in all_results if all_results[f].get("spread_stats")]
    if not factor_names:
        log.warning("No factors with spread_stats — skipping comparison chart.")
        return

    d1_cagr = [all_results[f]["spread_stats"].get("D1_CAGR", np.nan) for f in factor_names]
    d10_cagr = [all_results[f]["spread_stats"].get("D10_CAGR", np.nan) for f in factor_names]
    d1_sharpe = [all_results[f]["spread_stats"].get("D1_Sharpe", np.nan) for f in factor_names]
    d10_sharpe = [all_results[f]["spread_stats"].get("D10_Sharpe", np.nan) for f in factor_names]
    spreads = [all_results[f]["spread_stats"].get("CAGR (%)", np.nan) for f in factor_names]

    x = np.arange(len(factor_names))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Factor Comparison – Indian Equities (NIFTY 500, 2010–present)",
                 fontweight="bold", fontsize=14)

    # CAGR
    axes[0].bar(x - w / 2, d1_cagr, w, label="D1 (Low)", color="#2166ac", alpha=0.85)
    axes[0].bar(x + w / 2, d10_cagr, w, label="D10 (High)", color="#d73027", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(factor_names, rotation=30, ha="right")
    axes[0].set_title("D1 vs D10 CAGR (%)", fontweight="bold")
    axes[0].set_ylabel("CAGR (%)")
    axes[0].legend()

    # Sharpe
    axes[1].bar(x - w / 2, d1_sharpe, w, label="D1 (Low)", color="#2166ac", alpha=0.85)
    axes[1].bar(x + w / 2, d10_sharpe, w, label="D10 (High)", color="#d73027", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(factor_names, rotation=30, ha="right")
    axes[1].set_title("D1 vs D10 Sharpe Ratio", fontweight="bold")
    axes[1].set_ylabel("Sharpe")
    axes[1].legend()

    # D10-D1 spread
    bar_colors = ["#2166ac" if (s is not None and not np.isnan(s) and s >= 0) else "#d73027"
                  for s in spreads]
    axes[2].bar(x, spreads, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(factor_names, rotation=30, ha="right")
    axes[2].set_title("D10 – D1 Spread CAGR (%)", fontweight="bold")
    axes[2].set_ylabel("Spread CAGR (%)")
    axes[2].text(0.01, 0.99, "Blue=factor works (D10>D1)", transform=axes[2].transAxes,
                 va="top", fontsize=8, color="#2166ac")

    fig.tight_layout()
    _save(fig, save_dir / "factor_comparison.png")


def plot_equity_comparison(
    all_results: dict[str, dict],
    save_dir: Path = None,
) -> None:
    """Overlay D1 equity curves for all factors on one chart (and D10 on another)."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT
    factor_names = list(all_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Cross-Factor Equity Curves – NIFTY 500", fontweight="bold", fontsize=14)

    for fname in factor_names:
        dr = all_results[fname].get("decile_returns", {})
        color = FACTOR_COLORS.get(fname, "#555555")
        if 1 in dr:
            cum = (1 + dr[1].fillna(0)).cumprod()
            ax1.plot(cum.index, cum.values, color=color, lw=1.8, label=f"{fname} D1")
        if 10 in dr:
            cum = (1 + dr[10].fillna(0)).cumprod()
            ax2.plot(cum.index, cum.values, color=color, lw=1.8, label=f"{fname} D10")

    for ax, title in [(ax1, "D1 (Low factor score) Portfolios"),
                      (ax2, "D10 (High factor score) Portfolios")]:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Wealth (log scale)")
        ax.legend(framealpha=0.9)
        _year_fmt(ax)

    fig.tight_layout()
    _save(fig, save_dir / "equity_comparison.png")


def plot_sharpe_heatmap(
    all_results: dict[str, dict],
    save_dir: Path = None,
) -> None:
    """Heatmap: factors x deciles, colored by Sharpe ratio."""
    _apply_style()
    save_dir = save_dir or CHART_ROOT

    factor_names = list(all_results.keys())
    data = []
    for fname in factor_names:
        m = all_results[fname].get("metrics", pd.DataFrame())
        if m.empty or "Sharpe" not in m.columns:
            data.append([np.nan] * 10)
        else:
            row = [m.loc[d, "Sharpe"] if d in m.index else np.nan for d in range(1, 11)]
            data.append(row)

    if not data:
        return

    mat = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(13, max(4, len(factor_names) * 0.9)))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(mat) - 0.1, vmax=np.nanmax(mat) + 0.1)

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"D{d}" for d in range(1, 11)])
    ax.set_yticks(range(len(factor_names)))
    ax.set_yticklabels(factor_names)
    ax.set_title("Sharpe Ratio by Factor and Decile", fontweight="bold", pad=12)

    for i in range(len(factor_names)):
        for j in range(10):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.6)
    fig.tight_layout()
    _save(fig, save_dir / "sharpe_heatmap.png")
