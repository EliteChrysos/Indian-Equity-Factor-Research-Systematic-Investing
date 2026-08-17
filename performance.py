"""
performance.py
--------------
Risk/return metrics and publication-quality charts for the MAX factor backtest.
"""

import logging
import warnings
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06          # ~India 10yr Gsec as proxy
FIGURE_DPI = 150
CHART_DIR = "charts"

import os
os.makedirs(CHART_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, label: str = "Portfolio") -> dict:
    """Compute comprehensive risk/return metrics from a daily return series."""
    r = returns.dropna()
    if r.empty:
        return {}

    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    excess = r - rf_daily

    cum = (1 + r).cumprod()
    n_years = len(r) / TRADING_DAYS

    cagr = cum.iloc[-1] ** (1 / n_years) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    ann_ret = r.mean() * TRADING_DAYS

    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan

    downside = r[r < rf_daily]
    sortino_denom = downside.std() * np.sqrt(TRADING_DAYS) if not downside.empty else np.nan
    sortino = (ann_ret - RISK_FREE_RATE) / sortino_denom if sortino_denom else np.nan

    drawdown = _max_drawdown(r)
    calmar = cagr / abs(drawdown) if drawdown != 0 else np.nan

    monthly_r = _to_monthly(r)
    win_rate = (monthly_r > 0).mean()
    best_month  = monthly_r.max()
    worst_month = monthly_r.min()
    avg_up   = monthly_r[monthly_r > 0].mean()
    avg_down = monthly_r[monthly_r < 0].mean()

    return {
        "Label":            label,
        "CAGR (%)":         round(cagr * 100, 2),
        "Ann. Return (%)":  round(ann_ret * 100, 2),
        "Ann. Vol (%)":     round(ann_vol * 100, 2),
        "Sharpe":           round(sharpe, 2),
        "Sortino":          round(sortino, 2),
        "Calmar":           round(calmar, 2),
        "Max Drawdown (%)": round(drawdown * 100, 2),
        "Win Rate (%)":     round(win_rate * 100, 1),
        "Best Month (%)":   round(best_month * 100, 2),
        "Worst Month (%)":  round(worst_month * 100, 2),
        "Avg Up Month (%)": round(avg_up * 100, 2),
        "Avg Dn Month (%)": round(avg_down * 100, 2),
        "N Days":           len(r),
        "N Years":          round(n_years, 1),
    }


def metrics_table(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a summary DataFrame from multiple return series."""
    rows = [compute_metrics(r, label=name) for name, r in returns_dict.items()]
    df = pd.DataFrame(rows).set_index("Label")
    return df


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def _to_monthly(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def _drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(TRADING_DAYS)


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bottom":  "#1a6faf",   # blue – low MAX (long)
    "mid":     "#f5a623",   # amber – middle
    "top":     "#d0021b",   # red – high MAX (short)
    "nifty50": "#555555",   # dark grey
    "ew":      "#7ed321",   # green
    "spread":  "#9013fe",   # purple
}

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f9fa",
        "axes.grid":        True,
        "grid.color":       "#e0e0e0",
        "grid.linewidth":   0.6,
        "font.family":      "sans-serif",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  9,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Individual charts
# ──────────────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    returns_dict: dict[str, pd.Series],
    title: str = "Equity Curve – MAX Factor Portfolios",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    color_map = {
        "MAX_Bottom_D1": PALETTE["bottom"],
        "MAX_Top_D10":   PALETTE["top"],
        "NIFTY50":       PALETTE["nifty50"],
        "EW_Universe":   PALETTE["ew"],
        "MAX_LS_Spread": PALETTE["spread"],
    }

    for name, rets in returns_dict.items():
        cum = (1 + rets.fillna(0)).cumprod()
        color = color_map.get(name, None)
        lw = 2.0 if name in ("MAX_Bottom_D1", "NIFTY50") else 1.5
        ls = "--" if name == "MAX_LS_Spread" else "-"
        ax.plot(cum.index, cum.values, label=name, color=color, lw=lw, ls=ls)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}"))
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Wealth (₹1 invested, log scale)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save_or_show(fig, save_path or f"{CHART_DIR}/equity_curve.png")


def plot_drawdown(
    returns_dict: dict[str, pd.Series],
    title: str = "Drawdown – MAX Factor Portfolios",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    color_map = {
        "MAX_Bottom_D1": PALETTE["bottom"],
        "MAX_Top_D10":   PALETTE["top"],
        "NIFTY50":       PALETTE["nifty50"],
        "EW_Universe":   PALETTE["ew"],
    }

    for name, rets in returns_dict.items():
        if name == "MAX_LS_Spread":
            continue
        dd = _drawdown_series(rets)
        color = color_map.get(name, None)
        lw = 2.0 if name == "MAX_Bottom_D1" else 1.5
        ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=color)
        ax.plot(dd.index, dd.values, label=name, color=color, lw=lw)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save_or_show(fig, save_path or f"{CHART_DIR}/drawdown.png")


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    window: int = 252,
    title: str = "Rolling 1-Year Sharpe Ratio",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    color_map = {
        "MAX_Bottom_D1": PALETTE["bottom"],
        "MAX_Top_D10":   PALETTE["top"],
        "NIFTY50":       PALETTE["nifty50"],
        "EW_Universe":   PALETTE["ew"],
    }

    for name, rets in returns_dict.items():
        if name == "MAX_LS_Spread":
            continue
        rs = _rolling_sharpe(rets, window)
        color = color_map.get(name, None)
        ax.plot(rs.index, rs.values, label=name, color=color, lw=1.5)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axhline(1, color="green", lw=0.6, ls=":", alpha=0.7)
    ax.set_title(f"{title} ({window}d window)", fontweight="bold", pad=12)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save_or_show(fig, save_path or f"{CHART_DIR}/rolling_sharpe.png")


def plot_monthly_heatmap(
    returns: pd.Series,
    title: str = "Monthly Return Heatmap",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    monthly = _to_monthly(returns) * 100

    pivot = monthly.to_frame("ret")
    pivot.index = pd.DatetimeIndex(pivot.index)
    pivot["year"]  = pivot.index.year
    pivot["month"] = pivot.index.month

    table = pivot.pivot_table(index="year", columns="month", values="ret")
    table.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec",
    ]

    fig, ax = plt.subplots(figsize=(14, max(5, len(table) * 0.45)))

    vmax = min(abs(table.values[~np.isnan(table.values)]).max(), 20)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(12))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index)

    for i in range(len(table)):
        for j in range(12):
            val = table.values[i, j]
            if not np.isnan(val):
                text_color = "black" if abs(val) < vmax * 0.6 else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7.5, color=text_color)

    plt.colorbar(im, ax=ax, label="Return (%)", shrink=0.6)
    ax.set_title(f"{title} – {returns.name}", fontweight="bold", pad=12)
    fig.tight_layout()
    _save_or_show(fig, save_path or f"{CHART_DIR}/monthly_heatmap.png")


def plot_factor_spread(
    decile_returns: dict[int, pd.Series],
    title: str = "Factor Spread: Bottom / Middle / Top Decile",
    save_path: Optional[str] = None,
) -> None:
    """Plot equity curves for deciles 1 (bottom), 5 (middle), 10 (top)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    decile_config = {
        1:  ("Low MAX (D1)",  PALETTE["bottom"], 2.2),
        5:  ("Mid MAX (D5)",  PALETTE["mid"],    1.5),
        10: ("High MAX (D10)", PALETTE["top"],   2.0),
    }

    for d, (label, color, lw) in decile_config.items():
        if d not in decile_returns:
            continue
        cum = (1 + decile_returns[d].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=label, color=color, lw=lw)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}"))
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylabel("Wealth (₹1 invested, log scale)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save_or_show(fig, save_path or f"{CHART_DIR}/factor_spread.png")


# ──────────────────────────────────────────────────────────────────────────────
# Combined dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_dashboard(
    returns_dict: dict[str, pd.Series],
    decile_returns: dict[int, pd.Series],
    save_path: Optional[str] = None,
) -> None:
    """4-panel summary dashboard."""
    _apply_style()
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "MAX Factor Anomaly – India Equities Backtest",
        fontsize=16, fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # Panel 1: Equity curve
    ax1 = fig.add_subplot(gs[0, 0])
    color_map = {
        "MAX_Bottom_D1": PALETTE["bottom"],
        "MAX_Top_D10":   PALETTE["top"],
        "NIFTY50":       PALETTE["nifty50"],
        "EW_Universe":   PALETTE["ew"],
    }
    for name, rets in returns_dict.items():
        if name == "MAX_LS_Spread":
            continue
        cum = (1 + rets.fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, label=name,
                 color=color_map.get(name), lw=1.8)
    ax1.set_yscale("log")
    ax1.set_title("Equity Curves (log scale)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    for name, rets in returns_dict.items():
        if name == "MAX_LS_Spread":
            continue
        dd = _drawdown_series(rets)
        ax2.plot(dd.index, dd.values, label=name,
                 color=color_map.get(name), lw=1.5)
    ax2.set_title("Drawdown", fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.legend(fontsize=8)

    # Panel 3: Rolling Sharpe
    ax3 = fig.add_subplot(gs[1, 0])
    for name, rets in returns_dict.items():
        if name == "MAX_LS_Spread":
            continue
        rs = _rolling_sharpe(rets)
        ax3.plot(rs.index, rs.values, label=name,
                 color=color_map.get(name), lw=1.5)
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_title("Rolling 1-Year Sharpe", fontweight="bold")
    ax3.legend(fontsize=8)

    # Panel 4: Factor spread (decile 1 vs 5 vs 10)
    ax4 = fig.add_subplot(gs[1, 1])
    decile_cfg = {
        1:  (PALETTE["bottom"], "Low MAX D1",   2.0),
        5:  (PALETTE["mid"],    "Mid MAX D5",   1.5),
        10: (PALETTE["top"],    "High MAX D10", 2.0),
    }
    for d, (color, label, lw) in decile_cfg.items():
        if d in decile_returns:
            cum = (1 + decile_returns[d].fillna(0)).cumprod()
            ax4.plot(cum.index, cum.values, color=color, label=label, lw=lw)
    ax4.set_yscale("log")
    ax4.set_title("Factor Spread (D1 / D5 / D10)", fontweight="bold")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax4.legend(fontsize=8)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    _save_or_show(fig, save_path or f"{CHART_DIR}/dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_metrics_table(df: pd.DataFrame) -> None:
    """Pretty-print the metrics table."""
    float_cols = df.select_dtypes(include=[float, int]).columns
    styled = df.copy()
    for col in float_cols:
        styled[col] = styled[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY")
    print("=" * 90)
    print(styled.to_string())
    print("=" * 90 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    log.info("Saved chart: %s", path)
    plt.close(fig)
