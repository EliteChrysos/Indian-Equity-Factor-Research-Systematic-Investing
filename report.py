"""
research/report.py
------------------
Generates the Markdown factor research report.
"""

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPORT_PATH = Path("research/factor_research_report.md")


def _fmt(val, fmt=".2f", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:{fmt}}{suffix}"


def _mono_verdict(monotonicity: dict, metric: str) -> str:
    if metric not in monotonicity:
        return "N/A"
    rho = monotonicity[metric]["rho"]
    pval = monotonicity[metric]["pval"]
    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    direction = "POSITIVE" if rho > 0 else "NEGATIVE"
    return f"{direction} (rho={rho:.3f}{stars})"


def _factor_section(factor_name: str, result: dict) -> str:
    """Generate one factor's section of the report."""
    metrics_df = result.get("metrics", pd.DataFrame())
    mono = result.get("monotonicity", {})
    ss = result.get("spread_stats", {})

    lines = [
        f"## {factor_name}",
        "",
    ]

    # Decile metrics table
    if not metrics_df.empty:
        lines += [
            "### Decile Performance",
            "",
            "| Decile | CAGR (%) | Ann. Vol (%) | Sharpe | Sortino | Calmar | Max DD (%) |",
            "|--------|----------|--------------|--------|---------|--------|------------|",
        ]
        for d in sorted(metrics_df.index):
            row = metrics_df.loc[d]
            lines.append(
                f"| D{int(d)} | "
                f"{_fmt(row.get('CAGR (%)'), '.2f')} | "
                f"{_fmt(row.get('Ann. Vol (%)'), '.2f')} | "
                f"{_fmt(row.get('Sharpe'), '.3f')} | "
                f"{_fmt(row.get('Sortino'), '.3f')} | "
                f"{_fmt(row.get('Calmar'), '.3f')} | "
                f"{_fmt(row.get('Max Drawdown (%)'), '.2f')} |"
            )
        lines.append("")

    # Factor spread summary
    if ss:
        d1_c = _fmt(ss.get("D1_CAGR"), ".2f", "%")
        d10_c = _fmt(ss.get("D10_CAGR"), ".2f", "%")
        spread_c = _fmt(ss.get("CAGR (%)"), ".2f", "%")
        d1_sh = _fmt(ss.get("D1_Sharpe"), ".3f")
        d10_sh = _fmt(ss.get("D10_Sharpe"), ".3f")
        d1_vol = _fmt(ss.get("D1_Vol"), ".1f", "%")
        d10_vol = _fmt(ss.get("D10_Vol"), ".1f", "%")
        d1_dd = _fmt(ss.get("D1_MaxDD"), ".1f", "%")
        d10_dd = _fmt(ss.get("D10_MaxDD"), ".1f", "%")

        lines += [
            "### Factor Spread (D10 – D1)",
            "",
            f"| Metric | D1 (Low) | D10 (High) | Spread (D10–D1) |",
            f"|--------|----------|------------|-----------------|",
            f"| CAGR | {d1_c} | {d10_c} | {spread_c} |",
            f"| Sharpe | {d1_sh} | {d10_sh} | — |",
            f"| Ann. Vol | {d1_vol} | {d10_vol} | — |",
            f"| Max DD | {d1_dd} | {d10_dd} | — |",
            "",
        ]

    # Monotonicity
    lines += [
        "### Monotonicity Analysis",
        "",
        "| Metric | Direction | Spearman rho | Significance |",
        "|--------|-----------|-------------|--------------|",
    ]
    for col in ["CAGR (%)", "Sharpe", "Ann. Vol (%)", "Max Drawdown (%)"]:
        if col in mono:
            rho = mono[col]["rho"]
            pval = mono[col]["pval"]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            direction = "POSITIVE" if rho > 0 else "NEGATIVE"
            lines.append(f"| {col} | {direction} | {rho:.4f} | {sig} (p={pval:.4f}) |")
        else:
            lines.append(f"| {col} | — | — | — |")

    # Interpretation snippet
    spread_cagr = ss.get("CAGR (%)", np.nan)
    d1_sharpe = ss.get("D1_Sharpe", np.nan)
    d10_sharpe = ss.get("D10_Sharpe", np.nan)
    cagr_rho = mono.get("CAGR (%)", {}).get("rho", np.nan)
    sharpe_rho = mono.get("Sharpe", {}).get("rho", np.nan)

    lines += ["", "### Interpretation", ""]
    if not np.isnan(spread_cagr):
        if spread_cagr > 5:
            lines.append(
                f"**Factor WORKS**: D10 outperforms D1 by {spread_cagr:.1f}% CAGR. "
                f"Higher {factor_name} score → higher returns."
            )
        elif spread_cagr < -5:
            lines.append(
                f"**Factor REVERSED**: D10 underperforms D1 by {abs(spread_cagr):.1f}% CAGR. "
                f"Lower {factor_name} score → higher returns. This is ANOMALOUS — consider that "
                f"D10 may carry excess risk rather than being a genuine return driver."
            )
        else:
            lines.append(
                f"**Factor FLAT**: D10 vs D1 spread is only {spread_cagr:.1f}% CAGR — "
                f"no clear directional signal."
            )

    if not np.isnan(d1_sharpe) and not np.isnan(d10_sharpe):
        if d1_sharpe > d10_sharpe:
            lines.append(
                f"**Risk-adjusted**: D1 Sharpe ({d1_sharpe:.3f}) > D10 Sharpe ({d10_sharpe:.3f}). "
                f"Even if D10 has higher raw return, it is NOT more efficient per unit of risk."
            )
        else:
            lines.append(
                f"**Risk-adjusted**: D10 Sharpe ({d10_sharpe:.3f}) > D1 Sharpe ({d1_sharpe:.3f}). "
                f"D10 is more efficient on a risk-adjusted basis."
            )

    lines += ["", "---", ""]
    return "\n".join(lines)


def generate_report(
    all_results: dict[str, dict],
    backtest_start: str = "2010-01-01",
    backtest_end: str = None,
) -> str:
    """Generate and save the full Markdown factor research report."""
    end_str = backtest_end or datetime.date.today().isoformat()
    run_date = datetime.date.today().isoformat()

    # Executive summary table
    summary_rows = []
    for fname, result in all_results.items():
        ss = result.get("spread_stats", {})
        mono = result.get("monotonicity", {})
        cagr_rho = mono.get("CAGR (%)", {}).get("rho", np.nan)
        sharpe_rho = mono.get("Sharpe", {}).get("rho", np.nan)
        cagr_sig = mono.get("CAGR (%)", {}).get("pval", 1.0)
        signal = "POSITIVE" if (not np.isnan(cagr_rho) and cagr_rho > 0 and cagr_sig < 0.05) else \
                 "REVERSED" if (not np.isnan(cagr_rho) and cagr_rho < 0 and cagr_sig < 0.05) else "FLAT"
        summary_rows.append({
            "Factor": fname,
            "D1 CAGR": ss.get("D1_CAGR", np.nan),
            "D10 CAGR": ss.get("D10_CAGR", np.nan),
            "Spread CAGR": ss.get("CAGR (%)", np.nan),
            "D1 Sharpe": ss.get("D1_Sharpe", np.nan),
            "D10 Sharpe": ss.get("D10_Sharpe", np.nan),
            "CAGR Mono rho": cagr_rho,
            "Sharpe Mono rho": sharpe_rho,
            "Signal": signal,
        })

    summary_df = pd.DataFrame(summary_rows).set_index("Factor")

    # Best factor by Sharpe of winning decile
    best_factor = None
    best_sharpe = -np.inf
    for row in summary_rows:
        max_sharpe = max(
            row.get("D1 Sharpe", -np.inf) if not np.isnan(row.get("D1 Sharpe", np.nan)) else -np.inf,
            row.get("D10 Sharpe", -np.inf) if not np.isnan(row.get("D10 Sharpe", np.nan)) else -np.inf,
        )
        if max_sharpe > best_sharpe:
            best_sharpe = max_sharpe
            best_factor = row["Factor"]

    lines = [
        "# Indian Equity Factor Research – Comprehensive Study",
        "",
        f"**Universe:** NIFTY 500 (current constituents)  ",
        f"**Backtest:** {backtest_start} to {end_str}  ",
        f"**Generated:** {run_date}  ",
        f"**Factors:** {', '.join(all_results.keys())}  ",
        "",
        "> **Survivorship bias caveat:** This study uses current NIFTY 500 constituents. Stocks that "
        "> delisted or dropped out of the index since 2010 are not included. This inflates returns, "
        "> particularly for high-volatility factors. Results are directionally informative but should "
        "> not be taken as live trading estimates.",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Factor | D1 CAGR | D10 CAGR | Spread | D1 Sharpe | D10 Sharpe | CAGR Mono | Signal |",
        "|--------|---------|----------|--------|-----------|------------|-----------|--------|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['Factor']} | "
            f"{_fmt(row['D1 CAGR'], '.1f', '%')} | "
            f"{_fmt(row['D10 CAGR'], '.1f', '%')} | "
            f"{_fmt(row['Spread CAGR'], '.1f', '%')} | "
            f"{_fmt(row['D1 Sharpe'], '.3f')} | "
            f"{_fmt(row['D10 Sharpe'], '.3f')} | "
            f"{_fmt(row['CAGR Mono rho'], '.3f')} | "
            f"**{row['Signal']}** |"
        )

    if best_factor:
        lines += [
            "",
            f"> **Best factor by risk-adjusted return:** {best_factor} "
            f"(best decile Sharpe: {best_sharpe:.3f})",
        ]

    lines += [
        "",
        "---",
        "",
    ]

    # Individual factor sections
    for fname, result in all_results.items():
        lines.append(_factor_section(fname, result))

    # Chart index
    lines += [
        "## Charts",
        "",
        "Charts saved to `research/charts/`:",
        "",
        "- `factor_comparison.png` — Side-by-side D1/D10 CAGR, Sharpe, and spread for all factors",
        "- `equity_comparison.png` — D1 and D10 equity curves overlaid across all factors",
        "- `sharpe_heatmap.png` — Sharpe ratio by factor × decile",
    ]
    for fname in all_results:
        lines += [
            f"- `{fname.lower()}/dashboard.png` — 4-panel summary",
            f"- `{fname.lower()}/decile_equity_curves.png`",
            f"- `{fname.lower()}/decile_bars.png`",
            f"- `{fname.lower()}/monotonicity.png`",
            f"- `{fname.lower()}/factor_spread.png`",
        ]

    lines += [
        "",
        "---",
        "*Generated by `research/run_research.py`. Not investment advice.*",
    ]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Report saved to %s", REPORT_PATH)
    return report
