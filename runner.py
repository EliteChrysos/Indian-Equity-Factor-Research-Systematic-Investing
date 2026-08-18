"""
research/runner.py
------------------
Generic decile pipeline: takes any factor DataFrame and produces
decile portfolios, performance metrics, and monotonicity analysis.

Imports and reuses portfolio.py and performance.py from the parent directory.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from factor_engine import get_all_decile_constituents
from portfolio import build_all_decile_returns
from performance import compute_metrics

log = logging.getLogger(__name__)


def run_factor_pipeline(
    factor_name: str,
    factor_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    backtest_start: str = "2010-01-01",
    backtest_end: str = None,
    n_deciles: int = 10,
) -> dict:
    """
    Full decile pipeline for any factor.

    Parameters
    ----------
    factor_name     : display label (e.g. "MAX", "MOMENTUM_6M")
    factor_df       : [rebalance_dates x tickers], higher = better/stronger
    daily_returns   : [trading_days x tickers]
    rebalance_dates : monthly DatetimeIndex

    Returns
    -------
    dict with keys:
      'factor_name'      : str
      'decile_returns'   : {1..10: pd.Series of daily returns}
      'metrics'          : DataFrame [decile x metric]
      'monotonicity'     : {metric_col: {'rho': float, 'pval': float}}
      'spread_stats'     : dict with D10-D1 spread metrics + D1/D10 summaries
      'all_constituents' : {decile: {date: [tickers]}}
    """
    if factor_df.empty or factor_df.notna().sum().sum() == 0:
        log.warning("%s: factor DataFrame is empty, skipping pipeline.", factor_name)
        return {
            "factor_name": factor_name,
            "decile_returns": {},
            "metrics": pd.DataFrame(),
            "monotonicity": {},
            "spread_stats": {},
            "all_constituents": {},
        }

    valid_per_date = factor_df.notna().sum(axis=1)
    log.info(
        "%s: factor coverage — avg %.0f stocks/period (min %d, max %d)",
        factor_name, valid_per_date.mean(), valid_per_date.min(), valid_per_date.max(),
    )

    # Build decile constituents and portfolio return series
    all_constituents = get_all_decile_constituents(factor_df, n_deciles=n_deciles)
    decile_returns_raw = build_all_decile_returns(daily_returns, all_constituents, rebalance_dates)

    # Trim to backtest window
    start_dt = pd.Timestamp(backtest_start)
    end_dt = pd.Timestamp(backtest_end) if backtest_end else pd.Timestamp.today()
    decile_returns: dict[int, pd.Series] = {}
    for d, r in decile_returns_raw.items():
        trimmed = r.loc[start_dt:end_dt].dropna()
        if not trimmed.empty:
            decile_returns[d] = trimmed

    if not decile_returns:
        log.warning("%s: all decile return series are empty after trimming.", factor_name)
        return {
            "factor_name": factor_name,
            "decile_returns": {},
            "metrics": pd.DataFrame(),
            "monotonicity": {},
            "spread_stats": {},
            "all_constituents": all_constituents,
        }

    # Compute metrics for each decile
    metrics_rows = []
    for d in range(1, n_deciles + 1):
        if d not in decile_returns:
            continue
        m = compute_metrics(decile_returns[d], label=f"{factor_name}_D{d}")
        m["Decile"] = d
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Decile") if metrics_rows else pd.DataFrame()

    # Monotonicity: Spearman rho between decile rank and metric value
    monotonicity: dict = {}
    if not metrics_df.empty and len(metrics_df) >= 5:
        for col in ["CAGR (%)", "Ann. Vol (%)", "Sharpe", "Max Drawdown (%)"]:
            if col not in metrics_df.columns:
                continue
            vals = metrics_df[col].dropna()
            if len(vals) < 5:
                continue
            rho, pval = stats.spearmanr(vals.index.astype(float), vals.values)
            monotonicity[col] = {"rho": round(rho, 4), "pval": round(pval, 4)}

    # Factor spread: D10 - D1 (positive if higher factor score = better)
    spread_stats: dict = {}
    if 1 in decile_returns and n_deciles in decile_returns:
        d1 = decile_returns[1]
        d10 = decile_returns[n_deciles]
        common = d1.index.intersection(d10.index)
        spread = d10.loc[common] - d1.loc[common]
        spread_stats = compute_metrics(spread, label=f"{factor_name}_D10mD1")
        if not metrics_df.empty:
            spread_stats["D1_CAGR"] = metrics_df.loc[1, "CAGR (%)"] if 1 in metrics_df.index else np.nan
            spread_stats["D10_CAGR"] = metrics_df.loc[n_deciles, "CAGR (%)"] if n_deciles in metrics_df.index else np.nan
            spread_stats["D1_Sharpe"] = metrics_df.loc[1, "Sharpe"] if 1 in metrics_df.index else np.nan
            spread_stats["D10_Sharpe"] = metrics_df.loc[n_deciles, "Sharpe"] if n_deciles in metrics_df.index else np.nan
            spread_stats["D1_Vol"] = metrics_df.loc[1, "Ann. Vol (%)"] if 1 in metrics_df.index else np.nan
            spread_stats["D10_Vol"] = metrics_df.loc[n_deciles, "Ann. Vol (%)"] if n_deciles in metrics_df.index else np.nan
            spread_stats["D1_MaxDD"] = metrics_df.loc[1, "Max Drawdown (%)"] if 1 in metrics_df.index else np.nan
            spread_stats["D10_MaxDD"] = metrics_df.loc[n_deciles, "Max Drawdown (%)"] if n_deciles in metrics_df.index else np.nan

    log.info(
        "%s: D1 CAGR=%.1f%%, D10 CAGR=%.1f%%, Spread=%.1f%%, D1 Sharpe=%.3f, D10 Sharpe=%.3f",
        factor_name,
        spread_stats.get("D1_CAGR", np.nan),
        spread_stats.get("D10_CAGR", np.nan),
        spread_stats.get("CAGR (%)", np.nan),
        spread_stats.get("D1_Sharpe", np.nan),
        spread_stats.get("D10_Sharpe", np.nan),
    )

    return {
        "factor_name": factor_name,
        "decile_returns": decile_returns,
        "metrics": metrics_df,
        "monotonicity": monotonicity,
        "spread_stats": spread_stats,
        "all_constituents": all_constituents,
    }
