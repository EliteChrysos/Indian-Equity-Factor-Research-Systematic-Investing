"""
factor_engine.py
----------------
Computes the MAX(5) factor for each stock at each rebalance date.

Definition (Bali, Cakici & Whitelaw 2011):
    MAX(N) = average of the N highest daily returns in the lookback window.

No look-ahead: on rebalance date T, only returns up to T-1 are used.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core factor calculation
# ──────────────────────────────────────────────────────────────────────────────

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple arithmetic daily returns from adjusted close prices."""
    returns = prices.pct_change()
    # Cap extreme single-day moves at ±50% (data artefact guard)
    returns = returns.clip(-0.50, 0.50)
    return returns


def compute_max_factor(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback: int = 21,
    n_max: int = 5,
    min_obs_ratio: float = 0.7,
    min_volume_series: Optional[pd.DataFrame] = None,
    min_adtv_crore: float = 1.0,
) -> pd.DataFrame:
    """
    Compute MAX(n_max) factor for every stock at every rebalance date.

    Parameters
    ----------
    returns          : daily return DataFrame [dates × tickers]
    rebalance_dates  : dates on which the factor is scored
    lookback         : trading days to look back
    n_max            : number of top returns to average
    min_obs_ratio    : minimum fraction of non-NaN days required in lookback
    min_volume_series: daily volume × price (value traded) DataFrame
    min_adtv_crore   : minimum average daily traded value in crores (₹10M)

    Returns
    -------
    DataFrame [rebalance_dates × tickers] of MAX scores (NaN = excluded)
    """
    min_obs = int(lookback * min_obs_ratio)
    scores: dict[pd.Timestamp, pd.Series] = {}

    log.info(
        "Computing MAX(%d) factor over %d rebalance dates …", n_max, len(rebalance_dates)
    )

    for date in rebalance_dates:
        # Use data strictly before the rebalance date (no look-ahead)
        mask = returns.index < date
        window = returns[mask].iloc[-lookback:]

        if len(window) < min_obs:
            scores[date] = pd.Series(dtype=float)
            continue

        # For each stock: top-N returns average
        row_scores: dict[str, float] = {}
        for ticker in window.columns:
            col = window[ticker].dropna()
            if len(col) < min_obs:
                continue
            top_n = col.nlargest(n_max)
            row_scores[ticker] = top_n.mean()

        score_series = pd.Series(row_scores)

        # Liquidity filter: require minimum ADTV
        if min_volume_series is not None:
            vol_window = min_volume_series[min_volume_series.index < date].iloc[-lookback:]
            adtv = vol_window.mean() / 1e7  # convert to crore
            liquid = adtv[adtv >= min_adtv_crore].index
            score_series = score_series[score_series.index.isin(liquid)]

        scores[date] = score_series

    factor_df = pd.DataFrame(scores).T
    factor_df.index.name = "date"

    valid_counts = factor_df.notna().sum(axis=1)
    log.info(
        "Factor computed. Avg valid stocks per period: %.0f (min: %d, max: %d)",
        valid_counts.mean(), valid_counts.min(), valid_counts.max(),
    )
    return factor_df


# ──────────────────────────────────────────────────────────────────────────────
# Decile ranking
# ──────────────────────────────────────────────────────────────────────────────

def rank_into_deciles(factor_row: pd.Series, n_deciles: int = 10) -> pd.Series:
    """
    Assign decile ranks 1..n_deciles to a cross-sectional factor row.
    Decile 1 = lowest MAX (low-lottery stocks – the long portfolio).
    Decile 10 = highest MAX (high-lottery stocks).
    Returns NaN for tickers that had NaN factor values.
    """
    valid = factor_row.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    labels = pd.qcut(valid, q=n_deciles, labels=False, duplicates="drop") + 1
    return labels.astype(float)


def get_decile_constituents(
    factor_df: pd.DataFrame,
    target_decile: int,
    n_deciles: int = 10,
) -> dict[pd.Timestamp, list[str]]:
    """
    For each rebalance date, return the list of tickers in `target_decile`.
    target_decile=1 → lowest MAX (anomaly long leg)
    target_decile=10 → highest MAX (anomaly short leg)
    """
    constituents: dict[pd.Timestamp, list[str]] = {}
    for date, row in factor_df.iterrows():
        deciles = rank_into_deciles(row, n_deciles=n_deciles)
        tickers = deciles[deciles == target_decile].index.tolist()
        constituents[date] = tickers
    return constituents


def get_all_decile_constituents(
    factor_df: pd.DataFrame,
    n_deciles: int = 10,
) -> dict[int, dict[pd.Timestamp, list[str]]]:
    """Return constituent maps for all deciles."""
    return {
        d: get_decile_constituents(factor_df, d, n_deciles)
        for d in range(1, n_deciles + 1)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Factor diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def factor_summary(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional summary stats of the MAX factor per period."""
    summary = pd.DataFrame({
        "mean":  factor_df.mean(axis=1),
        "std":   factor_df.std(axis=1),
        "p10":   factor_df.quantile(0.10, axis=1),
        "p50":   factor_df.quantile(0.50, axis=1),
        "p90":   factor_df.quantile(0.90, axis=1),
        "n_valid": factor_df.notna().sum(axis=1),
    })
    return summary
