"""
portfolio.py
------------
Constructs monthly-rebalanced equal-weight decile portfolios and computes
their daily return series.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Rebalance dates
# ──────────────────────────────────────────────────────────────────────────────

def get_rebalance_dates(
    prices: pd.DataFrame,
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DatetimeIndex:
    """
    Return the first trading day of each month within [start, end].
    These are the dates on which factor scores are computed and portfolios reset.
    """
    trading_days = prices.loc[start:end].index
    monthly_groups = trading_days.to_series().groupby(
        [trading_days.year, trading_days.month]
    )
    first_days = monthly_groups.first()
    rebalance = pd.DatetimeIndex(first_days.values)
    log.info(
        "Rebalance dates: %d months from %s to %s.",
        len(rebalance), rebalance[0].date(), rebalance[-1].date(),
    )
    return rebalance


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ──────────────────────────────────────────────────────────────────────────────

def build_portfolio_returns(
    daily_returns: pd.DataFrame,
    constituents: dict[pd.Timestamp, list[str]],
    rebalance_dates: pd.DatetimeIndex,
    label: str = "Portfolio",
) -> pd.Series:
    """
    Simulate a monthly-rebalanced equal-weight portfolio.

    On each rebalance date T, equal-weight the constituents for that month.
    Hold those weights until the next rebalance date T+1.
    Portfolio return on any day d = mean of constituent returns on d.

    Parameters
    ----------
    daily_returns  : DataFrame [dates × tickers]
    constituents   : {rebalance_date → [ticker, ...]} from factor_engine
    rebalance_dates: sorted sequence of rebalance dates

    Returns
    -------
    pd.Series of daily portfolio returns
    """
    portfolio_returns: list[tuple[pd.Timestamp, float]] = []

    for i, reb_date in enumerate(rebalance_dates):
        # Holding period: from this rebalance date up to (not including) the next
        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        holding = daily_returns.loc[reb_date:next_reb]
        if next_reb is not None:
            holding = holding.iloc[:-1]  # exclude next rebalance date itself

        tickers = constituents.get(reb_date, [])
        # Keep only tickers that have return data
        valid = [t for t in tickers if t in daily_returns.columns]
        if not valid:
            log.debug("%s: no valid tickers on %s.", label, reb_date.date())
            for day in holding.index:
                portfolio_returns.append((day, np.nan))
            continue

        period_returns = daily_returns.loc[holding.index, valid]
        # Equal-weight: row mean across available stocks
        daily_port = period_returns.mean(axis=1)

        for day, ret in daily_port.items():
            portfolio_returns.append((day, ret))

    if not portfolio_returns:
        return pd.Series(dtype=float, name=label)

    idx, vals = zip(*portfolio_returns)
    series = pd.Series(vals, index=pd.DatetimeIndex(idx), name=label)
    series = series[~series.index.duplicated(keep="first")].sort_index()
    log.info(
        "%s: %d daily returns, mean=%.4f%%, std=%.4f%%.",
        label, len(series), series.mean() * 100, series.std() * 100,
    )
    return series


def build_equal_weight_universe_returns(
    daily_returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    min_history_days: int = 126,
) -> pd.Series:
    """
    Equal-weight portfolio of ALL NIFTY 500 stocks (no factor filter).
    Used as a benchmark for the full universe.
    At each rebalance, only include stocks with at least min_history_days of data.
    """
    constituents: dict[pd.Timestamp, list[str]] = {}
    for reb_date in rebalance_dates:
        hist = daily_returns[daily_returns.index < reb_date]
        valid = hist.columns[hist.notna().sum() >= min_history_days].tolist()
        constituents[reb_date] = valid

    return build_portfolio_returns(
        daily_returns, constituents, rebalance_dates, label="EW_Universe"
    )


def build_benchmark_returns(
    benchmark_prices: pd.Series,
    label: str = "NIFTY50",
) -> pd.Series:
    """Convert benchmark index level to daily return series."""
    rets = benchmark_prices.pct_change().dropna()
    rets.name = label
    return rets


# ──────────────────────────────────────────────────────────────────────────────
# Spread portfolio (long bottom decile, short top decile)
# ──────────────────────────────────────────────────────────────────────────────

def build_long_short_spread(
    bottom_returns: pd.Series,
    top_returns: pd.Series,
    label: str = "MAX_LS_Spread",
) -> pd.Series:
    """
    Long bottom decile (low MAX), short top decile (high MAX).
    No leverage – gross exposure = 200%, net = 0%.
    """
    combined = pd.concat([bottom_returns, top_returns], axis=1).dropna()
    spread = combined.iloc[:, 0] - combined.iloc[:, 1]
    spread.name = label
    return spread


# ──────────────────────────────────────────────────────────────────────────────
# Multi-decile runner
# ──────────────────────────────────────────────────────────────────────────────

def build_all_decile_returns(
    daily_returns: pd.DataFrame,
    all_constituents: dict[int, dict[pd.Timestamp, list[str]]],
    rebalance_dates: pd.DatetimeIndex,
) -> dict[int, pd.Series]:
    """Build return series for every decile."""
    result: dict[int, pd.Series] = {}
    for decile, constituents in all_constituents.items():
        result[decile] = build_portfolio_returns(
            daily_returns, constituents, rebalance_dates, label=f"D{decile}"
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert daily return series to cumulative wealth index (start = 1)."""
    return (1 + returns.fillna(0)).cumprod()
