"""
research/factors.py
-------------------
Factor score computation for all 6 factors.

Each function returns a DataFrame [rebalance_dates x tickers] where:
  - Higher score  = stronger factor signal (all factors are direction-unified)
  - NaN           = stock excluded at that date (insufficient data / liquidity)

Factor directions:
  MAX          higher score = higher lottery preference (D10 = most lottery-like)
  MOMENTUM_6M  higher score = stronger 6M momentum
  MOMENTUM_12M higher score = stronger 12M momentum
  LOW_VOL      higher score = LOWER volatility (score = -vol, so D10 = least volatile)
  QUALITY      higher score = higher ROE/margins composite
  VALUE        higher score = cheaper on PB/PE/EV-EBITDA composite
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── parent-module imports ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from factor_engine import compute_daily_returns, compute_max_factor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. MAX Factor
# ──────────────────────────────────────────────────────────────────────────────

def compute_max_scores(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback: int = 21,
    n_max: int = 5,
    min_adtv_crore: float = 1.0,
) -> pd.DataFrame:
    """Wrap the validated MAX factor from parent factor_engine."""
    returns = compute_daily_returns(prices)
    vol_value = prices * volume
    return compute_max_factor(
        returns, rebalance_dates,
        lookback=lookback, n_max=n_max,
        min_obs_ratio=0.7,
        min_volume_series=vol_value,
        min_adtv_crore=min_adtv_crore,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2 & 3. Momentum Factors
# ──────────────────────────────────────────────────────────────────────────────

def _compute_momentum(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback: int,
    skip: int = 21,
    min_obs_ratio: float = 0.7,
) -> pd.DataFrame:
    """
    Total return over `lookback` days, skipping the most recent `skip` days.
    Uses prices strictly before the rebalance date (no look-ahead).
    """
    min_obs = int(lookback * min_obs_ratio)
    scores: dict = {}
    for date in rebalance_dates:
        hist = prices[prices.index < date]
        needed = lookback + skip + 1
        if len(hist) < needed:
            scores[date] = pd.Series(dtype=float)
            continue

        # Price at end of lookback window (skip days before today)
        p_end = hist.iloc[-(skip + 1)]
        # Price at start of lookback window
        p_start = hist.iloc[-(skip + lookback + 1)]

        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (p_end / p_start - 1).replace([np.inf, -np.inf], np.nan)

        # Require minimum observations within the window
        window_slice = hist.iloc[-(skip + lookback): -(skip) if skip > 0 else None]
        obs = window_slice.notna().sum()
        ret = ret.where(obs >= min_obs)
        scores[date] = ret

    df = pd.DataFrame(scores).T
    df.index.name = "date"
    return df


def compute_momentum_6m(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """6-month momentum: 126-day return, skip 21 days."""
    log.info("Computing MOMENTUM_6M factor ...")
    return _compute_momentum(prices, rebalance_dates, lookback=126, skip=21)


def compute_momentum_12m(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """12-month momentum: 252-day return, skip 21 days."""
    log.info("Computing MOMENTUM_12M factor ...")
    return _compute_momentum(prices, rebalance_dates, lookback=252, skip=21)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Low Volatility
# ──────────────────────────────────────────────────────────────────────────────

def compute_low_vol(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback: int = 63,
    min_obs_ratio: float = 0.7,
) -> pd.DataFrame:
    """
    Low Volatility factor: score = -annualized_vol(63d).
    Higher score → lower volatility → stronger LOW_VOL signal.
    D10 = least volatile stocks.
    """
    log.info("Computing LOW_VOL factor ...")
    returns = compute_daily_returns(prices)
    min_obs = int(lookback * min_obs_ratio)
    scores: dict = {}
    for date in rebalance_dates:
        win = returns[returns.index < date].iloc[-lookback:]
        if len(win) < min_obs:
            scores[date] = pd.Series(dtype=float)
            continue
        obs = win.notna().sum()
        ann_vol = win.std() * np.sqrt(252)
        ann_vol = ann_vol.where(obs >= min_obs)
        scores[date] = -ann_vol  # negate: less volatile = higher score
    df = pd.DataFrame(scores).T
    df.index.name = "date"
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5 & 6. Fundamental Factors (QUALITY, VALUE)
# ──────────────────────────────────────────────────────────────────────────────

def _ttm(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Trailing twelve months: sum of last 4 fiscal quarters before cutoff."""
    if df.empty:
        return pd.Series(dtype=float)
    avail = df[df.index < cutoff]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.iloc[-4:].sum(min_count=1)


def _most_recent(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Most recent quarterly balance-sheet value before cutoff."""
    if df.empty:
        return pd.Series(dtype=float)
    avail = df[df.index < cutoff]
    if avail.empty:
        return pd.Series(dtype=float)
    return avail.iloc[-1]


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, NaN-aware."""
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sigma


def compute_quality(
    fundamentals: dict,
    rebalance_dates: pd.DatetimeIndex,
    filing_lag_days: int = 45,
) -> pd.DataFrame:
    """
    Quality factor: composite z-score of ROE + Profit Margin + Operating Margin.
    Higher score = higher quality.

    Required keys in `fundamentals` dict:
      'net_income', 'total_revenue', 'operating_income', 'stockholder_equity'
    Each value is a DataFrame [fiscal_quarter_dates x tickers].
    """
    log.info("Computing QUALITY factor ...")
    ni = fundamentals.get("net_income", pd.DataFrame())
    rev = fundamentals.get("total_revenue", pd.DataFrame())
    oi = fundamentals.get("operating_income", pd.DataFrame())
    eq = fundamentals.get("stockholder_equity", pd.DataFrame())

    if any(x.empty for x in [ni, rev, oi, eq]):
        log.warning("QUALITY: fundamentals incomplete, returning empty factor.")
        return pd.DataFrame(index=rebalance_dates)

    lag = pd.Timedelta(days=filing_lag_days)
    scores: dict = {}

    for date in rebalance_dates:
        cutoff = date - lag
        ni_ttm = _ttm(ni, cutoff)
        rev_ttm = _ttm(rev, cutoff)
        oi_ttm = _ttm(oi, cutoff)
        eq_rec = _most_recent(eq, cutoff)

        if ni_ttm.empty or rev_ttm.empty:
            scores[date] = pd.Series(dtype=float)
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            roe = (ni_ttm / eq_rec.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            pm = (ni_ttm / rev_ttm.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            om = (oi_ttm / rev_ttm.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        # Winsorize to ±3 sigma before z-scoring
        for arr in [roe, pm, om]:
            mu, sig = arr.mean(), arr.std()
            arr.clip(mu - 3 * sig, mu + 3 * sig, inplace=True)

        composite = _zscore(roe) + _zscore(pm) + _zscore(om)
        scores[date] = composite

    df = pd.DataFrame(scores).T
    df.index.name = "date"
    return df


def compute_value(
    prices: pd.DataFrame,
    fundamentals: dict,
    rebalance_dates: pd.DatetimeIndex,
    filing_lag_days: int = 45,
) -> pd.DataFrame:
    """
    Value factor: composite of -P/B + -P/E + -EV/EBITDA (all negated).
    Higher score = cheaper on all three valuation metrics = stronger VALUE signal.

    Required keys in `fundamentals` dict:
      'net_income', 'stockholder_equity', 'total_debt', 'cash',
      'ebitda', 'shares_outstanding'
    """
    log.info("Computing VALUE factor ...")
    ni = fundamentals.get("net_income", pd.DataFrame())
    eq = fundamentals.get("stockholder_equity", pd.DataFrame())
    debt = fundamentals.get("total_debt", pd.DataFrame())
    cash = fundamentals.get("cash", pd.DataFrame())
    ebitda = fundamentals.get("ebitda", pd.DataFrame())
    shares = fundamentals.get("shares_outstanding", pd.DataFrame())

    if any(x.empty for x in [ni, eq, shares]):
        log.warning("VALUE: fundamentals incomplete, returning empty factor.")
        return pd.DataFrame(index=rebalance_dates)

    lag = pd.Timedelta(days=filing_lag_days)
    scores: dict = {}

    for date in rebalance_dates:
        cutoff = date - lag
        ni_ttm = _ttm(ni, cutoff)
        eq_rec = _most_recent(eq, cutoff)
        debt_rec = _most_recent(debt, cutoff).fillna(0) if not debt.empty else pd.Series(0, index=ni_ttm.index)
        cash_rec = _most_recent(cash, cutoff).fillna(0) if not cash.empty else pd.Series(0, index=ni_ttm.index)
        ebitda_ttm = _ttm(ebitda, cutoff) if not ebitda.empty else pd.Series(dtype=float)
        shares_rec = _most_recent(shares, cutoff)

        if ni_ttm.empty or eq_rec.empty or shares_rec.empty:
            scores[date] = pd.Series(dtype=float)
            continue

        # Price as of rebalance date
        price_row = prices[prices.index < date].iloc[-1]

        # Market cap
        mktcap = price_row * shares_rec

        with np.errstate(divide="ignore", invalid="ignore"):
            pb = (mktcap / eq_rec.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            pe = (mktcap / ni_ttm.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            ev = mktcap + debt_rec - cash_rec
            ev_ebitda = (ev / ebitda_ttm.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) if not ebitda_ttm.empty else pd.Series(np.nan, index=pb.index)

        # Exclude negative earnings/book (distressed/loss-making)
        pb = pb.where(pb > 0)
        pe = pe.where(pe > 0)
        ev_ebitda = ev_ebitda.where(ev_ebitda > 0)

        # Value = lower multiple is better; negate so higher score = cheaper
        components = [_zscore(-pb), _zscore(-pe)]
        if not ev_ebitda.dropna().empty:
            components.append(_zscore(-ev_ebitda))

        composite = sum(components)
        scores[date] = composite

    df = pd.DataFrame(scores).T
    df.index.name = "date"
    return df
