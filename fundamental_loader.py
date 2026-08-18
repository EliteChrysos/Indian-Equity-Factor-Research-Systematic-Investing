"""
research/fundamental_loader.py
-------------------------------
Downloads and caches quarterly fundamental data from yfinance.

Output: dict of DataFrames [fiscal_quarter_dates x tickers] for each metric.
Cached to research/cache/fundamentals.pkl (one file, all metrics).

Metrics extracted:
  Income statement (quarterly):
    net_income, total_revenue, operating_income, ebitda
  Balance sheet (quarterly):
    stockholder_equity, total_debt, cash, shares_outstanding
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "fundamentals.pkl"

# Row names to try, in priority order, for each metric
_INCOME_ROWS = {
    "net_income": [
        "Net Income", "NetIncome", "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ],
    "total_revenue": [
        "Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue",
    ],
    "operating_income": [
        "Operating Income", "OperatingIncome", "EBIT",
        "Operating Income Or Loss",
    ],
    "ebitda": [
        "EBITDA", "Ebitda",
        "Normalized EBITDA",
    ],
}

_BALANCE_ROWS = {
    "stockholder_equity": [
        "Stockholders Equity", "Total Stockholder Equity",
        "Stockholder Equity", "Common Stock Equity",
        "Total Equity Gross Minority Interest",
        "Stockholders Equity Net Of Treasury Stock",
    ],
    "total_debt": [
        "Total Debt", "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Total Liabilities Net Minority Interest",
    ],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Cash Equivalents And Federal Funds Sold",
        "Cash", "Cash Equivalents",
    ],
    "shares_outstanding": [
        "Ordinary Shares Number", "Share Issued",
        "Common Stock", "Common Shares Outstanding",
    ],
}


def _first_match(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return first row from df that matches any candidate name."""
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return pd.Series(dtype=float)


def _download_one(ticker: str) -> dict[str, pd.Series]:
    """Download and parse quarterly financials for one ticker."""
    result: dict[str, pd.Series] = {}
    t = yf.Ticker(ticker)

    # Income statement
    for attr in ["quarterly_financials", "quarterly_income_stmt"]:
        try:
            qf = getattr(t, attr, None)
            if qf is not None and not qf.empty:
                for metric, candidates in _INCOME_ROWS.items():
                    s = _first_match(qf, candidates)
                    if not s.empty and metric not in result:
                        result[metric] = s.sort_index()
                break
        except Exception:
            continue

    # Balance sheet
    for attr in ["quarterly_balance_sheet", "quarterly_balancesheet"]:
        try:
            qb = getattr(t, attr, None)
            if qb is not None and not qb.empty:
                for metric, candidates in _BALANCE_ROWS.items():
                    s = _first_match(qb, candidates)
                    if not s.empty and metric not in result:
                        result[metric] = s.sort_index()
                break
        except Exception:
            continue

    return result


def download_fundamentals(
    tickers: list[str],
    use_cache: bool = True,
    sleep_per_ticker: float = 0.25,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download quarterly fundamental data for all tickers.

    Returns
    -------
    dict: metric_name -> DataFrame [fiscal_dates x tickers]
    Fiscal dates are fiscal quarter end dates from yfinance.
    """
    if use_cache and not force_refresh and CACHE_FILE.exists():
        log.info("Loading fundamentals cache: %s", CACHE_FILE)
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    all_metrics = list(_INCOME_ROWS) + list(_BALANCE_ROWS)
    raw: dict[str, dict[str, pd.Series]] = {m: {} for m in all_metrics}
    n = len(tickers)

    log.info("Downloading fundamentals for %d tickers (est. %.0f min) ...", n, n * sleep_per_ticker / 60)

    for i, ticker in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            log.info("  ... %d/%d tickers", i, n)
        try:
            per_ticker = _download_one(ticker)
            for metric, series in per_ticker.items():
                if not series.empty:
                    raw[metric][ticker] = series
        except Exception as e:
            log.debug("Skip %s: %s", ticker, e)
        time.sleep(sleep_per_ticker)

    # Assemble wide DataFrames [dates x tickers]
    fundamentals: dict[str, pd.DataFrame] = {}
    for metric, ticker_dict in raw.items():
        if not ticker_dict:
            fundamentals[metric] = pd.DataFrame()
            log.warning("  %s: no data for any ticker", metric)
            continue
        df = pd.DataFrame(ticker_dict)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        fundamentals[metric] = df
        log.info("  %s: %d dates x %d tickers", metric, len(df), df.shape[1])

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(fundamentals, f)
    log.info("Fundamentals cached to %s", CACHE_FILE)

    return fundamentals


def get_coverage_report(fundamentals: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    """Return a coverage summary showing how many tickers have data per metric."""
    rows = []
    for metric, df in fundamentals.items():
        if df.empty:
            rows.append({"metric": metric, "tickers_covered": 0, "date_range": "—", "pct": 0.0})
            continue
        covered = df.columns.intersection(tickers)
        date_range = f"{df.index.min().date()} → {df.index.max().date()}"
        rows.append({
            "metric": metric,
            "tickers_covered": len(covered),
            "date_range": date_range,
            "pct": round(100 * len(covered) / len(tickers), 1),
        })
    return pd.DataFrame(rows).set_index("metric")
