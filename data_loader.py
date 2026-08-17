"""
data_loader.py
--------------
Downloads and caches NIFTY 500 price/volume data from Yahoo Finance.

SURVIVORSHIP BIAS WARNING:
    The constituent list is scraped from the current NIFTY 500.  Stocks that
    were in the index historically but have since been delisted, merged, or
    removed are NOT included.  This creates an upward bias in backtest returns.
    There is no freely available point-in-time constituent history for India.
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# NIFTY 500 constituent list
# ──────────────────────────────────────────────────────────────────────────────

NIFTY500_NSE_URL = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
)

# Fallback: a representative ~100-stock subset of large NIFTY 500 names.
# Used only when the live NSE download fails.
FALLBACK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
    "KOTAKBANK", "LT", "SBIN", "BAJFINANCE", "BHARTIARTL", "ASIANPAINT",
    "HCLTECH", "MARUTI", "AXISBANK", "ITC", "SUNPHARMA", "TITAN",
    "ULTRACEMCO", "WIPRO", "NESTLEIND", "TECHM", "POWERGRID", "NTPC",
    "ONGC", "JSWSTEEL", "TATASTEEL", "COALINDIA", "BPCL", "DRREDDY",
    "CIPLA", "DIVISLAB", "HEROMOTOCO", "BAJAJFINSV", "GRASIM", "ADANIPORTS",
    "HINDALCO", "EICHERMOT", "TATACONSUM", "APOLLOHOSP", "BRITANNIA",
    "SHREECEM", "SBILIFE", "HDFCLIFE", "PIDILITIND", "HAVELLS",
    "DABUR", "BERGEPAINT", "MARICO", "COLPAL", "GODREJCP", "MUTHOOTFIN",
    "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "INDUSINDBK", "RBLBANK",
    "TATAPOWER", "ADANIENT", "DMART", "ZOMATO", "NYKAA", "PAYTM",
    "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "IOB",
    "SAIL", "NMDC", "MOIL", "NATIONALUM",
    "AMBUJACEM", "ACC", "RAMCOCEM",
    "GODREJPROP", "DLF", "PRESTIGE", "OBEROIRLTY",
    "TRENT", "PAGEIND", "MCDOWELL-N", "RADICO",
    "IPCALAB", "AUROPHARMA", "LUPIN", "TORNTPHARM", "ALKEM", "GLENMARK",
    "CHOLAFIN", "M&MFIN", "BAJAJ-AUTO", "ESCORTS", "ASHOKLEY", "M&M",
    "TATAMOTORS", "MOTHERSON", "BOSCHLTD",
    "HDFCAMC", "ICICIGI", "ICICIPRULI",
    "RECLTD", "PFC", "IRFC", "NHPC",
    "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS",
    "ZYDUSLIFE", "BIOCON", "NATCO", "GRANULES",
    "AAPL",  # intentional dud – tests error handling
]


def get_nifty500_tickers(use_cache: bool = True) -> list[str]:
    """
    Return NSE ticker symbols for NIFTY 500 constituents with .NS suffix.
    Tries live NSE CSV first; falls back to bundled list on failure.
    """
    cache_file = CACHE_DIR / "nifty500_tickers.txt"
    if use_cache and cache_file.exists():
        tickers = cache_file.read_text().splitlines()
        log.info("Loaded %d tickers from cache.", len(tickers))
        return tickers

    tickers = _fetch_nse_csv()
    if not tickers:
        log.warning("NSE download failed – using fallback symbol list.")
        tickers = [s for s in FALLBACK_SYMBOLS if s != "AAPL"]  # drop dud

    tickers_ns = [f"{s}.NS" for s in tickers]
    cache_file.write_text("\n".join(tickers_ns))
    log.info("Saved %d tickers to cache.", len(tickers_ns))
    return tickers_ns


def _fetch_nse_csv() -> list[str]:
    """Download NIFTY 500 CSV from NSE and return Symbol column."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        session = requests.Session()
        # Warm up session cookie
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        resp = session.get(NIFTY500_NSE_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        symbols = df["Symbol"].dropna().str.strip().tolist()
        log.info("Fetched %d symbols from NSE.", len(symbols))
        return symbols
    except Exception as exc:
        log.warning("NSE fetch failed: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Price / volume download
# ──────────────────────────────────────────────────────────────────────────────

def download_price_data(
    tickers: list[str],
    start: str = "2009-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    batch_size: int = 50,
    sleep_between_batches: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices, raw close, and volume for all tickers.

    Returns
    -------
    adj_close : DataFrame  [dates × tickers]
    close     : DataFrame  [dates × tickers]
    volume    : DataFrame  [dates × tickers]
    """
    adj_close_cache = CACHE_DIR / "adj_close.parquet"
    volume_cache    = CACHE_DIR / "volume.parquet"

    if use_cache and adj_close_cache.exists() and volume_cache.exists():
        log.info("Loading price data from parquet cache …")
        adj_close = pd.read_parquet(adj_close_cache)
        volume    = pd.read_parquet(volume_cache)
        log.info("Cache loaded: %d stocks, %d days.", adj_close.shape[1], adj_close.shape[0])
        return adj_close, adj_close, volume

    log.info("Downloading data for %d tickers in batches of %d …", len(tickers), batch_size)

    all_close:  dict[str, pd.Series] = {}
    all_volume: dict[str, pd.Series] = {}
    failed: list[str] = []

    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    for idx, batch in enumerate(batches, 1):
        log.info("  Batch %d/%d (%d tickers) …", idx, len(batches), len(batch))
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.warning("  Batch %d download error: %s", idx, exc)
            failed.extend(batch)
            continue

        # yfinance returns MultiIndex columns when >1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            close_df  = raw["Close"]
            volume_df = raw["Volume"]
        else:
            # Single ticker (shouldn't happen with batch, but guard anyway)
            close_df  = raw[["Close"]].rename(columns={"Close": batch[0]})
            volume_df = raw[["Volume"]].rename(columns={"Volume": batch[0]})

        for ticker in batch:
            if ticker in close_df.columns:
                all_close[ticker]  = close_df[ticker]
                all_volume[ticker] = volume_df[ticker]
            else:
                failed.append(ticker)

        if idx < len(batches):
            time.sleep(sleep_between_batches)

    if failed:
        log.warning("Failed / missing tickers (%d): %s", len(failed), failed[:10])

    adj_close = pd.DataFrame(all_close).sort_index()
    volume    = pd.DataFrame(all_volume).sort_index()

    # Remove entirely-empty columns
    adj_close = adj_close.dropna(how="all", axis=1)
    volume    = volume.dropna(how="all", axis=1)

    # Winsorize single-day returns > 75% to suppress bad corporate-action data
    rets = adj_close.pct_change()
    adj_close = _remove_extreme_return_tickers(adj_close, rets, threshold=0.75)
    volume = volume[adj_close.columns]

    log.info(
        "Download complete: %d valid tickers, %d trading days.",
        adj_close.shape[1], adj_close.shape[0],
    )

    adj_close.to_parquet(adj_close_cache)
    volume.to_parquet(volume_cache)
    log.info("Saved to parquet cache.")

    return adj_close, adj_close, volume


def _remove_extreme_return_tickers(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    threshold: float = 0.75,
    max_allowed_extreme_days: int = 5,
) -> pd.DataFrame:
    """
    Drop tickers that have more than `max_allowed_extreme_days` single-day
    returns exceeding `threshold`. These usually indicate bad data.
    """
    extreme_counts = (returns.abs() > threshold).sum()
    bad = extreme_counts[extreme_counts > max_allowed_extreme_days].index.tolist()
    if bad:
        log.warning("Dropping %d tickers with extreme return spikes: %s", len(bad), bad[:10])
    return prices.drop(columns=bad, errors="ignore")


def load_benchmark_data(
    start: str = "2009-01-01",
    end: Optional[str] = None,
) -> dict[str, pd.Series]:
    """
    Download NIFTY 50 (^NSEI) and NIFTY 500 (^CRSLDX) index levels.
    Returns dict of {name: price_series}.
    """
    cache_file = CACHE_DIR / "benchmarks.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        return {col: df[col].dropna() for col in df.columns}

    symbols = {"NIFTY50": "^NSEI", "NIFTY500": "^CRSLDX"}
    result: dict[str, pd.Series] = {}
    for name, sym in symbols.items():
        try:
            raw = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if not raw.empty:
                result[name] = raw["Close"].squeeze().rename(name)
                log.info("Downloaded benchmark %s (%d days).", name, len(result[name]))
            else:
                log.warning("No data for benchmark %s.", name)
        except Exception as exc:
            log.warning("Benchmark %s download failed: %s", name, exc)

    df = pd.DataFrame(result)
    df.to_parquet(cache_file)
    return result


def clear_cache() -> None:
    """Delete all cached files to force a fresh download."""
    for f in CACHE_DIR.glob("*"):
        f.unlink()
    log.info("Cache cleared.")
