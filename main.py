"""
main.py
-------
Orchestrates the MAX factor backtest for Indian equities (NIFTY 500 universe).

Usage:
    python main.py                          # full run
    python main.py --clear-cache            # delete cached data and re-download
    python main.py --start 2015-01-01       # shorter backtest window
    python main.py --no-charts              # skip chart generation

SURVIVORSHIP BIAS WARNING:
    This backtest uses the *current* NIFTY 500 constituent list.
    Historically delisted or removed companies are excluded, which
    creates an upward bias in all reported returns.
"""

import argparse
import logging
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── project modules ────────────────────────────────────────────────────────────
from data_loader import (
    get_nifty500_tickers,
    download_price_data,
    load_benchmark_data,
    clear_cache,
)
from factor_engine import (
    compute_daily_returns,
    compute_max_factor,
    get_all_decile_constituents,
    factor_summary,
)
from portfolio import (
    get_rebalance_dates,
    build_portfolio_returns,
    build_equal_weight_universe_returns,
    build_benchmark_returns,
    build_long_short_spread,
    build_all_decile_returns,
    cumulative_returns,
)
from performance import (
    metrics_table,
    print_metrics_table,
    plot_equity_curve,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_monthly_heatmap,
    plot_factor_spread,
    plot_dashboard,
)


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAX Factor Backtest – India Equities")
    p.add_argument("--start",        default="2010-01-01", help="Backtest start date")
    p.add_argument("--end",          default=None,         help="Backtest end date (default: today)")
    p.add_argument("--lookback",     type=int, default=21, help="Lookback window in trading days")
    p.add_argument("--n-max",        type=int, default=5,  help="Number of top returns to average")
    p.add_argument("--n-deciles",    type=int, default=10, help="Number of deciles")
    p.add_argument("--min-adtv",     type=float, default=1.0, help="Min avg daily traded value (₹ crore)")
    p.add_argument("--clear-cache",  action="store_true",  help="Delete cached data and re-download")
    p.add_argument("--no-charts",    action="store_true",  help="Skip chart generation")
    p.add_argument("--no-cache",     action="store_true",  help="Ignore cache, force fresh download")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("  MAX FACTOR BACKTEST – INDIAN EQUITIES (NIFTY 500 UNIVERSE)")
    print("=" * 70)
    print(f"  Period  : {args.start} → {args.end or 'today'}")
    print(f"  MAX({args.n_max}) lookback : {args.lookback} trading days")
    print(f"  Deciles : {args.n_deciles}")
    print(f"  Min ADTV: ₹{args.min_adtv} crore")
    print()
    print("  ⚠  SURVIVORSHIP BIAS: using current NIFTY 500 constituents only.")
    print("     Historical delistings and removals are NOT included.")
    print("=" * 70 + "\n")

    # ── 1. Data ────────────────────────────────────────────────────────────────
    if args.clear_cache:
        log.info("Clearing cache …")
        clear_cache()

    use_cache = not args.no_cache

    log.info("Step 1/5 – Loading NIFTY 500 tickers …")
    tickers = get_nifty500_tickers(use_cache=use_cache)
    log.info("%d tickers loaded.", len(tickers))

    log.info("Step 2/5 – Downloading price & volume data …")
    # Download from 1 year before backtest start to warm up the lookback
    data_start = str(int(args.start[:4]) - 1) + args.start[4:]
    adj_close, _, volume = download_price_data(
        tickers, start=data_start, end=args.end, use_cache=use_cache
    )

    benchmarks = load_benchmark_data(start=data_start, end=args.end)

    # ── 2. Returns & rebalance dates ───────────────────────────────────────────
    log.info("Step 3/5 – Computing daily returns and rebalance schedule …")
    daily_returns = compute_daily_returns(adj_close)

    rebalance_dates = get_rebalance_dates(adj_close, start=args.start, end=args.end)

    # Volume-weighted average daily traded value (price × volume)
    adtv_series: pd.DataFrame = adj_close * volume  # ₹ value traded

    # ── 3. Factor ──────────────────────────────────────────────────────────────
    log.info("Step 4/5 – Computing MAX(%d) factor …", args.n_max)
    factor_df = compute_max_factor(
        returns=daily_returns,
        rebalance_dates=rebalance_dates,
        lookback=args.lookback,
        n_max=args.n_max,
        min_obs_ratio=0.7,
        min_volume_series=adtv_series,
        min_adtv_crore=args.min_adtv,
    )

    fsum = factor_summary(factor_df)
    log.info("Factor cross-sectional stats (first 3 rows):\n%s", fsum.head(3).to_string())

    # ── 4. Portfolios ──────────────────────────────────────────────────────────
    log.info("Step 5/5 – Building portfolios …")

    all_constituents = get_all_decile_constituents(factor_df, n_deciles=args.n_deciles)

    bottom_dec = all_constituents[1]
    mid_dec    = all_constituents[args.n_deciles // 2]
    top_dec    = all_constituents[args.n_deciles]

    ret_bottom = build_portfolio_returns(
        daily_returns, bottom_dec, rebalance_dates, label="MAX_Bottom_D1"
    )
    ret_top = build_portfolio_returns(
        daily_returns, top_dec, rebalance_dates, label="MAX_Top_D10"
    )
    ret_ew = build_equal_weight_universe_returns(daily_returns, rebalance_dates)
    ret_spread = build_long_short_spread(ret_bottom, ret_top)

    # Benchmark returns
    bench_returns: dict[str, pd.Series] = {}
    for name, prices in benchmarks.items():
        bench_returns[name] = build_benchmark_returns(prices, label=name)

    # All decile returns (for the spread chart)
    decile_returns = build_all_decile_returns(daily_returns, all_constituents, rebalance_dates)

    # Align everything to backtest period
    start_dt = pd.Timestamp(args.start)
    end_dt   = pd.Timestamp(args.end) if args.end else pd.Timestamp.today()

    def trim(s: pd.Series) -> pd.Series:
        return s.loc[start_dt:end_dt].dropna()

    returns_dict: dict[str, pd.Series] = {
        "MAX_Bottom_D1": trim(ret_bottom),
        "MAX_Top_D10":   trim(ret_top),
        "EW_Universe":   trim(ret_ew),
        "MAX_LS_Spread": trim(ret_spread),
    }
    for name, r in bench_returns.items():
        returns_dict[name] = trim(r)

    decile_returns_trimmed = {d: trim(r) for d, r in decile_returns.items()}

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    metrics_df = metrics_table(
        {k: v for k, v in returns_dict.items() if k != "MAX_LS_Spread"}
    )
    spread_metrics = metrics_table({"MAX_LS_Spread": returns_dict.get("MAX_LS_Spread", pd.Series())})

    print_metrics_table(metrics_df)
    print("\nLong-Short Spread Metrics:")
    print_metrics_table(spread_metrics)

    # Save metrics to CSV
    full_metrics = pd.concat([metrics_df, spread_metrics])
    full_metrics.to_csv("backtest_metrics.csv")
    log.info("Metrics saved to backtest_metrics.csv")

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    if not args.no_charts:
        log.info("Generating charts …")

        chart_returns = {k: v for k, v in returns_dict.items()}

        plot_equity_curve(chart_returns)
        plot_drawdown(chart_returns)
        plot_rolling_sharpe(chart_returns)
        plot_monthly_heatmap(returns_dict.get("MAX_Bottom_D1", pd.Series(name="MAX_Bottom_D1")))
        plot_factor_spread(decile_returns_trimmed)
        plot_dashboard(chart_returns, decile_returns_trimmed)

        log.info("All charts saved to charts/ directory.")

    log.info("Backtest complete.")
    print("\nDone. Charts → charts/   Metrics → backtest_metrics.csv   Log → backtest.log")


if __name__ == "__main__":
    main()
