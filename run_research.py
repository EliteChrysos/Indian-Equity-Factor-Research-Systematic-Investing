"""
research/run_research.py
------------------------
Main orchestrator for the Indian Equity Factor Research Framework.

Usage:
    python research/run_research.py                   # all 6 factors
    python research/run_research.py --skip-fundamental # skip QUALITY and VALUE
    python research/run_research.py --factors MAX,MOMENTUM_6M,LOW_VOL
    python research/run_research.py --start 2015-01-01
    python research/run_research.py --refresh-fundamentals  # force re-download

Outputs:
    research/factor_research_report.md   <- narrative report
    research/charts/                     <- all charts
    research/cache/fundamentals.pkl      <- cached fundamentals
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_nifty500_tickers, download_price_data, load_benchmark_data
from factor_engine import compute_daily_returns
from portfolio import get_rebalance_dates

from factors import (
    compute_max_scores,
    compute_momentum_6m,
    compute_momentum_12m,
    compute_low_vol,
    compute_quality,
    compute_value,
)
from runner import run_factor_pipeline
from charts import (
    plot_decile_equity_curves,
    plot_decile_bars,
    plot_monotonicity,
    plot_factor_spread_curve,
    plot_factor_dashboard,
    plot_factor_comparison,
    plot_equity_comparison,
    plot_sharpe_heatmap,
)
from report import generate_report


ALL_FACTORS = ["MAX", "MOMENTUM_6M", "MOMENTUM_12M", "LOW_VOL", "QUALITY", "VALUE"]
FUNDAMENTAL_FACTORS = {"QUALITY", "VALUE"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("research/research.log", mode="w", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indian Equity Factor Research Framework")
    p.add_argument("--start", default="2010-01-01", help="Backtest start date")
    p.add_argument("--end", default=None, help="Backtest end date (default: today)")
    p.add_argument("--min-adtv", type=float, default=1.0, help="Min ADTV filter in crore")
    p.add_argument(
        "--factors",
        default=",".join(ALL_FACTORS),
        help=f"Comma-separated list of factors to run. Options: {', '.join(ALL_FACTORS)}",
    )
    p.add_argument(
        "--skip-fundamental",
        action="store_true",
        help="Skip QUALITY and VALUE factors (avoids slow fundamental download)",
    )
    p.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Force re-download of fundamental data even if cache exists",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass price data cache (re-download from yfinance)",
    )
    return p.parse_args()


def print_banner(args, factors_to_run: list[str]) -> None:
    print("\n" + "=" * 70)
    print("  INDIAN EQUITY FACTOR RESEARCH FRAMEWORK")
    print("=" * 70)
    print(f"  Universe : NIFTY 500")
    print(f"  Period   : {args.start} -> {args.end or 'today'}")
    print(f"  Factors  : {', '.join(factors_to_run)}")
    print(f"  ADTV     : >{args.min_adtv} crore")
    print("=" * 70 + "\n")


def print_summary_table(all_results: dict) -> None:
    print("\n" + "=" * 80)
    print("  FACTOR RESEARCH RESULTS")
    print("=" * 80)
    hdr = f"  {'Factor':<16} {'D1 CAGR':>9} {'D10 CAGR':>9} {'Spread':>9} {'D1 Sharpe':>10} {'D10 Sharpe':>10}"
    print(hdr)
    print("  " + "-" * 68)
    for fname, result in all_results.items():
        ss = result.get("spread_stats", {})
        d1c = ss.get("D1_CAGR", float("nan"))
        d10c = ss.get("D10_CAGR", float("nan"))
        spr = ss.get("CAGR (%)", float("nan"))
        d1sh = ss.get("D1_Sharpe", float("nan"))
        d10sh = ss.get("D10_Sharpe", float("nan"))

        def f(v):
            return f"{v:>8.1f}%" if v == v else f"{'—':>9}"
        def fs(v):
            return f"{v:>9.3f}" if v == v else f"{'—':>9}"

        print(f"  {fname:<16} {f(d1c)} {f(d10c)} {f(spr)} {fs(d1sh)} {fs(d10sh)}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    Path("research").mkdir(exist_ok=True)
    setup_logging()
    log = logging.getLogger(__name__)

    requested = [f.strip().upper() for f in args.factors.split(",")]
    factors_to_run = [f for f in requested if f in ALL_FACTORS]
    if args.skip_fundamental:
        factors_to_run = [f for f in factors_to_run if f not in FUNDAMENTAL_FACTORS]
    if not factors_to_run:
        print("No valid factors selected. Choices:", ", ".join(ALL_FACTORS))
        sys.exit(1)

    print_banner(args, factors_to_run)

    # ── 1. Load price data (from existing cache) ──────────────────────────────
    log.info("Loading price data ...")
    tickers = get_nifty500_tickers(use_cache=True)
    data_start = str(int(args.start[:4]) - 1) + args.start[4:]
    adj_close, _, volume = download_price_data(
        tickers, start=data_start, end=args.end,
        use_cache=not args.no_cache,
    )
    log.info("Price data: %d tickers x %d days", adj_close.shape[1], adj_close.shape[0])

    daily_returns = compute_daily_returns(adj_close)
    rebalance_dates = get_rebalance_dates(adj_close, start=args.start, end=args.end)
    log.info("Rebalance dates: %d periods", len(rebalance_dates))

    # ── 2. Load fundamentals if needed ────────────────────────────────────────
    fundamentals = {}
    if any(f in factors_to_run for f in FUNDAMENTAL_FACTORS):
        log.info("Loading fundamental data ...")
        from fundamental_loader import download_fundamentals, get_coverage_report
        fundamentals = download_fundamentals(
            tickers,
            use_cache=True,
            force_refresh=args.refresh_fundamentals,
        )
        coverage = get_coverage_report(fundamentals, tickers)
        log.info("Fundamentals coverage:\n%s", coverage.to_string())

    # ── 3. Compute factor scores ──────────────────────────────────────────────
    factor_dfs: dict[str, pd.DataFrame] = {}

    if "MAX" in factors_to_run:
        log.info("--- Computing MAX factor ---")
        factor_dfs["MAX"] = compute_max_scores(
            adj_close, volume, rebalance_dates, min_adtv_crore=args.min_adtv
        )

    if "MOMENTUM_6M" in factors_to_run:
        log.info("--- Computing MOMENTUM_6M factor ---")
        factor_dfs["MOMENTUM_6M"] = compute_momentum_6m(adj_close, rebalance_dates)

    if "MOMENTUM_12M" in factors_to_run:
        log.info("--- Computing MOMENTUM_12M factor ---")
        factor_dfs["MOMENTUM_12M"] = compute_momentum_12m(adj_close, rebalance_dates)

    if "LOW_VOL" in factors_to_run:
        log.info("--- Computing LOW_VOL factor ---")
        factor_dfs["LOW_VOL"] = compute_low_vol(adj_close, rebalance_dates)

    if "QUALITY" in factors_to_run:
        log.info("--- Computing QUALITY factor ---")
        factor_dfs["QUALITY"] = compute_quality(fundamentals, rebalance_dates)

    if "VALUE" in factors_to_run:
        log.info("--- Computing VALUE factor ---")
        factor_dfs["VALUE"] = compute_value(adj_close, fundamentals, rebalance_dates)

    # ── 4. Run decile pipelines ───────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for fname, factor_df in factor_dfs.items():
        log.info("--- Running decile pipeline: %s ---", fname)
        result = run_factor_pipeline(
            factor_name=fname,
            factor_df=factor_df,
            daily_returns=daily_returns,
            rebalance_dates=rebalance_dates,
            backtest_start=args.start,
            backtest_end=args.end,
        )
        all_results[fname] = result

        # Per-factor charts
        if result["decile_returns"]:
            from charts import CHART_ROOT
            save_dir = CHART_ROOT / fname.lower()
            try:
                plot_decile_equity_curves(fname, result["decile_returns"], save_dir)
                plot_decile_bars(fname, result["metrics"], save_dir)
                plot_monotonicity(fname, result["metrics"], result["monotonicity"], save_dir)
                plot_factor_spread_curve(fname, result["decile_returns"], save_dir)
                plot_factor_dashboard(fname, result["decile_returns"],
                                      result["metrics"], result["monotonicity"], save_dir)
            except Exception as e:
                log.warning("Chart error for %s: %s", fname, e)

    # ── 5. Cross-factor comparison charts ────────────────────────────────────
    log.info("Generating cross-factor comparison charts ...")
    from charts import CHART_ROOT
    try:
        plot_factor_comparison(all_results, CHART_ROOT)
        plot_equity_comparison(all_results, CHART_ROOT)
        plot_sharpe_heatmap(all_results, CHART_ROOT)
    except Exception as e:
        log.warning("Cross-factor chart error: %s", e)

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── 7. Generate report ────────────────────────────────────────────────────
    log.info("Generating factor research report ...")
    report = generate_report(all_results, backtest_start=args.start, backtest_end=args.end)

    print("=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print("  Report : research/factor_research_report.md")
    print("  Charts : research/charts/")
    print("  Log    : research/research.log")
    print("=" * 70 + "\n")

    # Preview first 50 lines of report
    preview_lines = report.split("\n")[:50]
    print("\n".join(preview_lines))
    extra = len(report.split("\n")) - 50
    if extra > 0:
        print(f"\n... [{extra} more lines in research/factor_research_report.md]")


if __name__ == "__main__":
    main()
