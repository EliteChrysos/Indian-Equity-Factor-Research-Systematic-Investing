"""
validation_main.py
------------------
Orchestrates the complete MAX factor validation study.

Usage:
    python validation_main.py                        # full study
    python validation_main.py --skip-universe        # skip sub-universe runs (faster)
    python validation_main.py --skip-robustness      # skip the 15-cell parameter grid
    python validation_main.py --fast                 # skip both above

Outputs:
    validation_report.md          ← narrative report
    robustness_grid.csv           ← parameter robustness table
    charts/validation/*.png       ← all validation charts
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("validation.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── core modules ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import get_nifty500_tickers, download_price_data, load_benchmark_data
from factor_engine import (
    compute_daily_returns, compute_max_factor,
    get_all_decile_constituents, get_decile_constituents,
)
from portfolio import (
    get_rebalance_dates, build_portfolio_returns,
    build_equal_weight_universe_returns, build_benchmark_returns,
    build_all_decile_returns,
)
from performance import metrics_table, print_metrics_table

# ── validation modules ─────────────────────────────────────────────────────────
from validation.look_ahead_audit    import run_all_look_ahead_audits
from validation.survivorship_audit  import run_survivorship_audit
from validation.decile_analysis     import run_decile_analysis
from validation.momentum_correlation import (
    run_momentum_correlation, compute_momentum_factors,
)
from validation.factor_neutralization import run_factor_neutralization
from validation.universe_analysis   import run_universe_analysis
from validation.robustness          import run_robustness_analysis
from validation.report              import generate_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAX Factor Validation Study")
    p.add_argument("--start",           default="2010-01-01")
    p.add_argument("--end",             default=None)
    p.add_argument("--lookback",        type=int, default=21)
    p.add_argument("--n-max",           type=int, default=5)
    p.add_argument("--skip-universe",   action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--fast",            action="store_true",
                   help="Alias for --skip-universe --skip-robustness")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.skip_universe   = True
        args.skip_robustness = True

    print("\n" + "=" * 70)
    print("  MAX FACTOR VALIDATION STUDY – INDIA EQUITIES")
    print("=" * 70)
    print(f"  Backtest  : {args.start} → {args.end or 'today'}")
    print(f"  Sections  : look-ahead | survivorship | decile | momentum |")
    print(f"              neutralization", end="")
    print(f" | universe" if not args.skip_universe else "", end="")
    print(f" | robustness" if not args.skip_robustness else "", end="")
    print("\n" + "=" * 70 + "\n")

    # ── 0. Load base data (uses existing cache) ────────────────────────────
    log.info("Loading cached price data …")
    tickers = get_nifty500_tickers(use_cache=True)
    data_start = str(int(args.start[:4]) - 1) + args.start[4:]
    adj_close, _, volume = download_price_data(
        tickers, start=data_start, end=args.end, use_cache=True
    )
    benchmarks = load_benchmark_data(start=data_start, end=args.end)

    daily_returns   = compute_daily_returns(adj_close)
    vol_value       = adj_close * volume   # ₹ traded value
    rebalance_dates = get_rebalance_dates(adj_close, start=args.start, end=args.end)

    # ── Recompute base factor + portfolios ─────────────────────────────────
    log.info("Recomputing base MAX(%d) factor …", args.n_max)
    factor_df = compute_max_factor(
        daily_returns, rebalance_dates,
        lookback=args.lookback, n_max=args.n_max,
        min_adtv_crore=1.0, min_volume_series=vol_value,
    )

    all_constituents = get_all_decile_constituents(factor_df, n_deciles=10)
    decile_returns   = build_all_decile_returns(daily_returns, all_constituents, rebalance_dates)

    start_dt = pd.Timestamp(args.start)
    end_dt   = pd.Timestamp(args.end) if args.end else pd.Timestamp.today()

    def trim(s: pd.Series) -> pd.Series:
        return s.loc[start_dt:end_dt].dropna()

    decile_returns_trimmed = {d: trim(r) for d, r in decile_returns.items()}
    bottom_ret = decile_returns_trimmed[1]
    top_ret    = decile_returns_trimmed[10]

    # Benchmark
    bench_ret = pd.Series(dtype=float)
    if "NIFTY50" in benchmarks:
        bench_ret = build_benchmark_returns(benchmarks["NIFTY50"], "NIFTY50")
        bench_ret = trim(bench_ret)

    # Read base metrics from CSV if it exists, else compute
    base_metrics: dict = {}
    if Path("backtest_metrics.csv").exists():
        bm_df = pd.read_csv("backtest_metrics.csv", index_col=0)
        for label in bm_df.index:
            base_metrics[label] = bm_df.loc[label].to_dict()
    else:
        for name, r in [
            ("MAX_Bottom_D1", bottom_ret),
            ("MAX_Top_D10",   top_ret),
        ]:
            from performance import compute_metrics
            base_metrics[name] = compute_metrics(r, name)

    # ── 1. Look-ahead audit ────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("SECTION 1: Look-Ahead Bias Audit")
    log.info("=" * 50)
    look_ahead_results = run_all_look_ahead_audits(
        prices=adj_close,
        returns=daily_returns,
        rebalance_dates=rebalance_dates,
        all_constituents=all_constituents[1],
        lookback=args.lookback,
        n_max=args.n_max,
    )

    # ── 2. Survivorship audit ──────────────────────────────────────────────
    log.info("=" * 50)
    log.info("SECTION 2: Survivorship Bias Audit")
    log.info("=" * 50)
    survivorship_results = run_survivorship_audit(
        prices=adj_close,
        factor_df=factor_df,
        rebalance_dates=rebalance_dates,
        backtest_start=args.start,
    )

    # ── 3. Decile analysis ─────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("SECTION 3: Decile Analysis")
    log.info("=" * 50)
    decile_results = run_decile_analysis(
        decile_returns=decile_returns_trimmed,
        all_constituents=all_constituents,
    )

    # ── 4. Momentum correlation ────────────────────────────────────────────
    log.info("=" * 50)
    log.info("SECTION 4: Momentum Correlation")
    log.info("=" * 50)
    momentum_results = run_momentum_correlation(
        prices=adj_close,
        factor_df=factor_df,
        daily_returns=daily_returns,
        rebalance_dates=rebalance_dates,
        bottom_returns=bottom_ret,
        top_returns=top_ret,
        benchmark_returns=bench_ret,
    )

    # ── 5. Factor neutralization ───────────────────────────────────────────
    log.info("=" * 50)
    log.info("SECTION 5: Factor Neutralization")
    log.info("=" * 50)
    from validation.momentum_correlation import compute_momentum_factors
    momentum_dfs = compute_momentum_factors(adj_close, rebalance_dates)
    neutralization_results = run_factor_neutralization(
        factor_df=factor_df,
        momentum_dfs=momentum_dfs,
        daily_returns=daily_returns,
        rebalance_dates=rebalance_dates,
        raw_bottom=bottom_ret,
        raw_top=top_ret,
        backtest_start=args.start,
    )

    # ── 6. Universe analysis (optional) ───────────────────────────────────
    universe_results: dict = {}
    if not args.skip_universe:
        log.info("=" * 50)
        log.info("SECTION 6: Universe Analysis")
        log.info("=" * 50)
        universe_results = run_universe_analysis(
            all_prices=adj_close,
            all_volume_value=vol_value,
            backtest_start=args.start,
            backtest_end=args.end,
            lookback=args.lookback,
            n_max=args.n_max,
        )
    else:
        log.info("Skipping universe analysis (--skip-universe).")

    # ── 7. Robustness (optional) ───────────────────────────────────────────
    robustness_df = pd.DataFrame()
    if not args.skip_robustness:
        log.info("=" * 50)
        log.info("SECTION 7: Robustness Analysis")
        log.info("=" * 50)
        robustness_df = run_robustness_analysis(
            prices=adj_close,
            backtest_start=args.start,
            backtest_end=args.end,
            min_adtv_crore=1.0,
            vol_series=vol_value,
        )
    else:
        log.info("Skipping robustness analysis (--skip-robustness).")
        # Still try to load from CSV if it exists
        if Path("robustness_grid.csv").exists():
            robustness_df = pd.read_csv("robustness_grid.csv")
            log.info("Loaded existing robustness_grid.csv (%d rows).", len(robustness_df))

    # ── 8. Generate report ─────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("Generating validation report …")
    log.info("=" * 50)
    report = generate_report(
        look_ahead_results=look_ahead_results,
        survivorship_results=survivorship_results,
        decile_results=decile_results,
        momentum_results=momentum_results,
        neutralization_results=neutralization_results,
        universe_results=universe_results,
        robustness_df=robustness_df,
        base_metrics=base_metrics,
    )

    print("\n" + "─" * 70)
    print("  VALIDATION COMPLETE")
    print("─" * 70)
    print("  Report   → validation_report.md")
    print("  Charts   → charts/validation/")
    print("  Grid     → robustness_grid.csv")
    print("  Log      → validation.log")
    print("─" * 70 + "\n")

    # Print first 60 lines of report to stdout as preview
    preview = "\n".join(report.split("\n")[:60])
    print(preview)
    if len(report.split("\n")) > 60:
        print(f"\n... [{len(report.split(chr(10))) - 60} more lines in validation_report.md]")


if __name__ == "__main__":
    main()
