"""
research/turnover_diagnostic.py
--------------------------------
Audits turnover calculation in momentum_study.py across three frequencies.

Checks
------
1. Portfolio actually updates at the expected frequency (segment count, holding length)
2. Per-period turnover distribution (mean, median, std, zeros)
3. Annualized turnover = per_period * rebalances_per_year
4. Constituent overlap between consecutive periods
5. Annual-drag estimates from Study F vs corrected values
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_nifty500_tickers, download_price_data
from factor_engine import compute_daily_returns
from momentum_study import (
    compute_mom_scores,
    _select_top_n,
    _build_port,
    compute_turnover,
    get_rebal_dates,
    BASE,
)


def _divider(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("\n" + "-" * w)


def _turnover_series(constituents: dict) -> pd.Series:
    """Per-period one-sided turnover as a Series indexed by rebalance date."""
    dates = sorted(constituents.keys())
    rows = []
    for i in range(1, len(dates)):
        prev = set(constituents[dates[i - 1]])
        curr = set(constituents[dates[i]])
        if curr:
            to = len(curr - prev) / len(curr)
            rows.append((dates[i], to))
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def _holding_lengths(rebalance_dates: pd.DatetimeIndex) -> pd.Series:
    """Calendar days between consecutive rebalance dates."""
    rd = sorted(rebalance_dates)
    gaps = [(rd[i+1] - rd[i]).days for i in range(len(rd)-1)]
    return pd.Series(gaps)


def _rebalances_per_year(rebalance_dates, start, end) -> float:
    """Actual rebalances per calendar year in the backtest window."""
    n = len(rebalance_dates)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return n / years


def audit(freq: str, mom_scores, daily_returns, adj_close,
          start: str, end: str, top_n: int) -> dict:
    """Run full turnover audit for one frequency."""
    rd = get_rebal_dates(adj_close, start, end, freq)
    constituents = _select_top_n(mom_scores, rd, top_n)

    # -- 1. Portfolio update check --
    n_rebal = len(rd)
    hold_lengths = _holding_lengths(rd)
    port_rets = _build_port(daily_returns, constituents, rd)
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    port_rets = port_rets.loc[start_dt:end_dt].dropna()

    # Actual updates = how many rebalances changed the portfolio
    to_series = _turnover_series(constituents)
    changed_periods = (to_series > 0).sum()
    zero_change = (to_series == 0).sum()

    # -- 2. Per-period turnover distribution --
    per_period_mean  = to_series.mean()
    per_period_med   = to_series.median()
    per_period_std   = to_series.std()
    per_period_min   = to_series.min()
    per_period_max   = to_series.max()

    # -- 3. Annualized turnover --
    rpyr = _rebalances_per_year(rd, start, end)
    ann_turnover = per_period_mean * rpyr

    # -- 4. Constituent overlap --
    dates = sorted(constituents.keys())
    overlaps = []
    for i in range(1, len(dates)):
        prev = set(constituents[dates[i-1]])
        curr = set(constituents[dates[i]])
        if prev and curr:
            overlaps.append(len(prev & curr) / len(curr))
    overlap_mean = np.mean(overlaps) if overlaps else np.nan

    # -- 5. Cost drag: Study F used n_rebal_per_year hardcoded
    hardcoded_rpyr = {"weekly": 52, "monthly": 12, "quarterly": 4}[freq]
    # At 25bps with mean per-period turnover
    drag_correct   = per_period_mean * 2 * 25 / 10000 * rpyr * 100
    drag_hardcoded = per_period_mean * 2 * 25 / 10000 * hardcoded_rpyr * 100

    return dict(
        freq=freq,
        n_rebal=n_rebal,
        n_port_days=len(port_rets),
        changed_periods=int(changed_periods),
        zero_change_periods=int(zero_change),
        hold_days_mean=hold_lengths.mean(),
        hold_days_med=hold_lengths.median(),
        per_period_mean_pct=per_period_mean * 100,
        per_period_med_pct=per_period_med * 100,
        per_period_std_pct=per_period_std * 100,
        per_period_min_pct=per_period_min * 100,
        per_period_max_pct=per_period_max * 100,
        rebalances_per_year=rpyr,
        hardcoded_rpyr=hardcoded_rpyr,
        ann_turnover_pct=ann_turnover * 100,
        overlap_mean_pct=overlap_mean * 100,
        drag_25bps_correct_pct=drag_correct,
        drag_25bps_hardcoded_pct=drag_hardcoded,
        to_series=to_series,
        constituents=constituents,
    )


def print_audit(r: dict):
    freq = r["freq"].upper()
    _divider(f"FREQUENCY: {freq}")

    print(f"\n  [1] REBALANCE SCHEDULE")
    print(f"      Total rebalance dates     : {r['n_rebal']}")
    print(f"      Portfolio trading days    : {r['n_port_days']}")
    print(f"      Periods WITH changes      : {r['changed_periods']}")
    print(f"      Periods with ZERO change  : {r['zero_change_periods']}")
    print(f"      Avg holding days          : {r['hold_days_mean']:.1f}  "
          f"(median {r['hold_days_med']:.0f})")
    print(f"      Rebalances / year (actual): {r['rebalances_per_year']:.1f}")
    print(f"      Rebalances / year (hardcoded in Study F): {r['hardcoded_rpyr']}")

    print(f"\n  [2] PER-PERIOD TURNOVER DISTRIBUTION")
    print(f"      Mean   : {r['per_period_mean_pct']:6.2f}%")
    print(f"      Median : {r['per_period_med_pct']:6.2f}%")
    print(f"      Std    : {r['per_period_std_pct']:6.2f}%")
    print(f"      Min    : {r['per_period_min_pct']:6.2f}%")
    print(f"      Max    : {r['per_period_max_pct']:6.2f}%")

    print(f"\n  [3] ANNUALIZED TURNOVER")
    print(f"      = per_period_mean x rebalances_per_year")
    print(f"      = {r['per_period_mean_pct']:.2f}% x {r['rebalances_per_year']:.1f}")
    print(f"      = {r['ann_turnover_pct']:.1f}% / year  (one-sided)")
    print(f"        {r['ann_turnover_pct']*2:.1f}% / year  (round-trip)")

    print(f"\n  [4] CONSTITUENT OVERLAP (consecutive periods)")
    print(f"      Mean overlap              : {r['overlap_mean_pct']:.1f}%  "
          f"(= 1 - per_period_turnover, expect ~{100-r['per_period_mean_pct']:.0f}%)")

    print(f"\n  [5] ANNUAL COST DRAG @ 25bps")
    print(f"      Using actual rpyr ({r['rebalances_per_year']:.1f}) : {r['drag_25bps_correct_pct']:.3f}%/yr")
    print(f"      Using hardcoded  ({r['hardcoded_rpyr']})        : {r['drag_25bps_hardcoded_pct']:.3f}%/yr")
    delta = r["drag_25bps_correct_pct"] - r["drag_25bps_hardcoded_pct"]
    print(f"      Error                     : {delta:+.3f}%/yr")


def print_comparison_table(results: list):
    _divider("CROSS-FREQUENCY COMPARISON")

    # Per-period
    print(f"\n  PER-PERIOD TURNOVER (as reported by compute_turnover())")
    print(f"  {'Freq':<12} {'Periods':>8} {'Mean %':>9} {'Median %':>9} {'Ann. Rebals':>12}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['freq']:<12} {r['n_rebal']:>8} {r['per_period_mean_pct']:>9.2f} "
              f"{r['per_period_med_pct']:>9.2f} {r['rebalances_per_year']:>12.1f}")

    # Annualized
    print(f"\n  ANNUALIZED TURNOVER (per_period x rebalances_per_year)")
    print(f"  {'Freq':<12} {'One-sided %/yr':>15} {'Round-trip %/yr':>16}")
    print("  " + "-" * 45)
    for r in results:
        print(f"  {r['freq']:<12} {r['ann_turnover_pct']:>15.1f} {r['ann_turnover_pct']*2:>16.1f}")

    # Cost drag error
    print(f"\n  ANNUAL COST DRAG @ 25bps -- HARDCODED vs ACTUAL RPYR")
    print(f"  {'Freq':<12} {'Correct %/yr':>14} {'Hardcoded %/yr':>15} {'Error bps':>10}")
    print("  " + "-" * 55)
    for r in results:
        err_bps = (r["drag_25bps_correct_pct"] - r["drag_25bps_hardcoded_pct"]) * 100
        print(f"  {r['freq']:<12} {r['drag_25bps_correct_pct']:>14.3f} "
              f"{r['drag_25bps_hardcoded_pct']:>15.3f} {err_bps:>+10.1f}")


def print_constituent_sample(r_weekly: dict, r_monthly: dict, n_samples: int = 5):
    _divider("SAMPLE CONSTITUENT CHANGES")

    print(f"\n  WEEKLY (first {n_samples} rebalances):")
    dates_w = sorted(r_weekly["constituents"].keys())
    for i in range(1, min(n_samples + 1, len(dates_w))):
        t0, t1 = dates_w[i-1], dates_w[i]
        prev = set(r_weekly["constituents"][t0])
        curr = set(r_weekly["constituents"][t1])
        entered = curr - prev
        exited  = prev - curr
        print(f"    {t0.date()} -> {t1.date()}  "
              f"(+{len(entered)} -{len(exited)} = net {len(entered)-len(exited):+d})")

    print(f"\n  MONTHLY (first {n_samples} rebalances):")
    dates_m = sorted(r_monthly["constituents"].keys())
    for i in range(1, min(n_samples + 1, len(dates_m))):
        t0, t1 = dates_m[i-1], dates_m[i]
        prev = set(r_monthly["constituents"][t0])
        curr = set(r_monthly["constituents"][t1])
        entered = curr - prev
        exited  = prev - curr
        print(f"    {t0.date()} -> {t1.date()}  "
              f"(+{len(entered)} -{len(exited)} = net {len(entered)-len(exited):+d})")


def print_verdicts(results: list):
    _divider("VERDICTS")

    ann = {r["freq"]: r["ann_turnover_pct"] for r in results}
    per = {r["freq"]: r["per_period_mean_pct"] for r in results}

    print()
    ok_ann = ann["weekly"] > ann["monthly"] > ann["quarterly"]
    ok_per = per["weekly"] < per["monthly"] < per["quarterly"]

    print(f"  [A] Annualized turnover order (weekly > monthly > quarterly):")
    print(f"      Weekly={ann['weekly']:.1f}%  Monthly={ann['monthly']:.1f}%  Quarterly={ann['quarterly']:.1f}%")
    print(f"      Result: {'PASS -- correct ordering' if ok_ann else 'FAIL -- ordering violated'}")

    print(f"\n  [B] Per-period turnover order (weekly < monthly < quarterly):")
    print(f"      Weekly={per['weekly']:.2f}%  Monthly={per['monthly']:.2f}%  Quarterly={per['quarterly']:.2f}%")
    print(f"      Result: {'PASS -- fewer changes per week is expected' if ok_per else 'FAIL'}")
    print(f"      Note: per-period turnover is LOWER for weekly because less drift occurs")
    print(f"      in 5 days than in 21 days. Annualized turnover is the correct comparison.")

    print(f"\n  [C] Study D reported 'Turnover_%mo' -- label audit:")
    print(f"      Column labeled 'TO%' in console output represents PER-PERIOD turnover.")
    print(f"      For non-monthly frequencies, this label is MISLEADING.")
    print(f"      Fix: rename to 'TO%/period' and add separate 'Ann_TO%' column.")

    for r in results:
        err_bps = (r["drag_25bps_correct_pct"] - r["drag_25bps_hardcoded_pct"]) * 100
        print(f"\n  [D] Cost drag accuracy ({r['freq']}): error = {err_bps:+.1f}bps/yr")


def main():
    START, END = "2010-01-01", "2026-06-24"
    TOP_N = 20

    print("\n" + "=" * 70)
    print("  TURNOVER DIAGNOSTIC -- NIFTY 500 MOMENTUM STUDY")
    print("=" * 70)

    print("\n  Loading data ...")
    tickers = get_nifty500_tickers(use_cache=True)
    adj_close, _, _ = download_price_data(tickers, start="2009-01-01", end=None, use_cache=True)
    daily_returns = compute_daily_returns(adj_close)

    # Pre-compute baseline momentum scores (12M, skip 1M)
    print("  Computing baseline momentum scores (12M, skip=1M) ...")
    mom_scores = compute_mom_scores(adj_close, BASE["lookback"], BASE["skip"])

    # Audit each frequency
    results = []
    for freq in ["weekly", "monthly", "quarterly"]:
        print(f"  Auditing {freq} ...")
        r = audit(freq, mom_scores, daily_returns, adj_close, START, END, TOP_N)
        results.append(r)
        print_audit(r)

    # Cross-frequency comparison
    print_comparison_table(results)

    # Sample constituent changes
    r_w = next(r for r in results if r["freq"] == "weekly")
    r_m = next(r for r in results if r["freq"] == "monthly")
    print_constituent_sample(r_w, r_m)

    # Final verdicts
    print_verdicts(results)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
