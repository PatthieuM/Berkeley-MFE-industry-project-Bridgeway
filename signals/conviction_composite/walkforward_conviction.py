"""
walkforward_conviction.py -- walk-forward (rolling) test of the conviction signal.

Instead of one 2006-2015 train / 2016-2025 test split, every month is out-of-sample:
at each formation month t the ONLY fitted parameter (the small-cap market-cap
cutoff) is recomputed from purchases observed strictly BEFORE t (expanding window,
requires >= MIN_HIST prior buys), the frozen 0-4 conviction score is applied, and
positions are held forward. Uses the whole 2006-2025 history as OOS -> best-powered
honest estimate. Bias-free CRSP returns.

Conviction score per open-market buy (+1 each): small-cap (below rolling median),
holding_fraction>0.10, breadth (buy_cluster or joint_owner_count>=2), senior (>=2).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest as B
import crsp_reprice as C

import paths
ANALYSIS = paths.ANALYSIS
MIN_HIST = 500          # need this many prior buys before we trust the rolling cutoff
HOLD = 63               # ~1 quarter


def load():
    """Load the handoff event table and precompute the three score inputs that need no fitting.

    Adds ``seniority`` (3 = CEO/CFO/COO/Chair-President, 2 = other officer,
    1 = other), ``big_stake`` (trade adds more than 10% to the holding),
    ``breadth`` (clustered buying or two or more owners on the filing) and
    ``senior`` (seniority of 2 or more).
    """
    d = pd.read_parquet(ANALYSIS / "event_results.parquet")
    d["signal_date"] = pd.to_datetime(d["signal_date"])
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    top = {"CEO", "CFO", "COO", "Chair / President"}
    d["seniority"] = np.where(d["role"].isin(top), 3, np.where(d["role"] == "Other officer", 2, 1))
    d["big_stake"] = (d["holding_fraction"].fillna(0) > 0.10)
    d["breadth"] = (d["buy_cluster"] == True) | (d["joint_owner_count"] >= 2)
    d["senior"] = d["seniority"] >= 2
    return d


def rolling_conviction(buys: pd.DataFrame) -> pd.Series:
    """Conviction score with a small-cap cutoff fit on strictly prior buys (expanding)."""
    b = buys.sort_values("signal_date")
    caps = b["lag_market_cap"].to_numpy(float)
    # Expanding median of the market caps of all earlier rows in date order; shift(1)
    # excludes the row itself. Buys filed earlier on the same date count as prior.
    roll_med = pd.Series(caps).expanding().median().shift(1).to_numpy()
    small = caps < roll_med                       # NaN cutoff (early rows) -> False
    small = np.where(np.isnan(roll_med), False, small)
    score = (small.astype(int)
             + b["big_stake"].to_numpy().astype(int)
             + b["breadth"].to_numpy().astype(int)
             + b["senior"].to_numpy().astype(int))
    n_prior = np.arange(len(b))                   # #buys observed before this one
    score = np.where(n_prior >= MIN_HIST, score, np.nan)   # not enough history yet
    return pd.Series(score, index=b.index)


def _key(f):
    """Rename ``permno`` to ``ticker``, the key column the backtest helpers expect."""
    return f.drop(columns=[c for c in ["ticker"] if c in f.columns]).rename(columns={"permno": "ticker"})


def main():
    """Score every buy walk-forward and report event-level and calendar-time results.

    The first ``MIN_HIST`` buys only seed the rolling size cutoff and are not
    scored. Calendar-time books: all buys, high-conviction buys (score of 3 or
    more), and a long/short of high-conviction buys against all sells.
    """
    d = load()
    buys = d[(d.side == 1)].copy()
    buys["conviction"] = rolling_conviction(buys)
    usable = buys.dropna(subset=["conviction"])
    print(f"walk-forward: {len(usable)} buys scored OOS "
          f"({usable.signal_date.min().date()}–{usable.signal_date.max().date()}); "
          f"first {MIN_HIST} buys used only to seed the rolling cutoff")

    # event-level, whole history OOS
    print("\n=== EVENT-LEVEL (all walk-forward, 1q excess vs XBI, firm bootstrap) ===")
    okc = usable[usable.XBI_status_1q == "ok"]
    for lab, sub in [("all buys", okc),
                     ("low conviction (<=1)", okc[okc.conviction <= 1]),
                     ("high conviction (>=3)", okc[okc.conviction >= 3]),
                     ("top conviction (==4)", okc[okc.conviction == 4])]:
        s = _key(sub.dropna(subset=["excess_XBI_1q"])).rename(columns={"excess_XBI_1q": "r"})
        if s["ticker"].nunique() < 5 or len(s) < 40:
            print(f"  {lab:24s} (too few)")
            continue
        r = B.cluster_bootstrap(s, "r", nboot=1500)
        st = "***" if r['p'] < .01 else "**" if r['p'] < .05 else "*" if r['p'] < .10 else ""
        print(f"  {lab:24s} n={r['n']:5d} mean={r['mean']:+6.2%} "
              f"CI=[{r['ci_lo']:+.2%},{r['ci_hi']:+.2%}] p={r['p']:.3f}{st}")

    # calendar-time, whole history OOS
    print("\n=== CALENDAR-TIME (walk-forward, hold 63 sessions) ===")
    panels = C.load_crsp_panels()
    xbi = C.load_xbi()["ret"]
    hi = usable[(usable.conviction >= 3) & usable.entry_date.notna()]
    sells = d[(d.side == -1) & d.entry_date.notna()]
    def show(ev, name, ls=False):
        """Print calendar-time performance of a long book of events and return its stats."""
        st = B.perf_stats(B.calendar_daily_returns(_key(ev), HOLD, panels, long_short=ls), bench=xbi, name=name)
        if "cagr" not in st:
            print(f"  {name}: few")
            return None
        print(f"  {name:30s} n={len(ev):5d} CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
              f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
              f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
        return st
    show(usable[usable.entry_date.notna()], "all buys")
    ld = B.calendar_daily_returns(_key(hi), HOLD, panels)
    show(hi, "high-conviction buys (long)")
    sd = B.calendar_daily_returns(_key(sells), HOLD, panels)
    # Long/short: half long the high-conviction book, half short the sells book.
    # The short leg carries no edge of its own; it hedges out biotech beta.
    idx = ld.index.union(sd.index)
    lsd = 0.5 * ld.reindex(idx).fillna(0) - 0.5 * sd.reindex(idx).fillna(0)
    st = B.perf_stats(lsd, bench=xbi, name="LS")
    print(f"  {'L/S hi-conv buys vs sells':30s}       CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
          f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
          f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
    stx = B.perf_stats(xbi.loc["2007":], name="XBI")
    print(f"  {'XBI (walk-forward window)':30s}       CAGR={stx['cagr']:+6.1%} Sharpe={stx['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
