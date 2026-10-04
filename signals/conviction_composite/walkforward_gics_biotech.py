"""
walkforward_gics_biotech.py -- walk-forward conviction signal on the STRICT GICS
biotech universe (GICS 35201010), not the SIC 2834/2836 set which mixes in pharma.

Restricts events to (permno, signal_date) pairs where the firm was a point-in-time
member of the GICS-biotech universe at the trade date, using
universe/biotech_universe_intervals.parquet (under BRIDGEWAY_DATA), then runs the same rolling
walk-forward conviction test as walkforward_conviction.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest as B
import crsp_reprice as C
import walkforward_conviction as W

import paths
UNIV = paths.UNIVERSE
MIN_HIST, HOLD = W.MIN_HIST, W.HOLD


def biotech_intervals() -> pd.DataFrame:
    """Load the strict-GICS biotech membership intervals (``is_biotech`` rows only).

    An open-ended interval (no ``effective_end``) is treated as running to 2100.
    """
    u = pd.read_parquet(UNIV, columns=["permno", "effective_start", "effective_end", "is_biotech"])
    u = u[u["is_biotech"] == True].copy()
    u["permno"] = pd.to_numeric(u["permno"], errors="coerce").astype("Int64")
    u["effective_start"] = pd.to_datetime(u["effective_start"])
    u["effective_end"] = pd.to_datetime(u["effective_end"]).fillna(pd.Timestamp("2100-01-01"))
    return u.dropna(subset=["permno"])


def in_universe(events: pd.DataFrame, iv: pd.DataFrame) -> pd.Series:
    """True where the event's permno was GICS-biotech at signal_date (point-in-time)."""
    by = {int(p): g[["effective_start", "effective_end"]].to_numpy()
          for p, g in iv.groupby("permno")}
    sd = pd.to_datetime(events["signal_date"]).to_numpy()
    pm = pd.to_numeric(events["permno"], errors="coerce").to_numpy()
    out = np.zeros(len(events), bool)
    for i in range(len(events)):
        if not np.isfinite(pm[i]):
            continue
        rows = by.get(int(pm[i]))
        if rows is None:
            continue
        d = sd[i]
        out[i] = bool(((rows[:, 0] <= d) & (rows[:, 1] >= d)).any())
    return pd.Series(out, index=events.index)


def main():
    """Restrict events to strict GICS biotech at the trade date, then run the walk-forward test."""
    d = W.load()
    iv = biotech_intervals()
    # Keep only events whose firm was a strict-GICS biotech member on the signal date.
    keep = in_universe(d, iv)
    d = d[keep].copy()
    print(f"GICS-biotech universe: {iv.permno.nunique()} permnos | "
          f"events in universe at trade date: {len(d)} across {d.permno.nunique()} firms "
          f"({d.signal_date.min().date()}–{d.signal_date.max().date()})")

    buys = d[d.side == 1].copy()
    buys["conviction"] = W.rolling_conviction(buys)
    usable = buys.dropna(subset=["conviction"])
    print(f"walk-forward buys scored OOS: {len(usable)}")

    print("\n=== EVENT-LEVEL (walk-forward, 1q excess vs XBI, firm bootstrap) ===")
    okc = usable[usable.XBI_status_1q == "ok"]
    for lab, sub in [("all buys", okc),
                     ("low conviction (<=1)", okc[okc.conviction <= 1]),
                     ("high conviction (>=3)", okc[okc.conviction >= 3])]:
        s = W._key(sub.dropna(subset=["excess_XBI_1q"])).rename(columns={"excess_XBI_1q": "r"})
        if s["ticker"].nunique() < 5 or len(s) < 40:
            print(f"  {lab:24s} (too few)")
            continue
        r = B.cluster_bootstrap(s, "r", nboot=1500)
        st = "***" if r['p'] < .01 else "**" if r['p'] < .05 else "*" if r['p'] < .10 else ""
        print(f"  {lab:24s} n={r['n']:5d} mean={r['mean']:+6.2%} "
              f"CI=[{r['ci_lo']:+.2%},{r['ci_hi']:+.2%}] p={r['p']:.3f}{st}")

    print("\n=== CALENDAR-TIME (walk-forward, hold 63 sessions) ===")
    panels = C.load_crsp_panels()
    xbi = C.load_xbi()["ret"]
    hi = usable[(usable.conviction >= 3) & usable.entry_date.notna()]
    sells = d[(d.side == -1) & d.entry_date.notna()]
    def show(ev, name):
        """Print calendar-time performance of a long book of events."""
        st = B.perf_stats(B.calendar_daily_returns(W._key(ev), HOLD, panels), bench=xbi, name=name)
        if "cagr" not in st:
            print(f"  {name}: few")
            return
        print(f"  {name:30s} n={len(ev):5d} CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
              f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
              f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
    show(usable[usable.entry_date.notna()], "all buys")
    show(hi, "high-conviction buys (long)")
    ld = B.calendar_daily_returns(W._key(hi), HOLD, panels)
    sd = B.calendar_daily_returns(W._key(sells), HOLD, panels)
    # Long/short: half long the high-conviction book, half short the sells book.
    # The short leg carries no edge of its own; it hedges out biotech beta.
    idx = ld.index.union(sd.index)
    lsd = 0.5 * ld.reindex(idx).fillna(0) - 0.5 * sd.reindex(idx).fillna(0)
    st = B.perf_stats(lsd, bench=xbi, name="LS")
    print(f"  {'L/S hi-conv buys vs sells':30s}       CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
          f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
          f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
    stx = B.perf_stats(xbi.loc["2007":], name="XBI")
    print(f"  {'XBI':30s}       CAGR={stx['cagr']:+6.1%} Sharpe={stx['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
