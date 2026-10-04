"""
conviction_signal.py -- a better insider-BUY signal, tested OUT-OF-SAMPLE.

The basic univariate signals (side, seniority, cluster, contrarian) are weak on the
full bias-free universe. The literature says insider informativeness concentrates in
small names bought with genuine conviction and breadth (Lakonishok-Lee 2001;
Cohen-Malloy-Pomorski 2012). We combine four economically-motivated, look-ahead-free
ingredients into a 0-4 **conviction score** for each open-market purchase:

    +1  small_cap    lag_market_cap below the TRAIN-period median
    +1  big_stake    holding_fraction > 0.10 (adds >10% to own position)
    +1  breadth      buy_cluster OR joint_owner_count >= 2 (multiple/again buyers)
    +1  senior       officer or C-suite (role-based seniority >= 2)

No weights are fitted (equal-weight count) to avoid overfitting; the ONLY parameter
learned on train is the market-cap median. Train = 2006-2015; the score is then
frozen and evaluated on 2016-2025 it has never seen.

Data: the bias-free full-universe event table (excess-vs-XBI 1q returns,
delisting + price-anomaly handling applied) + CRSP daily panels for calendar-time.
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
COL = "excess_XBI_1q"
TRAIN_END = "2015-12-31"


def load() -> pd.DataFrame:
    """Load the handoff event table and add a 1-3 ``seniority`` code.

    3 = CEO/CFO/COO/Chair-President, 2 = other officer, 1 = everyone else
    (directors, 10% owners).
    """
    d = pd.read_parquet(ANALYSIS / "event_results.parquet")
    d["signal_date"] = pd.to_datetime(d["signal_date"])
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    top = {"CEO", "CFO", "COO", "Chair / President"}
    d["seniority"] = np.where(d["role"].isin(top), 3, np.where(d["role"] == "Other officer", 2, 1))
    return d


def add_conviction(d: pd.DataFrame, mktcap_median: float) -> pd.DataFrame:
    """Add the 0-4 ``conviction`` score to each event.

    One point each for: market cap below ``mktcap_median`` (small cap); the trade
    adding more than 10% to the insider's holding (big stake); clustered buying or
    two or more owners on the filing (breadth); officer or C-suite (senior).
    """
    d = d.copy()
    small = d["lag_market_cap"] < mktcap_median
    # holding_fraction is missing for joint filings and brand-new positions (no
    # prior holding); fillna(0) scores those as "not a big stake".
    big_stake = d["holding_fraction"].fillna(0) > 0.10
    # Note: joint_owner_count counts every reporting owner on one filing, which
    # includes affiliated entities (e.g. a fund and its general partner) reporting
    # the same trade; buy_cluster already requires distinct single-owner filings.
    breadth = (d["buy_cluster"] == True) | (d["joint_owner_count"] >= 2)
    senior = d["seniority"] >= 2
    d["conviction"] = (small.astype(int) + big_stake.astype(int)
                       + breadth.astype(int) + senior.astype(int))
    return d


def _key(f):
    """Rename ``permno`` to ``ticker``, the key column the backtest helpers expect."""
    return f.drop(columns=[c for c in ["ticker"] if c in f.columns]).rename(columns={"permno": "ticker"})


def boot(sub):
    """Firm-bootstrap the mean 1-quarter excess return of ``sub``.

    Returns None when there are fewer than 5 firms or 40 events.
    """
    s = _key(sub.dropna(subset=[COL])).rename(columns={COL: "r"})
    if s["ticker"].nunique() < 5 or len(s) < 40:
        return None
    return B.cluster_bootstrap(s, "r", nboot=1000)


def main():
    """Fit the size cutoff on 2006-2015, then report the frozen score on 2016-2025.

    Prints the in-sample gradient by score, the out-of-sample event-level results,
    and out-of-sample calendar-time books (all buys, high-conviction buys, and
    the long/short of high-conviction buys against all sells).
    """
    d = load()
    buys = d[(d.side == 1) & (d.XBI_status_1q == "ok")].copy()
    # The only fitted parameter: the median market cap of training-period buys.
    train_mask = buys.signal_date <= TRAIN_END
    mktcap_median = buys.loc[train_mask, "lag_market_cap"].median()
    buys = add_conviction(buys, mktcap_median)
    print(f"train median market cap = ${mktcap_median/1e6:,.0f}M  "
          f"(train buys {train_mask.sum()}, test buys {(~train_mask).sum()})")

    print("\n=== IN-SAMPLE (2006-2015): mean 1q excess vs XBI by conviction score ===")
    tr = buys[train_mask]
    for sc in range(5):
        r = boot(tr[tr.conviction == sc])
        if r:
            print(f"  score {sc}: n={r['n']:5d} mean={r['mean']:+6.2%} p={r['p']:.3f}")

    print("\n=== OUT-OF-SAMPLE (2016-2025): FROZEN score ===")
    te = buys[~train_mask]
    for label, sub in [("all buys (baseline)", te),
                       ("low conviction (<=1)", te[te.conviction <= 1]),
                       ("high conviction (>=3)", te[te.conviction >= 3]),
                       ("top conviction (==4)", te[te.conviction == 4])]:
        r = boot(sub)
        if r:
            st = "***" if r['p'] < .01 else "**" if r['p'] < .05 else "*" if r['p'] < .10 else ""
            print(f"  {label:26s} n={r['n']:5d} mean={r['mean']:+6.2%} "
                  f"CI=[{r['ci_lo']:+.2%},{r['ci_hi']:+.2%}] p={r['p']:.3f}{st}")

    # ---- calendar-time OOS backtest: high-conviction buys, long & beta-neutral L/S ----
    print("\n=== OUT-OF-SAMPLE calendar-time (hold 63 sessions ~1q) ===")
    panels = C.load_crsp_panels()
    xbi = C.load_xbi()["ret"]
    def bt(ev, name, ls=False):
        """Print calendar-time performance for a long book of events ``ev``."""
        st = B.perf_stats(B.calendar_daily_returns(_key(ev), 63, panels, long_short=ls),
                          bench=xbi, name=name)
        if "cagr" not in st:
            print(f"  {name}: few")
            return
        print(f"  {name:30s} n={len(ev):5d} CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
              f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
              f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
    te_all = d[(d.signal_date > TRAIN_END) & d.entry_date.notna()]
    te_b = add_conviction(te_all[te_all.side == 1], mktcap_median)
    te_s = te_all[te_all.side == -1]
    hi = te_b[te_b.conviction >= 3]
    bt(te_b, "all buys (OOS)")
    bt(hi, "high-conviction buys (OOS)")
    # dollar-neutral L/S: long high-conviction buys, short all sells (beta hedge)
    ld = B.calendar_daily_returns(_key(hi), 63, panels)
    sd = B.calendar_daily_returns(_key(te_s), 63, panels)
    idx = ld.index.union(sd.index)
    lsd = 0.5 * ld.reindex(idx).fillna(0) - 0.5 * sd.reindex(idx).fillna(0)
    st = B.perf_stats(lsd, bench=xbi, name="LS")
    print(f"  {'L/S hi-conv buys vs sells':30s}       CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
          f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
          f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
    stx = B.perf_stats(xbi.loc["2016":], name="XBI")
    print(f"  {'XBI (2016-2025)':30s}       CAGR={stx['cagr']:+6.1%} Sharpe={stx['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
