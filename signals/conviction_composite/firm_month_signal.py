"""
firm_month_signal.py -- cross-sectional firm-month net-insider-conviction signal.

Rather than trading individual filings (noisy, overlapping), aggregate insider
activity to a **firm x calendar-month** panel, rank firms cross-sectionally each
month by net insider demand, and hold a **monthly-rebalanced** long/short book.
Monthly rebalancing => non-overlapping monthly returns => honest t-stats, and it
matches the one effect that flickered in the earlier 20-year study (net dollar flow).

Formation at end of month m (all filings public by then); hold month m+1 (no
look-ahead). Returns are firm monthly total return minus XBI (bias-free CRSP,
delisting included).

Two scores (each ranked cross-sectionally within the month):
  breadth   = #distinct buyers - #distinct sellers
  conv_net  = sum(conviction over buys) - #sells        (conviction from
              conviction_signal.py: small-cap + big-stake + breadth + senior)

Train = form months <= 2015-12; frozen and evaluated on 2016-2025.
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
TRAIN_END = pd.Period("2015-12", "M")   # informational; main() sets the cutoff per frequency


_RULE = {"M": "ME", "Q": "QE"}   # pandas resample rule for each period frequency


def period_returns(panels, freq="M"):
    """period(Period) x permno matrix of total returns from CRSP daily."""
    cols = {}
    for pm, p in panels.items():
        m = (1 + p["ret"]).resample(_RULE[freq]).prod() - 1
        m.index = m.index.to_period(freq)
        cols[pm] = m
    return pd.DataFrame(cols)


def build_firm_month(events, mktcap_median, freq="M"):
    """Aggregate events to one row per firm and period with the two ranking scores.

    Per (permno, period): distinct buyers and sellers, summed conviction over
    buys, and number of sells. ``breadth`` = buyers - sellers; ``conv_net`` =
    summed buy conviction - sells. The conviction score is the same 0-4 score as
    in ``conviction_signal.py``, with a fixed size cutoff ``mktcap_median``.
    """
    e = events.copy()
    e["month"] = e["signal_date"].dt.to_period(freq)
    top = {"CEO", "CFO", "COO", "Chair / President"}
    e["seniority"] = np.where(e["role"].isin(top), 3, np.where(e["role"] == "Other officer", 2, 1))
    small = e["lag_market_cap"] < mktcap_median
    big = e["holding_fraction"].fillna(0) > 0.10
    breadth = (e["buy_cluster"] == True) | (e["joint_owner_count"] >= 2)
    senior = e["seniority"] >= 2
    e["conviction"] = small.astype(int) + big.astype(int) + breadth.astype(int) + senior.astype(int)
    buys = e[e.side == 1]
    sells = e[e.side == -1]
    g = e.groupby(["permno", "month"])
    fm = pd.DataFrame({
        "n_buyers": buys.groupby(["permno", "month"])["owner_cik"].nunique(),
        "n_sellers": sells.groupby(["permno", "month"])["owner_cik"].nunique(),
        "conv_buy": buys.groupby(["permno", "month"])["conviction"].sum(),
        "n_sells": sells.groupby(["permno", "month"]).size(),
    })
    fm = fm.fillna(0.0)
    fm["breadth"] = fm["n_buyers"] - fm["n_sellers"]
    fm["conv_net"] = fm["conv_buy"] - fm["n_sells"]
    return fm.reset_index()


def backtest_xs(fm, mret, xbi_m, score, q=3, name="", verbose=True):
    """Monthly-rebalanced cross-sectional long/short on `score` (tercile by default)."""
    rows = {}
    long_excess = {}
    for month, grp in fm.groupby("month"):
        # Form on period `month`, earn the next period's return (no look-ahead).
        nxt = month + 1
        if nxt not in mret.index:
            continue
        grp = grp[grp["permno"].isin(mret.columns)]
        if len(grp) < 6:
            continue
        r_next = mret.loc[nxt]
        xb = xbi_m.get(nxt, np.nan)
        s = grp.set_index("permno")[score]
        # need dispersion to rank
        if s.nunique() < 3:
            continue
        hi = s[s >= s.quantile(1 - 1 / q)].index   # top tercile: long
        lo = s[s <= s.quantile(1 / q)].index       # bottom tercile: short
        rl = r_next.reindex(hi).dropna()
        rs = r_next.reindex(lo).dropna()
        if len(rl) < 2 or len(rs) < 2:
            continue
        rows[nxt] = rl.mean() - rs.mean()                 # long-short
        if np.isfinite(xb):
            long_excess[nxt] = rl.mean() - xb              # long-only excess vs XBI
    ls = pd.Series(rows).sort_index()
    lx = pd.Series(long_excess).sort_index()
    return ls, lx


def stats_periodic(m, name, ppy):
    """Format CAGR, volatility, Sharpe and Newey-West t for a periodic return series.

    ``ppy`` is periods per year (12 monthly, 4 quarterly).
    """
    m = m.dropna()
    if len(m) < 12:
        return f"{name}: too few periods"
    ann = m.mean() * ppy
    vol = m.std() * np.sqrt(ppy)
    sh = ann / vol if vol else np.nan
    t = B._nw_t(m.values)
    cum = (1 + m).prod() ** (ppy / len(m)) - 1
    return (f"{name:34s} n={len(m):3d} CAGR={cum:+6.1%} vol={vol:5.1%} "
            f"Sharpe={sh:+.2f} t_NW={t:+.2f}")


def main():
    """Run both scores at monthly and quarterly rebalancing, in and out of sample.

    The size cutoff is the median market cap of 2006-2015 buys; formation
    periods up to 2015 are in-sample, later ones out-of-sample.
    """
    d = pd.read_parquet(ANALYSIS / "event_results.parquet")
    d["signal_date"] = pd.to_datetime(d["signal_date"])
    buys_train = d[(d.side == 1) & (d.signal_date <= "2015-12-31")]
    mktcap_median = buys_train["lag_market_cap"].median()
    print(f"train median cap ${mktcap_median/1e6:,.0f}M")
    print("loading CRSP panels ...")
    panels = C.load_crsp_panels()
    xbi = C.load_xbi()["ret"]

    for freq, ppy, train_end in [("M", 12, pd.Period("2015-12", "M")),
                                 ("Q", 4, pd.Period("2015Q4", "Q"))]:
        fm = build_firm_month(d, mktcap_median, freq=freq)
        ret = period_returns(panels, freq=freq)
        xb = (1 + xbi).resample(_RULE[freq]).prod() - 1
        xb.index = xb.index.to_period(freq)
        fm_tr = fm[fm.month <= train_end]
        fm_te = fm[fm.month > train_end]
        rebal = {"M": "monthly", "Q": "quarterly"}[freq]
        for score in ["breadth", "conv_net"]:
            print(f"\n===== {score} ({rebal} rebal, tercile L/S) =====")
            for label, sub in [("IN-SAMPLE 2006-2015", fm_tr), ("OUT-OF-SAMPLE 2016-2025", fm_te)]:
                ls, lx = backtest_xs(sub, ret, xb, score)
                print(f"  [{label}]")
                print("   " + stats_periodic(ls, "long/short (top-bottom tercile)", ppy))
                print("   " + stats_periodic(lx, "long-only excess vs XBI", ppy))


if __name__ == "__main__":
    main()
