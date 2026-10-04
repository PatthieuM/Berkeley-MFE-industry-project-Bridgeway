"""
walkforward_full_gics.py -- walk-forward conviction backtest on the FULL GICS
universe: the existing handoff GICS events PLUS the freshly-crawled events for the
firms he never collected (CONVICTION_OUTPUT_DIR/gics_missing_events.parquet), priced on the
combined CRSP panels (handoff + data/crsp_prices from fetch_crsp_prices.py).

Steps:
1. Build features + 1-quarter forward excess-vs-XBI returns for the crawled
   events (to match the columns the walk-forward needs).
2. Concatenate with the handoff GICS events (same columns).
3. Run the identical rolling walk-forward conviction test.

Run AFTER the crawl (scrape_gics_events.py) and price fetch finish:
    CRSP_EXTRA_DIR=<extra CRSP panels> python signals/conviction_composite/walkforward_full_gics.py
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest as B
import crsp_reprice as C
import walkforward_conviction as W
import walkforward_gics_biotech as G

import paths
CRAWLED = paths.OUTPUT / "gics_missing_events.parquet"
HOLD = 63
_TOP = ("chief executive", "ceo", "president", "chair", "chief financial", "cfo",
        "chief operating", "coo")
_OFF = ("chief", "officer", "vice president", "vp", "general counsel", "treasurer", "secretary")

# columns the walk-forward consumes
COLS = ["permno", "signal_date", "entry_date", "side", "seniority",
        "holding_fraction", "buy_cluster", "joint_owner_count",
        "lag_market_cap", "excess_XBI_1q", "XBI_status_1q"]


def _seniority(title, is_off, is_dir, is_ten):
    """Map a crawled filer to the 0-3 seniority scale used by the handoff events.

    3 = CEO/CFO/COO/president/chair by title, 2 = other officer, 1 = director or
    10% owner, 0 = none of these.
    """
    t = str(title or "").lower()
    if any(k in t for k in _TOP):
        return 3
    if is_off or any(k in t for k in _OFF):
        return 2
    if is_dir or is_ten:
        return 1
    return 0


def load_price_panels():
    """Return ``{permno: (dates, daily total returns, market cap in $ thousands)}``.

    Reads the handoff CRSP panels plus any panels in ``CRSP_EXTRA_DIR``; on a
    duplicate (permno, date) the handoff row wins.
    """
    dirs = [C.MKT] + ([Path(os.environ["CRSP_EXTRA_DIR"])] if os.environ.get("CRSP_EXTRA_DIR") else [])
    files = [f for d in dirs for f in sorted(glob.glob(str(d / "daily_*.parquet")))]
    cols = ["permno", "date", "dlyret", "dlyclose", "dlyhigh", "dlycap"]
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    a = pd.concat(frames, ignore_index=True)
    a["date"] = pd.to_datetime(a["date"])
    a = a.drop_duplicates(["permno", "date"], keep="first").sort_values(["permno", "date"])
    panels = {}
    for pm, g in a.groupby("permno"):
        panels[int(pm)] = (g["date"].values.astype("datetime64[ns]"),
                           g["dlyret"].to_numpy("float64"),
                           g["dlycap"].to_numpy("float64"))
    return panels


def build_crawled_features(raw: pd.DataFrame, panels, xbi) -> pd.DataFrame:
    """Build the walk-forward input columns (``COLS``) for crawled events.

    Computes per-filing owner counts, seniority, holding fraction, entry date
    (first session after the filing), lagged market cap, and the 63-session
    excess return over XBI with its status (``ok`` or
    ``calendar_right_censored``), then flags clustered buying.
    """
    xd = xbi.index.values.astype("datetime64[ns]")
    xr = xbi.values
    raw = raw.copy()
    raw["signal_date"] = pd.to_datetime(raw["signal_date"])
    raw["shares"] = pd.to_numeric(raw["shares"], errors="coerce")
    # per-filing breadth
    jo = raw.groupby("accession")["owner_cik"].transform("nunique")
    raw["joint_owner_count"] = jo.fillna(1).astype(int)
    raw["seniority"] = [_seniority(t, o, d, x) for t, o, d, x in zip(
        raw.get("officer_title"), raw.get("is_officer"), raw.get("is_director"),
        raw.get("is_ten_percent_owner"))]
    # Holding before the trade = holding after it minus the signed trade size.
    # Unlike the handoff table, joint filings are not blanked and the ratio is capped at 10.
    prior = raw["shares_owned_following"] - raw["side"] * raw["shares"]
    raw["holding_fraction"] = (raw["shares"] / prior.where(prior > 0, np.nan)).clip(upper=10)

    sig = raw["signal_date"].values.astype("datetime64[ns]")
    n = len(raw)
    entry = np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    cap = np.full(n, np.nan)
    exq = np.full(n, np.nan)
    status = np.array(["missing_entry_close"] * n, dtype=object)
    clus = np.zeros(n, bool)
    pm_arr = raw["permno"].astype(int).values
    buy = raw[raw.side == 1]   # used below for buy_cluster
    for i in range(n):
        v = panels.get(int(pm_arr[i]))
        if v is None:
            continue
        d, r, c = v
        pos = int(np.searchsorted(d, sig[i], side="right"))
        if pos >= len(d):
            continue
        entry[i] = d[pos]
        cap[i] = c[pos - 1] * 1000 if pos > 0 and np.isfinite(c[pos - 1]) else np.nan  # dlycap is $thousands
        # Excess return over the 63-session hold; windows running past the data are censored.
        end = pos + HOLD
        if end < len(r):
            seg = r[pos + 1:end + 1]
            seg = seg[np.isfinite(seg)]
            fwd = np.prod(1 + seg) - 1 if len(seg) else np.nan
            xp = int(np.searchsorted(xd, sig[i], side="right"))
            xseg = xr[xp + 1:xp + 1 + HOLD] if xp < len(xr) else np.array([])
            xseg = xseg[np.isfinite(xseg)]
            xfwd = np.prod(1 + xseg) - 1 if len(xseg) else np.nan
            if np.isfinite(fwd) and np.isfinite(xfwd):
                exq[i] = fwd - xfwd
                status[i] = "ok"
            else:
                status[i] = "calendar_right_censored"
        else:
            status[i] = "calendar_right_censored"
    raw["entry_date"] = entry
    raw["lag_market_cap"] = cap
    raw["excess_XBI_1q"] = exq
    raw["XBI_status_1q"] = status
    # buy_cluster: >=2 distinct buyers on the same permno within the trailing 90
    # calendar days. The handoff table instead uses 20 sessions and only
    # single-owner filings, so the two definitions differ.
    raw["buy_cluster"] = False
    for pm, gidx in buy.groupby("permno").groups.items():
        g = raw.loc[gidx].sort_values("signal_date")
        ds = g["signal_date"].values
        ow = g["owner_cik"].values
        for j, idx in enumerate(g.index):
            lo = ds[j] - np.timedelta64(90, "D")
            win = (ds <= ds[j]) & (ds >= lo)
            raw.at[idx, "buy_cluster"] = len(set(ow[win])) >= 2
    return raw[COLS]


def main():
    """Combine crawled and handoff GICS events and run the walk-forward conviction test."""
    if not CRAWLED.exists():
        print(f"crawl output not found: {CRAWLED}\nrun scrape_gics_events.py first.")
        return
    panels_cap = load_price_panels()
    xbi = C.load_xbi()["ret"]
    raw = pd.read_parquet(CRAWLED)
    print(f"crawled events: {len(raw)} across {raw.permno.nunique()} firms")
    new = build_crawled_features(raw, panels_cap, xbi)
    print(f"  featured: {new.excess_XBI_1q.notna().sum()} priced (1q)")

    # existing handoff GICS events, same columns
    d = W.load()
    d = d[G.in_universe(d, G.biotech_intervals())]
    old = d[COLS].copy()
    combined = pd.concat([old, new], ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    combined["entry_date"] = pd.to_datetime(combined["entry_date"])
    print(f"COMBINED GICS universe: {len(combined)} events across {combined.permno.nunique()} firms "
          f"(was {old.permno.nunique()} handoff-only)")

    # derive conviction inputs + rolling score
    combined["big_stake"] = combined["holding_fraction"].fillna(0) > 0.10
    combined["breadth"] = (combined["buy_cluster"] == True) | (combined["joint_owner_count"] >= 2)
    combined["senior"] = combined["seniority"] >= 2
    buys = combined[combined.side == 1].copy()
    buys["conviction"] = W.rolling_conviction(buys)
    usable = buys.dropna(subset=["conviction"])

    print("\n=== EVENT-LEVEL (walk-forward, 1q excess vs XBI, firm bootstrap) ===")
    okc = usable[usable.XBI_status_1q == "ok"]
    for lab, sub in [("all buys", okc), ("low conviction (<=1)", okc[okc.conviction <= 1]),
                     ("high conviction (>=3)", okc[okc.conviction >= 3])]:
        s = W._key(sub.dropna(subset=["excess_XBI_1q"])).rename(columns={"excess_XBI_1q": "r"})
        if s["ticker"].nunique() < 5 or len(s) < 40:
            print(f"  {lab:24s} (too few)")
            continue
        r = B.cluster_bootstrap(s, "r", nboot=1500)
        st = "***" if r['p'] < .01 else "**" if r['p'] < .05 else "*" if r['p'] < .10 else ""
        print(f"  {lab:24s} n={r['n']:5d} mean={r['mean']:+6.2%} p={r['p']:.3f}{st}")

    print("\n=== CALENDAR-TIME (walk-forward, hold 63 sessions) ===")
    panels = C.load_crsp_panels()
    xr = C.load_xbi()["ret"]
    hi = usable[(usable.conviction >= 3) & usable.entry_date.notna()]
    sells = combined[(combined.side == -1) & combined.entry_date.notna()]
    def show(ev, name):
        """Print calendar-time performance of a long book of events."""
        st = B.perf_stats(B.calendar_daily_returns(W._key(ev), HOLD, panels), bench=xr, name=name)
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
    st = B.perf_stats(lsd, bench=xr, name="LS")
    print(f"  {'L/S hi-conv buys vs sells':30s}       CAGR={st['cagr']:+6.1%} Sharpe={st['sharpe']:+.2f} "
          f"tNW={st['t_nw']:+.2f} alpha={st.get('alpha_ann',np.nan):+6.1%} "
          f"aT={st.get('alpha_t',np.nan):+.2f} beta={st.get('beta_xbi',np.nan):+.2f}")
    stx = B.perf_stats(xr.loc["2007":], name="XBI")
    print(f"  {'XBI':30s}       CAGR={stx['cagr']:+6.1%} Sharpe={stx['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
