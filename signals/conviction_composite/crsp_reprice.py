"""
crsp_reprice.py -- re-price the insider events on bias-free CRSP data.

The team stored the authorized WRDS/CRSP daily panels (with delisting returns)
under the handoff ``market`` folder (``paths.MKT``). This re-prices the SAME
event set as ``out_scaled/insider_events.parquet`` using CRSP instead of yfinance,
so the ONLY thing that changes is the price feed -> it isolates the survivorship
effect. ``dlyret`` already includes CIZ delisting treatment, so a name that
delists mid-horizon contributes its terminal (often deeply negative) return
rather than silently vanishing.

Mapping: event ``issuer_cik`` -> PERMNO via the Compustat CCM ``cik_links`` table
(date-validated). Benchmark: XBI from ``etf_daily``. Outputs a CRSP-priced event
table with forward raw / excess / residual returns, mirroring ``returns.py``.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

import paths

MKT = paths.MKT
HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "12m": 252}
BETA_WINDOW, BETA_MIN_OBS = 252, 60


def load_crsp_panels() -> dict[int, pd.DataFrame]:
    """permno -> daily panel (index=date; close, high, ret) from CRSP dlyret.

    Reads the handoff extract, plus any extra directory named in the
    ``CRSP_EXTRA_DIR`` env var (e.g. output of src/fetch_crsp_prices.py) so the
    full universe can be priced. Duplicate (permno, date) rows keep the
    first source (the handoff extract)."""
    import os
    dirs = [MKT]
    extra = os.environ.get("CRSP_EXTRA_DIR")
    if extra:
        dirs.append(Path(extra))
    files = [f for d in dirs for f in sorted(glob.glob(str(d / "daily_*.parquet")))]
    frames = [pd.read_parquet(f, columns=["permno", "date", "dlyret", "dlyclose", "dlyhigh"])
              for f in files]
    alld = pd.concat(frames, ignore_index=True)
    alld["date"] = pd.to_datetime(alld["date"])
    alld = alld.drop_duplicates(subset=["permno", "date"], keep="first")
    panels: dict[int, pd.DataFrame] = {}
    for permno, g in alld.groupby("permno"):
        g = g.sort_values("date").set_index("date")
        panels[int(permno)] = pd.DataFrame({
            "close": g["dlyclose"].to_numpy("float64"),
            "high": g["dlyhigh"].to_numpy("float64"),
            "ret": g["dlyret"].to_numpy("float64"),   # total return incl. delisting
        }, index=g.index)
    return panels


def load_xbi() -> pd.DataFrame:
    """Return XBI daily total returns (``ret`` column, date index) from the handoff ``etf_daily``
    file.
    """
    etf = pd.read_parquet(MKT / "etf_daily.parquet")
    x = etf[etf["ticker"] == "XBI"].copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date").set_index("date")
    return pd.DataFrame({"ret": x["dlyret"].to_numpy("float64")}, index=x.index)


def map_cik_to_permno(events: pd.DataFrame) -> pd.DataFrame:
    """Attach a date-validated PERMNO to each event via cik_links."""
    links = pd.read_parquet(MKT / "cik_links.parquet")
    links["cik10"] = links["cik"].astype(str).str.zfill(10)
    links["linkdt"] = pd.to_datetime(links["linkdt"])
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(pd.Timestamp("2100-01-01"))
    # prefer primary links
    links["prim_rank"] = (links["linkprim"] == "P").astype(int)
    ev = events.copy()
    ev["cik10"] = ev["issuer_cik"].astype(str).str.zfill(10)
    ev["signal_date"] = pd.to_datetime(ev["signal_date"])
    permnos = np.full(len(ev), np.nan)
    by_cik = {c: g for c, g in links.groupby("cik10")}
    for i, (cik, sd) in enumerate(zip(ev["cik10"].values, ev["signal_date"].values)):
        g = by_cik.get(cik)
        if g is None:
            continue
        sd = pd.Timestamp(sd)
        valid = g[(g["linkdt"] <= sd) & (g["linkenddt"] >= sd)]
        pick = valid if not valid.empty else g   # fall back to any link for the cik
        pick = pick.sort_values("prim_rank", ascending=False)
        permnos[i] = pick["lpermno"].iloc[0]
    ev["permno"] = permnos
    return ev


def _beta(stock: pd.DataFrame, xbi_ret: pd.Series, entry: pd.Timestamp) -> tuple[float, int]:
    """Estimate the stock's beta to XBI from up to 252 daily returns strictly before ``entry``.

    Returns ``(beta, n_obs)``; beta is NaN when fewer than 60 overlapping days
    exist or XBI variance is zero, so no post-entry data enters the estimate.
    """
    hist = stock.loc[stock.index < entry, "ret"].dropna().iloc[-BETA_WINDOW:]
    pair = pd.concat([hist.rename("s"), xbi_ret.rename("m")], axis=1, join="inner").dropna()
    if len(pair) < BETA_MIN_OBS:
        return np.nan, len(pair)
    v = pair["m"].var()
    if not np.isfinite(v) or v == 0:
        return np.nan, len(pair)
    return float(pair["s"].cov(pair["m"]) / v), len(pair)


def _fwd(ret: np.ndarray, pos: int, h: int) -> float:
    """Compounded forward total return over h sessions from bar pos+1..pos+h."""
    end = min(pos + h, len(ret) - 1)
    seg = ret[pos + 1:end + 1]
    seg = seg[np.isfinite(seg)]
    # A path shorter than the horizon (e.g. the stock delisted) still counts: only a
    # window with no observed return at all is missing. The outer length check is
    # kept from an earlier version and has no effect beyond the inner one.
    if len(seg) < max(1, int(0.5 * h)):
        if len(seg) == 0:
            return np.nan
    return float(np.prod(1 + seg) - 1)


def reprice(events: pd.DataFrame, panels: dict[int, pd.DataFrame], xbi: pd.DataFrame) -> pd.DataFrame:
    """Attach CRSP forward returns to every event.

    For each event with a PERMNO and a price panel, entry is the first session
    strictly after ``signal_date``. For each horizon in ``HORIZONS`` it stores the
    stock's forward return (``fwd_ret``), XBI's over the same sessions
    (``sfwd_ret``), their difference (``abn_ret``) and the beta-adjusted residual
    (``resid_ret``), plus the entry date and the pre-entry beta.
    """
    xbi_ret = xbi["ret"].dropna()
    xbi_dates = xbi.index.values.astype("datetime64[ns]")
    xbi_arr = xbi["ret"].to_numpy("float64")
    out = events.copy().reset_index(drop=True)
    sig = pd.to_datetime(out["signal_date"]).values.astype("datetime64[ns]")
    entry = np.full(len(out), np.datetime64("NaT"), "datetime64[ns]")
    betas = np.full(len(out), np.nan)
    bn = np.zeros(len(out), int)
    cols = {f"{p}_{h}": np.full(len(out), np.nan)
            for h in HORIZONS for p in ("fwd_ret", "sfwd_ret", "abn_ret", "resid_ret")}
    for i in range(len(out)):
        pm = out.at[i, "permno"]
        if not np.isfinite(pm):
            continue
        panel = panels.get(int(pm))
        if panel is None:
            continue
        d = panel.index.values.astype("datetime64[ns]")
        r = panel["ret"].to_numpy("float64")
        # Entry = first session strictly after the filing date (no same-day trading).
        pos = int(np.searchsorted(d, sig[i], side="right"))
        if pos >= len(d):
            continue
        entry[i] = d[pos]
        spos = int(np.searchsorted(xbi_dates, sig[i], side="right"))
        b, n = _beta(panel, xbi_ret, pd.Timestamp(d[pos]))
        betas[i], bn[i] = b, n
        for h, days in HORIZONS.items():
            f = _fwd(r, pos, days)
            s = _fwd(xbi_arr, spos, days) if spos < len(xbi_arr) else np.nan
            cols[f"fwd_ret_{h}"][i] = f
            cols[f"sfwd_ret_{h}"][i] = s
            cols[f"abn_ret_{h}"][i] = f - s if np.isfinite(f) and np.isfinite(s) else np.nan
            if np.isfinite(b) and np.isfinite(f) and np.isfinite(s):
                cols[f"resid_ret_{h}"][i] = f - b * s
    out["entry_date"] = entry
    out["sector_beta"] = betas
    out["sector_beta_nobs"] = bn
    for k, v in cols.items():
        out[k] = v
    return out


def main() -> None:
    """Re-price the earlier 87-ticker event sample on CRSP (the survivorship head-to-head).

    Reads ``out_scaled/insider_events_featured.parquet`` next to this file and
    writes ``insider_events_crsp.parquet`` beside it. Not used by the conviction
    scripts, which import only the loaders above.
    """
    here = Path(__file__).resolve().parent
    ev = pd.read_parquet(here / "out_scaled" / "insider_events_featured.parquet")
    print(f"loading CRSP panels from {MKT} ...")
    panels = load_crsp_panels()
    xbi = load_xbi()
    ev = map_cik_to_permno(ev)
    print(f"events {len(ev)} | mapped to permno: {ev.permno.notna().sum()} "
          f"({ev.permno.notna().mean():.1%}) | CRSP permnos {len(panels)} | XBI days {len(xbi)}")
    out = reprice(ev, panels, xbi)
    priced = out["fwd_ret_6m"].notna().sum()
    print(f"CRSP-priced (6m fwd): {priced} ({priced/len(out):.1%})   "
          f"[yfinance priced ~{ev['issuer_cik'].notna().sum()} attempted]")
    dest = here / "out_scaled" / "insider_events_crsp.parquet"
    out.to_parquet(dest, index=False)
    print("wrote", dest)


if __name__ == "__main__":
    main()
