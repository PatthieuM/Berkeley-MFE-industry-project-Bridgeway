"""
scrape_gics_events.py -- crawl EDGAR for the GICS-biotech firms missing from
the handoff event table, using the shared Form 4 parser (shared/sec/form4_parser).

For every PERMNO in the strict-GICS biotech universe that has NO events in
event_results.parquet (~427 firms / 443 CIKs), pull Form 4 (+5) via
the shared Form 4 parser's engines, keep the open-market P/S non-derivative transactions
(the actual trading decisions), and save one row per event with the fields the
conviction signal needs. Restartable: each CIK's result is cached to
per_cik/<cik>.parquet, so re-running skips finished CIKs.

Downstream (separate step): map to CRSP prices in data/crsp_prices, compute
features + forward returns, and re-run the walk-forward on the full universe.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402  (also puts the submission root on sys.path)
from shared.sec.form4_parser.engines import Form4Engine, Form5Engine  # noqa: E402
from shared.sec.form4_parser.client import EdgarClient  # noqa: E402

UNIV = paths.UNIVERSE
EVENTS = paths.EVENTS
OUT = paths.OUTPUT
CACHE = OUT / "per_cik"
CACHE.mkdir(parents=True, exist_ok=True)

OPEN_MARKET = ("P", "S")
KEEP = ["issuer_cik", "issuer_ticker", "owner_cik", "owner_name", "is_director",
        "is_officer", "is_ten_percent_owner", "officer_title", "security_title",
        "transaction_date", "transaction_code", "shares", "price_per_share",
        "shares_owned_following", "rule_10b5_1", "acceptance_datetime",
        "edgar_filing_date", "accession"]


def missing_cik_permno() -> pd.DataFrame:
    """List the universe firms (CIK, PERMNO) that have no events in the handoff table.

    These are the firms the crawl fills in.
    """
    ev = pd.read_parquet(EVENTS, columns=["permno"])
    have = set(ev.permno.astype(int).unique())
    u = pd.read_parquet(UNIV, columns=["permno", "cik"])
    u["permno"] = pd.to_numeric(u["permno"], errors="coerce")
    u = u.dropna(subset=["permno", "cik"]).copy()
    u["permno"] = u["permno"].astype(int)
    u["cik"] = u["cik"].astype(str).str.zfill(10)
    miss = u[~u.permno.isin(have)].drop_duplicates("cik")
    return miss[["cik", "permno"]].reset_index(drop=True)


def open_market_events(raw: pd.DataFrame, cik: str, permno: int) -> pd.DataFrame:
    """Keep the open-market trades from one firm's parsed filings.

    Filters to non-derivative, non-amendment transaction rows with code P (buy)
    or S (sell), attaches the PERMNO and a ``side`` of +1/-1, and dates each trade
    by its EDGAR acceptance date (falling back to the filing date, then the
    transaction date).
    """
    if raw.empty:
        return pd.DataFrame()
    m = ((raw.get("table") == "non_derivative")
         & (raw.get("transaction_code").isin(OPEN_MARKET))
         & (raw.get("row_kind") == "transaction")
         # Known issue: when is_amendment is an object column holding NaN, ~ raises
         # "bad operand type for unary ~: 'float'"; main() catches it and caches that
         # CIK with no events, so the firm silently drops out of the crawl.
         & (~raw.get("is_amendment", False)))
    t = raw[m][[c for c in KEEP if c in raw.columns]].copy()
    if t.empty:
        return t
    t["permno"] = permno
    t["shares"] = pd.to_numeric(t["shares"], errors="coerce")
    t["price_per_share"] = pd.to_numeric(t["price_per_share"], errors="coerce")
    t["side"] = np.where(t["transaction_code"] == "P", 1, -1)
    acc = pd.to_datetime(t["acceptance_datetime"], errors="coerce", utc=True)
    filed = pd.to_datetime(t["edgar_filing_date"], errors="coerce", utc=True)
    sig = acc.fillna(filed).dt.tz_localize(None).dt.normalize()
    t["signal_date"] = sig.fillna(pd.to_datetime(t["transaction_date"], errors="coerce"))
    return t[t["signal_date"].notna()]


def main() -> None:
    """Crawl Forms 4 and 5 for every missing firm, caching one file per CIK.

    Re-running skips CIKs that already have a cache file. With ``--shard i/n``
    several workers split the CIK list; ``--consolidate`` merges the per-CIK
    files into ``gics_missing_events.parquet``.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-ciks", type=int, help="Only process the first N CIKs (smoke test).")
    ap.add_argument("--limit-filings", type=int, help="Cap filings per CIK (smoke test).")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: this worker takes CIKs where index %% n == i. Run n workers in parallel.")
    ap.add_argument("--min-interval", type=float, default=0.15,
                    help="Seconds between requests PER worker. With n workers keep n*(1/interval) <= 10/s "
                         "(e.g. 4 workers -> 0.45).")
    ap.add_argument("--consolidate", action="store_true",
                    help="Only consolidate per_cik/*.parquet into the combined file, then exit.")
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    if args.consolidate:
        parts = [pd.read_parquet(f) for f in CACHE.glob("*.parquet")]
        parts = [p for p in parts if len(p)]
        allev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        allev.to_parquet(OUT / "gics_missing_events.parquet", index=False)
        print(f"consolidated {len(allev)} events across "
              f"{allev.permno.nunique() if len(allev) else 0} firms")
        return

    miss = missing_cik_permno()
    if args.limit_ciks:
        miss = miss.head(args.limit_ciks)
    if n > 1:
        miss = miss.iloc[i::n].reset_index(drop=True)   # this worker's shard
    print(f"shard {i}/{n}: {len(miss)} CIKs to crawl (Form 4 + 5), min_interval={args.min_interval}s")
    client = EdgarClient(min_interval=args.min_interval)
    f4 = Form4Engine(client=client, cache_dir=str(OUT / ".cache"))
    f5 = Form5Engine(client=client, cache_dir=str(OUT / ".cache"))

    done = 0
    for _, row in miss.iterrows():
        cik, permno = row["cik"], int(row["permno"])
        cache_f = CACHE / f"{cik}.parquet"
        if cache_f.exists():
            done += 1
            continue
        frames = []
        for eng in (f4, f5):
            try:
                raw = eng.scrape(int(cik), limit=args.limit_filings)
                frames.append(open_market_events(raw, cik, permno))
            except Exception as exc:  # noqa: BLE001 - skip a bad CIK, keep going
                print(f"  CIK {cik} form{eng.FORM} failed: {exc}")
        ev = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
            not f.empty for f in frames) else pd.DataFrame(columns=KEEP + ["permno", "side", "signal_date"])
        ev.to_parquet(cache_f, index=False)   # empty frame => marks CIK done, no events
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(miss)} CIKs done")

    if n > 1:
        print(f"\nshard {i}/{n} DONE ({done} CIKs). Consolidate all shards with: "
              f"python scrape_gics_events.py --consolidate")
        return
    # single-worker run: consolidate
    parts = [pd.read_parquet(f) for f in CACHE.glob("*.parquet")]
    parts = [p for p in parts if len(p)]
    allev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    allev.to_parquet(OUT / "gics_missing_events.parquet", index=False)
    print(f"\nDONE: {len(allev)} open-market events across "
          f"{allev.permno.nunique() if len(allev) else 0} firms -> {OUT}/gics_missing_events.parquet")


if __name__ == "__main__":
    main()
