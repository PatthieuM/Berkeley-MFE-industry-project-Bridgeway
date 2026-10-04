"""Build the full-GICS inputs of the conviction scripts from the shared data layer.

``walkforward_full_gics.py`` needs two inputs that are not produced on this branch:
the events of the universe firms absent from the handoff table, and CRSP panels
for those firms with a ``date`` column. This script builds both offline, without
the EDGAR crawl (``scrape_gics_events.py``) or a separate WRDS pull:

1. ``<CONVICTION_OUTPUT_DIR>/crsp_extra/daily_<year>.parquet``: the shared CRSP
   pull (``shared/market/pull_universe_prices.py``) with ``dlycaldt`` renamed to
   ``date``. Pass the folder to the walk-forward through ``CRSP_EXTRA_DIR``.
2. ``<CONVICTION_OUTPUT_DIR>/gics_missing_events.parquet``: the table the crawl
   would write, built from the normalized SEC quarterly data sets
   (``shared/sec/acquire_normalize.py``), restricted to dates on which the firm
   was in the biotech universe.

Differences from the crawl: original Form 4 filings only (the normalized archive
holds no Form 5), and ``signal_date`` is the SEC filing date because the data
sets carry no acceptance time. The handoff table ``event_results.parquet`` is
still required; see README.md in this folder.

Run:
    python signals/conviction_composite/prepare_inputs.py
    CRSP_EXTRA_DIR=<printed folder> python signals/conviction_composite/walkforward_full_gics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import paths
from shared.config import DEFAULT_CONFIG

CRSP_EXTRA = paths.OUTPUT / "crsp_extra"
CRAWLED = paths.OUTPUT / "gics_missing_events.parquet"


def convert_prices() -> int:
    """Copy the shared CRSP panels with the date column the conviction loader expects."""
    CRSP_EXTRA.mkdir(parents=True, exist_ok=True)
    files = sorted(DEFAULT_CONFIG.universe_price_dir.glob("daily_*.parquet"))
    for path in files:
        pd.read_parquet(path).rename(columns={"dlycaldt": "date"}).to_parquet(CRSP_EXTRA / path.name, index=False)
    return len(files)


def missing_cik_permno() -> pd.DataFrame:
    """List the universe firms (CIK, PERMNO) with no events in the handoff table.

    Same selection as ``scrape_gics_events.missing_cik_permno``.
    """
    have = set(pd.read_parquet(paths.EVENTS, columns=["permno"]).permno.astype(int).unique())
    u = pd.read_parquet(paths.UNIVERSE, columns=["permno", "cik"])
    u["permno"] = pd.to_numeric(u["permno"], errors="coerce")
    u["cik"] = pd.to_numeric(u["cik"], errors="coerce")
    u = u.dropna(subset=["permno", "cik"]).astype({"permno": int, "cik": int})
    return u[~u.permno.isin(have)].drop_duplicates("cik")[["cik", "permno"]].reset_index(drop=True)


def in_universe(events: pd.DataFrame) -> pd.Series:
    """True where the event's PERMNO was in the biotech universe at ``signal_date``."""
    u = pd.read_parquet(paths.UNIVERSE, columns=["permno", "effective_start", "effective_end", "is_biotech"])
    u = u[u["is_biotech"] == True].copy()  # noqa: E712
    u["permno"] = pd.to_numeric(u["permno"], errors="coerce")
    u["effective_start"] = pd.to_datetime(u["effective_start"])
    u["effective_end"] = pd.to_datetime(u["effective_end"]).fillna(pd.Timestamp("2100-01-01"))
    x = events[["permno", "signal_date"]].reset_index().merge(u.dropna(subset=["permno"]), on="permno", how="left")
    hit = x["signal_date"].between(x["effective_start"], x["effective_end"])
    return hit.groupby(x["index"]).any().reindex(events.index, fill_value=False)


def build_missing_events() -> pd.DataFrame:
    """Build the crawl-equivalent event table from the normalized SEC archive."""
    targets = missing_cik_permno()
    cik_to_permno = dict(zip(targets.cik, targets.permno))
    parts = []
    for sub_path in sorted(DEFAULT_CONFIG.normalized_sec_dir.glob("year=*/quarter=*/submissions.parquet")):
        sub = pd.read_parquet(sub_path, columns=["accession", "filing_date", "form", "issuer_cik", "ticker",
                                                 "is_amendment", "mentions_10b5_1"])
        sub["issuer_cik"] = pd.to_numeric(sub["issuer_cik"], errors="coerce")
        sub = sub[sub.issuer_cik.isin(cik_to_permno) & sub.form.astype(str).eq("4")
                  & ~sub.is_amendment.fillna(False).astype(bool)]
        if sub.empty:
            continue
        tx = pd.read_parquet(sub_path.parent / "transactions.parquet",
                             columns=["accession", "security_title", "transaction_date", "transaction_code",
                                      "shares", "price", "shares_after"])
        tx = tx[tx.accession.isin(sub.accession) & tx.transaction_code.isin(["P", "S"])]
        owners = pd.read_parquet(sub_path.parent / "owners.parquet",
                                 columns=["accession", "owner_cik", "owner_name", "is_director", "is_officer",
                                          "is_ten_percent_owner", "officer_title"])
        parts.append(tx.merge(sub, on="accession").merge(owners, on="accession", how="left"))
    d = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({
        "issuer_cik": d.issuer_cik.astype(int).astype(str).str.zfill(10),
        "issuer_ticker": d.ticker,
        "owner_cik": d.owner_cik,
        "owner_name": d.owner_name,
        "is_director": d.is_director,
        "is_officer": d.is_officer,
        "is_ten_percent_owner": d.is_ten_percent_owner,
        "officer_title": d.officer_title,
        "security_title": d.security_title,
        "transaction_date": d.transaction_date,
        "transaction_code": d.transaction_code,
        "shares": pd.to_numeric(d.shares, errors="coerce"),
        "price_per_share": pd.to_numeric(d.price, errors="coerce"),
        "shares_owned_following": pd.to_numeric(d.shares_after, errors="coerce"),
        "rule_10b5_1": d.mentions_10b5_1,
        "acceptance_datetime": pd.NaT,
        "edgar_filing_date": d.filing_date,
        "accession": d.accession,
        "permno": d.issuer_cik.astype(int).map(cik_to_permno).astype(int),
    })
    out["side"] = np.where(out.transaction_code.eq("P"), 1, -1)
    out["signal_date"] = pd.to_datetime(out.edgar_filing_date, errors="coerce").dt.normalize()
    out = out[out.signal_date.notna()].reset_index(drop=True)
    keep = in_universe(out)
    print(f"missing firms targeted: {len(targets)} CIKs; events built: {len(out)}; "
          f"outside the firm's universe interval and dropped: {int((~keep).sum())}")
    out = out[keep].reset_index(drop=True)
    CRAWLED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CRAWLED, index=False)
    print(f"wrote {CRAWLED}: {len(out)} events ({int(out.side.eq(1).sum())} buys, "
          f"{int(out.side.eq(-1).sum())} sells) across {out.permno.nunique()} firms")
    return out


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not paths.EVENTS.exists():
        raise SystemExit(f"handoff event table not found: {paths.EVENTS}\nsee README.md in this folder")
    n = convert_prices()
    print(f"wrote {n} CRSP panels with a date column; set CRSP_EXTRA_DIR={CRSP_EXTRA}")
    build_missing_events()
