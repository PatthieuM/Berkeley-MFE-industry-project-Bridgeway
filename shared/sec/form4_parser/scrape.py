"""Run the SEC Forms 3, 4, 5, and 144 parsing pipeline from the command line.

The CLI accepts a ticker or CIK and writes per-form CSV tables, optional quality
reports, and a combined ownership table when all forms are requested.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
if __package__:
    from .engines import get_engine, ENGINES
    from .client import EdgarClient
    from . import quality_checks as QC
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engines import get_engine, ENGINES
    from client import EdgarClient
    import quality_checks as QC


def _summary(form: str, df: pd.DataFrame) -> None:
    """Print a short report for one form: rows, filings, insiders, date range, formats and flags."""
    print("\n" + "=" * 68)
    print(f"FORM {form} -- {len(df):,} rows")
    if df.empty:
        print("(no rows)")
        print("=" * 68)
        return
    def known_count(column: str) -> int:
        values = df.get(column, pd.Series(dtype="string"))
        return values.replace("", pd.NA).nunique()

    print(f"filings={known_count('accession'):,}  insiders={known_count('owner_cik'):,}")
    if "transaction_date" in df and df["transaction_date"].notna().any():
        print(f"transaction dates: {df['transaction_date'].min().date()} -> {df['transaction_date'].max().date()}")
    print("format x status:")
    labels = df.reindex(columns=["source_format", "parse_status"]).fillna("unknown").replace("", "unknown")
    print(labels.groupby(["source_format", "parse_status"]).size().to_string())
    if "form_flag" in df:
        flagged = df[df["form_flag"].fillna("").astype(bool)]
        if len(flagged):
            print(f"flags: {dict(flagged['form_flag'].value_counts())}")
    print("=" * 68)


def main() -> None:
    """Parse selected forms for a ticker or CIK and write CSV outputs."""
    ap = argparse.ArgumentParser(description="SEC insider filing parser (Forms 3/4/5/144).")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--ticker", default="AAPL")
    target.add_argument("--cik", type=int, help="Historical issuer CIK; bypass current ticker map")
    ap.add_argument("--form", default="4", choices=list(ENGINES) + ["all"])
    ap.add_argument("--limit", type=int, default=50, help="Max filings per form (0 = all)")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--cache-dir", default=".cache/sec-form4")
    ap.add_argument("--checks", action="store_true",
                    help="Run Bridgeway sanity checks (ownership continuity + price) and write reports")
    ap.add_argument("--market-data", type=Path, help="Optional market OHLC CSV for price QA")
    ap.add_argument("--split-adjustments", type=Path, help="Optional effective-date split-factor CSV")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit must be nonnegative (0 = all)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    limit = None if args.limit == 0 else args.limit
    forms = list(ENGINES) if args.form == "all" else [args.form]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = EdgarClient(cache_dir=args.cache_dir)
    issuer = args.cik if args.cik is not None else args.ticker
    stem = str(issuer).upper()
    frames = []
    for form in forms:
        df = get_engine(form, client=client).scrape(issuer, limit)
        _summary(form, df)
        if not df.empty:
            path = outdir / f"{stem}_form{form}.csv"
            df.to_csv(path, index=False)
            print(f"wrote {path} ({len(df):,} rows)")
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(frames) > 1:
        cpath = outdir / f"{stem}_insider_all.csv"
        combined.to_csv(cpath, index=False)
        print(f"\nwrote combined {cpath} ({len(combined):,} rows across forms {forms})")

    # -- Bridgeway sanity checks (continuity spans Forms 3/4/5 together) --
    if args.checks and not combined.empty:
        splits = pd.read_csv(args.split_adjustments) if args.split_adjustments else None
        cont = QC.check_ownership_continuity(combined, splits)
        if not cont.empty:
            cont.to_csv(outdir / f"{stem}_continuity.csv", index=False)
            print(f"\n[continuity] {len(cont)} tested rows -> "
                  f"{dict(cont['continuity_status'].value_counts())}")
            print(f"  wrote {outdir / f'{stem}_continuity.csv'}")
        # Without a supplied market table, comparable rows report NO_MARKET_DATA.
        market = pd.read_csv(args.market_data) if args.market_data else None
        price = QC.check_transaction_prices(combined, market)
        if not price.empty:
            price.to_csv(outdir / f"{stem}_price.csv", index=False)
            print(f"[price]     {len(price)} rows -> {dict(price['price_status'].value_counts())}")
            print(f"  wrote {outdir / f'{stem}_price.csv'}")
            if market is None:
                print("  Supply --market-data for comparison with market prices.")


if __name__ == "__main__":
    main()
