"""Fetch SEC insider filings (Forms 3, 4, 5 and 144) for a period and store them in SQLite.

Usage, reruns and exit codes are documented in README.md.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import logging

import pandas as pd

from shared.config import DATA_ROOT
from shared.sec.form144 import default_ciks
from shared.sec.form4_parser.client import EdgarClient
from shared.sec.form4_parser.discovery import INSIDER_FORMS, list_index_filings, list_insider_filings, resolve_cik
from shared.sec.form4_parser.engines import get_engine
from shared.sec.form4_parser.formats import STATUS_OK, STATUS_OK_LEGACY_TABLE, STATUS_OK_LEGACY_XML
from shared.sec.ingest.storage import FilingStore

logger = logging.getLogger(__name__)
DATABASE = DATA_ROOT / "sec/insider_filings.sqlite"
OK_STATUSES = (STATUS_OK, STATUS_OK_LEGACY_XML, STATUS_OK_LEGACY_TABLE)


def resolve_period(period=None, start=None, end=None) -> tuple[str, str]:
    """Return inclusive YYYY-MM-DD dates. A day, week or month ends on ``end``, which defaults to today."""
    if (period is None) == (start is None):
        raise ValueError("Give either a period (day, week, month) or a start date")
    last = pd.Timestamp(date.fromisoformat(end) if end else date.today())
    if period is None:
        first = pd.Timestamp(date.fromisoformat(start))
    elif period == "day":
        first = last
    elif period == "week":
        first = last - pd.Timedelta(days=6)
    elif period == "month":
        first = last - pd.DateOffset(months=1) + pd.Timedelta(days=1)
    else:
        raise ValueError("period must be day, week or month")
    if first > last:
        raise ValueError("start is after end")
    return first.date().isoformat(), last.date().isoformat()


def quarters(start: str, end: str) -> list[tuple[int, int]]:
    """Return the (year, quarter) pairs that overlap the period."""
    return [(quarter.year, quarter.quarter) for quarter in pd.period_range(start, end, freq="Q")]


def update_filings(store: FilingStore, client, *, start: str, end: str, ciks=None) -> dict:
    """Store new filings in the period and retry fetch or storage failures.

    ``ciks`` is the set of stocks. ``None`` means every filer in the EDGAR
    quarterly index. An SEC access rejection (PermissionError) stops the run.
    """
    targets = quarters(start, end) if ciks is None else sorted({int(cik) for cik in ciks})
    found, discovery_errors = {}, []
    for target in targets:
        try:
            if ciks is None:
                refs = list_index_filings(client, year=target[0], quarter=target[1], forms=INSIDER_FORMS)
            else:
                refs = list_insider_filings(client, target, forms=INSIDER_FORMS)
        except PermissionError:
            raise
        except Exception as error:  # noqa: BLE001 - record this target and carry on with the others
            logger.warning("Discovery failed for %s: %s", target, error)
            discovery_errors.append({"target": str(target), "error": str(error)})
            continue
        for ref in refs:
            if start <= ref.filing_date <= end:
                found.setdefault(ref.accession, ref)

    fetched = flagged = failed = 0
    for ref in found.values():
        if not store.needs_fetch(ref.accession):
            continue
        try:
            rows = get_engine(ref.base_form, client=client).fetch_filing(ref)
            store.save(ref, rows)
        except PermissionError:
            raise
        except Exception as error:  # noqa: BLE001 - keep the failure visible and carry on
            logger.warning("Filing %s failed: %s", ref.accession, error)
            # Preserve the cause of wrapped errors.
            store.save_failure(ref, error.__cause__ or error)
            failed += 1
        else:
            fetched += 1
            flagged += any(row.get("parse_status") not in OK_STATUSES for row in rows)
    return {"start": start, "end": end, "discovered": len(found), "fetched": fetched,
            "flagged": flagged, "failed": failed, "discovery_errors": discovery_errors}


def main(argv=None) -> int:
    """Run a crawl; exit 1 for fetch, storage or discovery failures, not parser flags."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--ciks", nargs="+", type=int, help="issuer CIKs")
    target.add_argument("--tickers", nargs="+", help="current tickers")
    target.add_argument("--universe", action="store_true", help="every issuer of the biotech universe with a CIK")
    target.add_argument("--all", action="store_true", help="every filer on EDGAR")
    parser.add_argument("--period", choices=("day", "week", "month"), help="period ending on --end")
    parser.add_argument("--start", help="first filing date, YYYY-MM-DD")
    parser.add_argument("--end", help="last filing date, YYYY-MM-DD; default today")
    args = parser.parse_args(argv)
    try:
        start, end = resolve_period(args.period, args.start, args.end)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = EdgarClient()
    ciks = args.ciks  # None with --all
    if args.tickers:
        ciks = [resolve_cik(client, ticker) for ticker in args.tickers]
    if args.universe:
        ciks = default_ciks()
    with FilingStore(DATABASE) as store:
        summary = update_filings(store, client, start=start, end=end, ciks=ciks)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] or summary["discovery_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
