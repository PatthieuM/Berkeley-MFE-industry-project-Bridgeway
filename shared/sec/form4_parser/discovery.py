"""Resolve tickers to CIKs and discover SEC Forms 3, 4, 5, and 144.

Inputs are SEC submissions JSON or master indexes; outputs are filing references
with archive URLs and timing metadata used by the parsing engines.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from dataclasses import dataclass
from typing import Iterable, Optional

if __package__:
    from .client import EdgarClient
else:
    from client import EdgarClient

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"
FILING_INDEX_URL = ARCHIVES_BASE + "/{acc_dash}-index.htm"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Insider forms we support. 3/4/5 share the ownership-XML schema; 144 is separate.
INSIDER_FORMS = ("3", "4", "5", "144")


def resolve_cik(client: EdgarClient, ticker: str) -> int:
    """Resolve a ticker to its current CIK using the SEC ticker map."""
    data = client.get_json(_TICKER_MAP_URL, use_cache=False)
    ticker_up = ticker.strip().upper()
    for row in data.values():
        if row["ticker"].upper() == ticker_up:
            return int(row["cik_str"])
    raise KeyError(f"Ticker {ticker!r} not found in SEC ticker map")


@dataclass
class FilingRef:
    """Pointer to one insider filing, from the submissions index."""
    cik: int
    accession: str
    form: str                 # "3", "4", "5", "144", "4/A", ...
    filing_date: str          # YYYY-MM-DD
    report_date: str
    acceptance_datetime: str  # e.g. "2026-08-20T16:31:05.000Z"
    primary_document: str
    submission_path: str = ""
    discovery_source: str = "submissions_api"

    @property
    def acc_nodash(self) -> str:
        """Return the accession number without separators for archive paths."""
        return self.accession.replace("-", "")

    @property
    def base_form(self) -> str:
        """Return the form type without an amendment suffix."""
        return self.form.split("/")[0]

    @property
    def is_amendment(self) -> bool:
        """Report whether the filing form carries an amendment suffix."""
        return "/" in self.form

    @property
    def index_url(self) -> str:
        """Return the human-readable EDGAR filing index URL."""
        return FILING_INDEX_URL.format(
            cik=self.cik, acc_nodash=self.acc_nodash, acc_dash=self.accession
        )

    @property
    def document_url(self) -> str:
        """Machine-readable primary document; falls back to the full submission
        text when primaryDocument is empty (very old filings)."""
        base = ARCHIVES_BASE.format(cik=self.cik, acc_nodash=self.acc_nodash)
        doc = self.primary_document
        if not doc:
            return f"{base}/{self.accession}.txt"
        if "/" in doc:                       # strip XSL viewer prefix
            doc = doc.split("/")[-1]
        return f"{base}/{doc}"

    @property
    def submission_url(self) -> str:
        """Return the complete-submission archive URL, preserving index paths."""
        if self.submission_path:
            return "https://www.sec.gov/Archives/" + self.submission_path.lstrip("/")
        base = ARCHIVES_BASE.format(cik=self.cik, acc_nodash=self.acc_nodash)
        return f"{base}/{self.accession}.txt"


def list_insider_filings(
    client: EdgarClient, cik: int, forms: Iterable[str] = INSIDER_FORMS
) -> list[FilingRef]:
    """Return every insider filing (default Forms 3/4/5/144) for a CIK, across
    the full paged submission history."""
    wanted = {f.upper().split("/")[0] for f in forms}
    data = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    refs: list[FilingRef] = []

    def _collect(block: dict) -> None:
        """Append a FilingRef for every wanted form in one block of the submissions JSON."""
        forms_ = block.get("form", [])
        n = len(forms_)
        acc_dt = block.get("acceptanceDateTime", [""] * n)
        rep = block.get("reportDate", [""] * n)
        for i in range(n):
            if forms_[i].upper().split("/")[0] not in wanted:
                continue
            refs.append(FilingRef(
                cik=cik,
                accession=block["accessionNumber"][i],
                form=forms_[i],
                filing_date=block["filingDate"][i],
                report_date=rep[i] if i < len(rep) else "",
                acceptance_datetime=acc_dt[i] if i < len(acc_dt) else "",
                primary_document=block.get("primaryDocument", [""] * n)[i],
            ))

    _collect(data["filings"]["recent"])
    for extra in data["filings"].get("files", []):
        _collect(client.get_json(f"https://data.sec.gov/submissions/{extra['name']}"))

    logger.info("Found %d filings for CIK %d (forms=%s)", len(refs), cik, sorted(wanted))
    unique = {}
    for ref in refs:
        # Recent API records generally have richer metadata than old shards.
        unique.setdefault(ref.accession, ref)
    return sorted(unique.values(),
                  key=lambda ref: (ref.filing_date, ref.accession), reverse=True)


def list_filings_for_ciks(
    client: EdgarClient,
    ciks: Iterable[int | str],
    forms: Iterable[str] = INSIDER_FORMS,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[FilingRef]:
    """Enumerate explicit historical CIKs without current-ticker resolution.

    Dates are filing-date discovery bounds, inclusive; trading eligibility is
    separately based on acceptance and point-in-time issuer membership.
    """
    forms = tuple(forms)
    refs = {}
    for cik in sorted({int(value) for value in ciks}):
        if cik <= 0:
            raise ValueError("CIKs must be positive integers")
        for ref in list_insider_filings(client, cik, forms=forms):
            if start_date and ref.filing_date < start_date:
                continue
            if end_date and ref.filing_date > end_date:
                continue
            refs.setdefault(ref.accession, ref)
    return sorted(refs.values(), key=lambda ref: (ref.filing_date, ref.accession))


def parse_master_index(content: bytes | str, forms: Iterable[str] = INSIDER_FORMS) -> list[FilingRef]:
    """Parse SEC daily/quarterly master.idx files, retaining archive filenames.

    Index CIK can identify a reporting owner. The parsed filing's issuer CIK,
    never this index CIK by assumption, must drive subsequent universe matching.
    """
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    wanted = {str(form).upper().split("/")[0] for form in forms}
    refs = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        cik, _, form, filing_date, filename = parts
        if form.upper().split("/")[0] not in wanted:
            continue
        match = re.search(r"(\d{10}-\d{2}-\d{6})\.txt$", filename)
        if not match or not re.fullmatch(r"edgar/data/\d+/[\d/-]+\.txt", filename):
            logger.warning("Unrecognized master-index archive path: %s", filename)
            continue
        try:
            date.fromisoformat(filing_date)
        except ValueError:
            logger.warning("Invalid master-index filing date: %s", filing_date)
            continue
        ref = FilingRef(int(cik), match.group(1), form, filing_date, "", "", "",
                        submission_path=filename, discovery_source="master_index")
        refs.setdefault(ref.accession, ref)
    return sorted(refs.values(), key=lambda ref: (ref.filing_date, ref.accession))


def list_index_filings(
    client: EdgarClient,
    *,
    year: int,
    quarter: Optional[int] = None,
    day: Optional[str] = None,
    forms: Iterable[str] = INSIDER_FORMS,
) -> list[FilingRef]:
    """Discover all filers for a quarter or one YYYY-MM-DD day, no ticker list."""
    if day:
        parsed = date.fromisoformat(day)
        if parsed.year != year:
            raise ValueError("day must belong to year")
        computed_quarter = (parsed.month - 1) // 3 + 1
        if quarter is not None and quarter != computed_quarter:
            raise ValueError("day must belong to quarter")
        url = (f"https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{computed_quarter}/"
               f"master.{parsed:%Y%m%d}.idx")
    else:
        if quarter not in (1, 2, 3, 4):
            raise ValueError("quarter must be 1, 2, 3 or 4")
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    # Refresh changing indices online; an offline client replays its stored snapshot.
    return parse_master_index(client.get_bytes(url, use_cache=False), forms=forms)
