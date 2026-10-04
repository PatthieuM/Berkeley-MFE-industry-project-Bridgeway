"""Fetch SEC ownership filings and convert them to canonical data frames.

Per-form engines discover filings, invoke the format-aware parser, append
provenance and validation fields, and return analysis-ready rows.
"""

from __future__ import annotations

import logging
import hashlib
import json
from typing import Optional

import pandas as pd

if __package__:
    from .client import EdgarClient
    from .discovery import resolve_cik, list_insider_filings, FilingRef
    from .formats import parse_filing, assign_row_ids, submission_acceptance, CANON_FIELDS, STATUS_ERROR, STATUS_REVIEW
else:
    from client import EdgarClient
    from discovery import resolve_cik, list_insider_filings, FilingRef
    from formats import parse_filing, assign_row_ids, submission_acceptance, CANON_FIELDS, STATUS_ERROR, STATUS_REVIEW

logger = logging.getLogger(__name__)

_NUMERIC_COLS = [
    "shares", "price_per_share", "shares_owned_following", "sequence_in_filing",
    "conversion_or_exercise_price", "underlying_security_shares", "owner_count",
    "sec144_units_to_be_sold", "sec144_aggregate_market_value",
    "sec144_units_outstanding",
    "owner_sequence", "derived_transaction_value", "reported_transaction_value",
]
_DATE_COLS = [
    "transaction_date", "period_of_report", "edgar_filing_date",
    "exercise_date", "expiration_date", "sec144_approx_sale_date",
]
# Provenance columns appended per row (not part of CANON_FIELDS).
_PROVENANCE = ["form_engine", "is_amendment", "form_flag", "accession",
               "edgar_filing_date", "filing_url", "xml_url", "source_url",
               "primary_document_url", "submission_url"]


class BaseOwnershipEngine:
    """Shared discovery, parsing, provenance, and frame conversion workflow."""
    FORM: str = ""
    NAME: str = ""
    DESCRIPTION: str = ""

    def __init__(self, client: Optional[EdgarClient] = None, cache_dir: Optional[str] = None):
        """Use the given EDGAR client, or create one caching to ``cache_dir``."""
        self.client = client or EdgarClient(cache_dir=cache_dir)

    # -- per-filing fetch/parse -------------------------------------------
    def fetch_filing(self, ref: FilingRef) -> list[dict]:
        """Fetch and parse one filing reference into canonical row dictionaries."""
        # Complete submission preserves the acceptance header and evidence from
        # footnotes/exhibits. Native ownership XML inside it has parser priority.
        rows, source_format, parse_status = [], "unknown", STATUS_ERROR
        source_url = ref.submission_url
        last_error = None
        header_acceptance = ""
        content = b""
        selected_content = b""
        status_rank = {STATUS_ERROR: 0, STATUS_REVIEW: 1}
        for url in dict.fromkeys((ref.submission_url, ref.document_url)):
            try:
                content = self.client.get_bytes(url)
                if url == ref.submission_url:
                    header_acceptance = submission_acceptance(content)
                candidate, fmt, status = parse_filing(content, url)
                if (candidate or status != STATUS_ERROR) and (
                    not rows or status_rank.get(status, 2) >= status_rank.get(parse_status, 2)
                ):
                    rows, source_format, parse_status, source_url = candidate, fmt, status, url
                    selected_content = content
                if status not in (STATUS_ERROR, STATUS_REVIEW):
                    break
            except PermissionError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("Filing source failed %s: %s", url, exc)
        if not rows and last_error is not None and parse_status == STATUS_ERROR:
            raise RuntimeError(f"Unable to fetch filing {ref.accession}") from last_error

        if not rows:
            rows = [{**dict.fromkeys(CANON_FIELDS, ""), "row_kind": "no_rows"}]

        for r in rows:
            r["source_format"] = source_format
            r["parse_status"] = parse_status
            if not r.get("issuer_cik") and ref.discovery_source != "master_index":
                r["issuer_cik"] = f"{ref.cik:010d}"
            r["form"] = ref.form
            r["form_engine"] = self.NAME
            r["is_amendment"] = ref.is_amendment
            r["accession"] = ref.accession
            r["edgar_filing_date"] = ref.filing_date
            if ref.acceptance_datetime:
                r["acceptance_datetime"] = ref.acceptance_datetime
                r["acceptance_source"] = "submissions_api"
            elif header_acceptance:
                r["acceptance_datetime"] = header_acceptance
                r["acceptance_source"] = "submission_header"
            if not r.get("source_sha256"):
                r["source_sha256"] = hashlib.sha256(selected_content).hexdigest()
            r["filing_url"] = ref.index_url
            r["xml_url"] = source_url
            r["source_url"] = source_url
            r["primary_document_url"] = ref.document_url
            r["submission_url"] = ref.submission_url
            r.setdefault("form_flag", "")
        self._flag_filing(rows)
        return assign_row_ids(rows, ref.accession)

    def _flag_filing(self, rows: list[dict]) -> None:
        """Apply form-specific checks after all filing rows are available."""
        for row in rows:
            self._flag(row)

    def _flag(self, row: dict) -> None:
        """Form-specific validation hook (subclasses override)."""

    # -- scraping ----------------------------------------------------------
    def scrape_rows(self, ticker_or_cik, limit: Optional[int] = None) -> list[dict]:
        """Discover and parse this engine's filings for a ticker or CIK."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative or None")
        cik = (int(ticker_or_cik) if isinstance(ticker_or_cik, int) or str(ticker_or_cik).isdigit()
               else resolve_cik(self.client, ticker_or_cik))
        refs = list_insider_filings(self.client, cik, forms=(self.FORM,))
        if limit is not None:
            refs = refs[:limit]
        logger.info("[Form %s] %d filings for CIK %d", self.FORM, len(refs), cik)
        rows: list[dict] = []
        for i, ref in enumerate(refs, 1):
            try:
                rows.extend(self.fetch_filing(ref))
            except PermissionError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("[Form %s] failed on %s: %s", self.FORM, ref.accession, exc)
                failure = dict.fromkeys(CANON_FIELDS, "")
                failure.update({
                    "form": ref.form, "document_type": ref.form, "form_engine": self.NAME,
                    "is_amendment": ref.is_amendment, "form_flag": "",
                    "accession": ref.accession, "edgar_filing_date": ref.filing_date,
                    "period_of_report": ref.report_date,
                    "issuer_cik": f"{ref.cik:010d}" if ref.discovery_source != "master_index" else "",
                    "acceptance_datetime": ref.acceptance_datetime,
                    "acceptance_source": "submissions_api" if ref.acceptance_datetime else "missing",
                    "filing_url": ref.index_url, "xml_url": "", "source_url": "",
                    "primary_document_url": ref.document_url, "submission_url": ref.submission_url,
                    "source_format": "unknown", "parse_warnings": json.dumps([f"{type(exc).__name__}: {exc}"]),
                    "parse_status": STATUS_ERROR, "row_kind": "fetch_error",
                })
                rows.extend(assign_row_ids([failure], ref.accession))
            if i % 25 == 0:
                logger.info("[Form %s] processed %d/%d", self.FORM, i, len(refs))
        return rows

    # -- DataFrame ---------------------------------------------------------
    def to_dataframe(self, rows: list[dict]) -> pd.DataFrame:
        """Convert canonical row dictionaries to a typed, ordered data frame."""
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in _DATE_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        order = [c for c in (CANON_FIELDS + _PROVENANCE) if c in df.columns]
        order += [c for c in df.columns if c not in order]
        df = df[order]
        if "transaction_date" in df.columns:
            df = df.sort_values([c for c in ("transaction_date", "edgar_filing_date") if c in df],
                                ascending=False, na_position="last").reset_index(drop=True)
        return df

    def scrape(self, ticker_or_cik, limit: Optional[int] = None) -> pd.DataFrame:
        """Discover and parse filings into an analysis-ready data frame."""
        return self.to_dataframe(self.scrape_rows(ticker_or_cik, limit))


class Form4Engine(BaseOwnershipEngine):
    """Parse changes in beneficial ownership reported on Form 4."""
    FORM, NAME = "4", "form4"
    DESCRIPTION = "Changes in beneficial ownership (trades)"

    def _flag_filing(self, rows):
        """Flag a Form 4 that reports only holdings and no transaction."""
        if any(row.get("parse_status") in (STATUS_ERROR, STATUS_REVIEW) for row in rows):
            return
        if not any(row.get("row_kind") == "transaction" for row in rows):
            for row in rows:
                if row.get("row_kind") in ("holding", "no_rows"):
                    row["form_flag"] = row.get("form_flag") or "no_transaction_on_form4"


class Form3Engine(BaseOwnershipEngine):
    """Parse initial beneficial-ownership holdings reported on Form 3."""
    FORM, NAME = "3", "form3"
    DESCRIPTION = "Initial statement of beneficial ownership (holdings)"

    def _flag(self, row):
        """Flag a Form 3 that reports a transaction (Form 3 should list holdings only)."""
        if row.get("row_kind") == "transaction" or (row.get("transaction_code") or "").strip():
            row["form_flag"] = row.get("form_flag") or "transaction_on_form3"


class Form5Engine(BaseOwnershipEngine):
    """Parse annual or deferred beneficial-ownership reports on Form 5."""
    FORM, NAME = "5", "form5"
    DESCRIPTION = "Annual statement (exempt/late trades)"


class Form144Engine(BaseOwnershipEngine):
    """Parse proposed restricted or control-security sales on Form 144."""
    FORM, NAME = "144", "form144"
    DESCRIPTION = "Notice of proposed sale of restricted/control securities"


ENGINES: dict[str, type[BaseOwnershipEngine]] = {
    "3": Form3Engine, "4": Form4Engine, "5": Form5Engine, "144": Form144Engine,
}


def get_engine(form: str, **kwargs) -> BaseOwnershipEngine:
    """Construct the ownership engine registered for ``form``."""
    try:
        engine_class = ENGINES[form]
    except KeyError:
        raise ValueError(f"No engine for form {form!r}; choose {sorted(ENGINES)}") from None
    return engine_class(**kwargs)
