"""Convert SEC ownership filings from multiple eras into one canonical schema.

``parse_filing`` accepts raw XML, HTML, text, SGML, or PDF bytes and returns
canonical rows plus source-format and parse-status labels for downstream engines.
"""

from __future__ import annotations

import logging
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

try:
    from lxml import etree
except ModuleNotFoundError:  # stdlib fallback keeps the offline parser usable
    etree = None
    import xml.etree.ElementTree as _stdlib_etree

if __package__:
    from . import legacy_form4 as legacy
else:  # Retain direct-script compatibility.
    import legacy_form4 as legacy

logger = logging.getLogger(__name__)


def _xml_root(content: bytes):
    """Parse strictly first; return recovery diagnostics alongside a salvaged tree."""
    if etree is not None:
        try:
            return etree.fromstring(content, parser=etree.XMLParser(resolve_entities=False)), []
        except etree.XMLSyntaxError as exc:
            parser = etree.XMLParser(recover=True, resolve_entities=False)
            root = etree.fromstring(content, parser=parser)
            warnings = ["Malformed XML was recovered; rows or fields may be incomplete."]
            warnings.extend(str(error) for error in parser.error_log)
            if len(warnings) == 1:
                warnings.append(str(exc))
            return root, warnings
    return _stdlib_etree.fromstring(content), []

# --- source_format / parse_status vocab ------------------------------------
FMT_XML = "ownership_xml"
FMT_XML_LEGACY = "ownership_xml_x0101"
FMT_144 = "form144_xml"
FMT_LEGACY_TXT = "legacy_text"
FMT_RENDERED_HTML = "rendered_html"
FMT_SUBMISSION = "submission_txt"
FMT_PDF = "pdf"
FMT_UNKNOWN = "unknown"

STATUS_OK = "ok"
STATUS_OK_LEGACY_XML = "ok_legacy_xml"
STATUS_OK_LEGACY_TABLE = "ok_legacy_table"
STATUS_EMPTY = "empty"
STATUS_PDF = "unsupported_pdf"
STATUS_REVIEW = "needs_review"
STATUS_ERROR = "error"

# --- canonical row schema ---------------------------------------------------
CANON_FIELDS = [
    # document-level
    "source_format", "parse_status", "parse_warnings", "form", "document_type", "schema_version",
    "period_of_report", "issuer_cik", "issuer_name", "issuer_ticker",
    "owner_cik", "owner_name", "owner_count",
    "owner_sequence", "owner_attribution_status", "economic_row_id", "owner_row_id",
    "is_derivative", "is_natural_person",
    "is_director", "is_officer", "is_ten_percent_owner", "is_other",
    "officer_title", "rule_10b5_1",
    # transaction-level
    "table", "row_kind", "sequence_in_filing", "security_title",
    "transaction_date", "deemed_execution_date", "transaction_form_type",
    "transaction_code", "equity_swap_involved", "shares", "price_per_share",
    "acquired_disposed", "shares_owned_following", "direct_or_indirect",
    "nature_of_ownership", "conversion_or_exercise_price", "exercise_date",
    "expiration_date", "underlying_security_title", "underlying_security_shares",
    "footnote_ids", "footnotes",
    "filing_footnotes", "remarks", "original_submission_date",
    "acceptance_datetime", "acceptance_source", "source_sha256",
    "reported_transaction_value", "derived_transaction_value",
    "legacy_owner_metadata",
    # Form 144-specific (blank on 3/4/5)
    "sec144_person", "sec144_relationship", "sec144_units_to_be_sold",
    "sec144_aggregate_market_value", "sec144_units_outstanding",
    "sec144_approx_sale_date", "sec144_exchange", "sec144_broker",
]


def _blank_row() -> dict:
    """Return a row with every canonical field set to an empty string."""
    return {f: "" for f in CANON_FIELDS}


def _to_bool(v):
    """Read SEC checkbox-style values: 1/true/yes/x -> True, 0/false/no -> False, else None."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "x"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


# ===========================================================================
# Native Ownership XML (Forms 3/4/5, both schema families, multi-owner)
# ===========================================================================

def _ln(tag) -> str:
    """local-name of a possibly namespaced lxml tag."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _find(el, name):
    """Return the first descendant of ``el`` whose local tag name is ``name``, or None."""
    if el is None:
        return None
    for node in el.iter():
        if _ln(node.tag) == name:
            return node
    return None


def _txt(el, name=None) -> str:
    """Text of ``name`` under ``el`` (or of ``el`` itself). Handles the common
    <field><value>X</value></field> wrapper."""
    node = _find(el, name) if name else el
    if node is None:
        return ""
    val = None
    for child in node.iter():
        if _ln(child.tag) == "value":
            val = child
            break
    target = val if (val is not None and val is not node) else node
    return " ".join("".join(target.itertext()).split())


def _extract_ownership_xml(content: bytes) -> bytes:
    """Cut the ``<ownershipDocument>`` XML out of a larger submission; return the input if absent."""
    text = content.decode("utf-8", errors="replace")
    match = re.search(r"<(?:\w+:)?ownershipDocument\b[\s\S]*?</(?:\w+:)?ownershipDocument\s*>", text, re.I)
    if match:
        return match.group(0).encode("utf-8")
    return content


def _footnote_map(root) -> dict:
    """Map footnote ids (without the ``F`` prefix) to their whitespace-collapsed text."""
    out = {}
    for fn in root.iter():
        if _ln(fn.tag) == "footnote":
            fid = fn.attrib.get("id", "").lstrip("Ff")
            if fid:
                out[fid] = " ".join("".join(fn.itertext()).split())
    return out


def _referenced_footnotes(node, notes) -> tuple[str, str]:
    """Collect the footnotes a node references.

    Returns ``("F1;F3", "F1: text | F3: text")``: the ids in order of first use
    and the matching texts.
    """
    ids = []
    for ref in node.iter():
        if _ln(ref.tag) == "footnoteId":
            fid = ref.attrib.get("id", "").lstrip("Ff")
            if fid and fid not in ids:
                ids.append(fid)
    id_str = ";".join(f"F{i}" for i in ids)
    text_str = " | ".join(f"F{i}: {notes.get(i, '')}" for i in ids)
    return id_str, text_str


def parse_ownership_xml(content: bytes) -> tuple[list[dict], str, str]:
    """Parse native Forms 3, 4, or 5 ownership XML into canonical rows."""
    root, warnings = _xml_root(_extract_ownership_xml(content))
    if root is None or _ln(root.tag) != "ownershipDocument":
        raise ValueError("Not an ownershipDocument XML")

    schema = _txt(root, "schemaVersion")
    doc_type = _txt(root, "documentType")
    issuer = _find(root, "issuer")

    # --- reporting owners: support MULTIPLE (joint filings) ---
    owners = [n for n in root.iter() if _ln(n.tag) == "reportingOwner"]
    owner_headers = []
    for owner_seq, ow in enumerate(owners, 1):
        oid = _find(ow, "reportingOwnerId")
        rel = _find(ow, "reportingOwnerRelationship")
        owner_headers.append({
            "owner_cik": _txt(oid, "rptOwnerCik"),
            "owner_name": _txt(oid, "rptOwnerName"),
            "owner_sequence": str(owner_seq),
            "owner_count": str(len(owners)),
            "owner_attribution_status": "single_owner" if len(owners) == 1 else "joint_owner_unallocated",
            "is_director": _to_bool(_txt(rel, "isDirector")),
            "is_officer": _to_bool(_txt(rel, "isOfficer")),
            "is_ten_percent_owner": _to_bool(_txt(rel, "isTenPercentOwner")),
            "is_other": _to_bool(_txt(rel, "isOther")),
            "officer_title": _txt(rel, "officerTitle"),
        })
    if not owner_headers:
        owner_headers = [{"owner_count": "0", "owner_sequence": "0", "owner_attribution_status": "unknown_owner"}]

    header = {
        "form": doc_type or "",
        "document_type": doc_type,
        "schema_version": schema,
        "period_of_report": _txt(root, "periodOfReport"),
        "issuer_cik": _txt(issuer, "issuerCik"),
        "issuer_name": _txt(issuer, "issuerName"),
        "issuer_ticker": _txt(issuer, "issuerTradingSymbol"),
        "rule_10b5_1": _to_bool(_txt(root, "aff10b5One")),
        "remarks": _txt(root, "remarks"),
        "original_submission_date": _txt(root, "dateOfOriginalSubmission"),
    }
    notes = _footnote_map(root)
    header["filing_footnotes"] = json.dumps(notes, sort_keys=True, ensure_ascii=False)
    header["parse_warnings"] = json.dumps(warnings, ensure_ascii=False) if warnings else ""
    is_legacy = schema.upper().startswith("X0101")

    rows: list[dict] = []
    # Modern layout uses table wrappers; X0101 uses flat <...Security> elements.
    specs = [
        ("nonDerivativeTable", "nonDerivativeSecurity", "non_derivative",
         "nonDerivativeTransaction", "nonDerivativeHolding"),
        ("derivativeTable", "derivativeSecurity", "derivative",
         "derivativeTransaction", "derivativeHolding"),
    ]
    seq = 0
    for table_tag, flat_tag, table_name, txn_tag, hold_tag in specs:
        table_el = _find(root, table_tag)
        if table_el is not None:
            entries = [(n, "transaction" if _ln(n.tag) == txn_tag else "holding")
                       for n in table_el if _ln(n.tag) in (txn_tag, hold_tag)]
        else:
            entries = [(n, "transaction" if _find(n, "transactionCoding") is not None else "holding")
                       for n in root if _ln(n.tag) == flat_tag]
        for node, kind in entries:
            seq += 1
            for owner in owner_headers:
                rows.append(_ownership_txn_row(node, {**header, **owner}, table_name, kind, seq, notes))

    status = STATUS_REVIEW if warnings else (STATUS_OK_LEGACY_XML if is_legacy else STATUS_OK)
    fmt = FMT_XML_LEGACY if is_legacy else FMT_XML
    if not rows:
        rows = [{**_blank_row(), **header, **owner, "row_kind": "no_rows"}
                for owner in owner_headers]
        return rows, fmt, STATUS_REVIEW if warnings else STATUS_EMPTY
    return rows, fmt, status


def _ownership_txn_row(node, header, table, kind, seq, notes) -> dict:
    """Build one canonical row from a transaction or holding node of an ownership XML.

    ``header`` holds the filing-level fields (issuer, owner, dates); ``table`` is
    non-derivative or derivative, ``kind`` transaction or holding, ``seq`` its
    order within the filing.
    """
    row = _blank_row()
    row.update(header)
    fid, ftext = _referenced_footnotes(node, notes)
    coding = _find(node, "transactionCoding")
    row.update({
        "table": table,
        "row_kind": kind,
        "sequence_in_filing": str(seq),
        "security_title": _txt(node, "securityTitle"),
        "transaction_date": _txt(node, "transactionDate"),
        "deemed_execution_date": _txt(node, "deemedExecutionDate"),
        "transaction_form_type": _txt(coding, "transactionFormType"),
        "transaction_code": _txt(coding, "transactionCode"),
        "equity_swap_involved": _to_bool(_txt(coding, "equitySwapInvolved")),
        "shares": _txt(node, "transactionShares"),
        "price_per_share": _txt(node, "transactionPricePerShare"),
        "acquired_disposed": _txt(node, "transactionAcquiredDisposedCode"),
        "shares_owned_following": _txt(node, "sharesOwnedFollowingTransaction"),
        "direct_or_indirect": _txt(node, "directOrIndirectOwnership"),
        "nature_of_ownership": _txt(node, "natureOfOwnership"),
        "conversion_or_exercise_price": _txt(node, "conversionOrExercisePrice"),
        "exercise_date": _txt(node, "exerciseDate"),
        "expiration_date": _txt(node, "expirationDate"),
        "underlying_security_title": _txt(node, "underlyingSecurityTitle"),
        "underlying_security_shares": _txt(node, "underlyingSecurityShares"),
        "footnote_ids": fid,
        "footnotes": ftext,
    })
    return row


# ===========================================================================
# Native Form 144 XML.
# ===========================================================================

def parse_form144_xml(content: bytes) -> tuple[list[dict], str, str]:
    """Parse native Form 144 XML into canonical proposed-sale rows."""
    text = content.decode("utf-8", errors="replace")
    embedded = re.search(r"<(?:\w+:)?edgarSubmission\b[\s\S]*?</(?:\w+:)?edgarSubmission\s*>", text, re.I)
    if embedded:
        content = embedded.group(0).encode("utf-8")
    root, warnings = _xml_root(content)
    if root is None:
        raise ValueError("Unparseable Form 144 XML")
    info = _find(root, "issuerInfo")
    if info is None:
        info = root
    person = _txt(info, "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold")
    rels = " | ".join(
        " ".join("".join(n.itertext()).split())
        for n in root.iter() if _ln(n.tag) == "relationshipToIssuer"
    )
    issuer_cik = _txt(info, "issuerCik")
    issuer_name = _txt(info, "issuerName")
    doc_type = _txt(root, "submissionType") or "144"
    sec_nodes = [n for n in root.iter() if _ln(n.tag) == "securitiesInformation"]
    header = {
        "form": doc_type, "document_type": doc_type,
        "issuer_cik": issuer_cik, "issuer_name": issuer_name,
        "owner_name": person, "owner_count": "1", "owner_sequence": "1",
        "owner_attribution_status": "proposed_sale_person",
        "sec144_person": person, "sec144_relationship": rels,
        "parse_warnings": json.dumps(warnings, ensure_ascii=False) if warnings else "",
    }

    rows: list[dict] = []
    for seq, node in enumerate(sec_nodes, start=1):
        row = _blank_row()
        row.update(header)
        broker = _find(node, "brokerOrMarketmakerDetails")
        row.update({
            "table": "form144",
            "row_kind": "proposed_sale",
            "sequence_in_filing": str(seq),
            "security_title": _txt(node, "securitiesClassTitle"),
            "shares": _txt(node, "noOfUnitsSold"),
            "sec144_units_to_be_sold": _txt(node, "noOfUnitsSold"),
            "sec144_aggregate_market_value": _txt(node, "aggregateMarketValue"),
            "sec144_units_outstanding": _txt(node, "noOfUnitsOutstanding"),
            "sec144_approx_sale_date": _txt(node, "approxSaleDate"),
            "sec144_exchange": _txt(node, "securitiesExchangeName"),
            "sec144_broker": _txt(broker, "name") if broker is not None else "",
        })
        rows.append(row)
    if not rows:
        rows = [{**_blank_row(), **header, "row_kind": "no_rows", "table": "form144"}]
        return rows, FMT_144, STATUS_REVIEW if warnings else STATUS_EMPTY
    return rows, FMT_144, STATUS_REVIEW if warnings else STATUS_OK


# ===========================================================================
# Legacy .txt / rendered .htm / SGML submissions -> via vendored parser
# ===========================================================================

_LEGACY_KEYMAP = {
    "ticker": "issuer_ticker",
    "aff10b5_1": "rule_10b5_1",
    "record_kind": "table",
    "transaction_shares": "shares",
    "transaction_price_per_share": "price_per_share",
    "shares_owned_after": "shares_owned_following",
}


def _from_legacy(result: dict) -> list[dict]:
    """Map legacy_form4's {document, transactions} into canonical rows."""
    doc = result["document"]
    rows = []
    for txn in result["transactions"]:
        row = _blank_row()
        # document fields
        for k, v in doc.items():
            key = _LEGACY_KEYMAP.get(k, k)
            if key in row:
                row[key] = v
        # transaction fields
        for k, v in txn.items():
            key = _LEGACY_KEYMAP.get(k, k)
            if key in row:
                row[key] = v
        row["row_kind"] = txn.get("row_kind") or (
            "transaction" if any(txn.get(key) for key in (
                "transaction_date", "transaction_code", "transaction_shares", "acquired_disposed"))
            else "holding")
        warnings = result.get("warnings", [])
        row["parse_warnings"] = json.dumps(warnings, ensure_ascii=False) if warnings else ""
        for b in ("is_director", "is_officer", "is_ten_percent_owner", "is_other",
                  "rule_10b5_1", "equity_swap_involved"):
            row[b] = _to_bool(row[b])
        rows.append(row)
    return rows


def _legacy_result(result: dict, fmt: str) -> tuple[list[dict], str, str]:
    """Preserve legacy diagnostics and distinguish incomplete extraction."""
    status = STATUS_REVIEW if result.get("warnings") else STATUS_OK_LEGACY_TABLE
    return _from_legacy(result), fmt, status


# ===========================================================================
# Format detection + dispatcher
# ===========================================================================

def detect_format(content: bytes, name: str = "") -> str:
    """Classify filing bytes using content markers and an optional filename."""
    if content[:5] == b"%PDF-":
        return FMT_PDF
    head = content[:8000].decode("utf-8", errors="replace").lower()
    if "<sec-document" in head or ("<document>" in head and "<type>" in head):
        return FMT_SUBMISSION
    if re.search(r"<(?:\w+:)?ownershipdocument\b", head):
        return FMT_XML
    if "securitiesinformation" in head or "noofunitssold" in head or \
       ("<submissiontype>144" in head) or "issuerinfo" in head:
        return FMT_144
    if "<!doctype html" in head or "<html" in head or name.lower().endswith((".htm", ".html")):
        return FMT_RENDERED_HTML
    lowered_full = content.decode("utf-8", errors="replace")
    if "STATEMENT OF CHANGES IN BENEFICIAL OWNERSHIP" in lowered_full.upper():
        return FMT_LEGACY_TXT
    return FMT_UNKNOWN


def _parse_filing(content: bytes, name: str = "") -> tuple[list[dict], str, str]:
    """Parse any filing document -> (rows, source_format, parse_status).

    Never raises for an unrecognized layout: returns a flagged placeholder so
    the filing is still recorded.
    """
    fmt = detect_format(content, name)
    try:
        if fmt == FMT_PDF:
            row = _blank_row()
            row["row_kind"] = "unsupported_pdf"
            row["parse_warnings"] = json.dumps(["PDF parsing is unsupported; inspect whether text extraction or OCR is needed."])
            return [row], FMT_PDF, STATUS_PDF

        if fmt == FMT_XML:
            return parse_ownership_xml(content)

        if fmt == FMT_144:
            return parse_form144_xml(content)

        if fmt == FMT_SUBMISSION:
            # Envelope may contain XML, HTML, or legacy text. Prefer native XML.
            text = content.decode("utf-8", errors="replace")
            if re.search(r"<(?:\w+:)?ownershipDocument\b", text, re.I):
                return parse_ownership_xml(content)
            if re.search(r"<(?:\w+:)?(?:securitiesInformation|issuerInfo)\b", text, re.I):
                return parse_form144_xml(content)
            result = legacy.parse_submission_text(text)
            return _legacy_result(result, FMT_SUBMISSION)

        if fmt == FMT_RENDERED_HTML:
            text = content.decode("utf-8", errors="replace")
            result = legacy.parse_rendered_html(text)
            return _legacy_result(result, FMT_RENDERED_HTML)

        if fmt == FMT_LEGACY_TXT:
            text = content.decode("utf-8", errors="replace")
            result = legacy.parse_legacy_text(text)
            return _legacy_result(result, FMT_LEGACY_TXT)

        # unknown
        row = _blank_row()
        row["row_kind"] = "unrecognized_format"
        return [row], FMT_UNKNOWN, STATUS_ERROR
    except legacy.Form4ParseError as exc:
        logger.warning("Legacy parse failed (%s): %s", fmt, exc)
        row = _blank_row()
        row["row_kind"] = "parse_error"
        row["parse_warnings"] = json.dumps([str(exc)])
        return [row], fmt, STATUS_ERROR
    except Exception as exc:  # noqa: BLE001 -- record, never crash the crawl
        logger.error("parse_filing failed (%s): %s", fmt, exc)
        row = _blank_row()
        row["row_kind"] = "parse_error"
        row["parse_warnings"] = json.dumps([f"{type(exc).__name__}: {exc}"])
        return [row], fmt, STATUS_ERROR


def assign_row_ids(rows: list[dict], filing_id: str) -> list[dict]:
    """Identify economic lines separately from owner associations.

    Monetary totals MUST deduplicate ``economic_row_id``. Joint filings link
    each reported owner to each economic line without asserting allocation.
    ``filing_id`` is the accession in the engine, raw SHA-256 for standalone use.
    """
    for fallback_seq, row in enumerate(rows, 1):
        seq = row.get("sequence_in_filing") or str(fallback_seq)
        row["sequence_in_filing"] = seq
        economic_key = f"{filing_id}|{row.get('table', '')}|{seq}"
        row["economic_row_id"] = hashlib.sha256(economic_key.encode()).hexdigest()
        owner_key = str(row.get("owner_cik") or row.get("owner_name") or "unknown").strip()
        owner_key += "|" + str(row.get("owner_sequence") or "1")
        row["owner_row_id"] = hashlib.sha256(f"{economic_key}|{owner_key}".encode()).hexdigest()
        row["is_derivative"] = row.get("table") == "derivative"
        row.setdefault("is_natural_person", None)
        # Ownership XML does not supply a reliable person/entity discriminator.
        # Do not infer one from a person's name or a role flag.
        if row.get("is_natural_person") == "":
            row["is_natural_person"] = None
        if not row.get("owner_attribution_status"):
            composite = "|" in str(row.get("owner_cik", ""))
            row["owner_attribution_status"] = "legacy_joint_owner_unresolved" if composite else (
                "single_owner" if owner_key != "unknown|1" else "unknown_owner")
        try:
            shares = Decimal(str(row.get("shares", "")).replace(",", ""))
            price = Decimal(str(row.get("price_per_share", "")).replace(",", ""))
            row["derived_transaction_value"] = str(shares * price) if shares.is_finite() and price.is_finite() else ""
        except InvalidOperation:
            row["derived_transaction_value"] = ""
        # A shares-times-price calculation is not an independently reported total.
        row.setdefault("reported_transaction_value", "")
    return rows


def submission_acceptance(content: bytes) -> str:
    """Convert the SEC SGML Eastern wall-clock acceptance to offset ISO time."""
    match = re.search(rb"<ACCEPTANCE-DATETIME>\s*(\d{14})", content, re.I)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1).decode(), "%Y%m%d%H%M%S").replace(
            tzinfo=ZoneInfo("America/New_York")).isoformat()
    except ValueError:
        return ""


def parse_filing(content: bytes, name: str = "") -> tuple[list[dict], str, str]:
    """Parse all supported filing formats and retain source-level provenance."""
    rows, fmt, status = _parse_filing(content, name)
    if fmt == FMT_SUBMISSION:
        # Older SGML headers may enumerate several owners even when the legacy
        # body parser only recovered the first. Header identity is reliable;
        # aggregated body role flags cannot be allocated to those owners.
        text = content.decode("utf-8", errors="replace")
        header = re.split(r"</SEC-HEADER>|<DOCUMENT>", text, maxsplit=1, flags=re.I)[0]
        blocks = re.findall(
            r"REPORTING-OWNER:\s*([\s\S]*?)(?=\n\s*(?:REPORTING-OWNER|ISSUER|SUBJECT COMPANY|FILED BY):|\Z)",
            header,
            re.I,
        )
        owners = []
        for block in blocks:
            cik = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", block, re.I)
            owner_name = re.search(r"COMPANY CONFORMED NAME:\s*([^\r\n]+)", block, re.I)
            if cik or owner_name:
                owners.append((cik.group(1) if cik else "", owner_name.group(1).strip() if owner_name else ""))
        if len(owners) > 1:
            expanded = []
            for seq, row in enumerate(rows, 1):
                row["sequence_in_filing"] = row.get("sequence_in_filing") or str(seq)
                metadata = {
                    k: row.get(k)
                    for k in (
                        "owner_cik",
                        "owner_name",
                        "officer_title",
                        "is_officer",
                        "is_director",
                        "is_ten_percent_owner",
                        "is_other",
                    )
                }
                for owner_seq, (cik, owner_name) in enumerate(owners, 1):
                    expanded.append({**row, "owner_cik": cik, "owner_name": owner_name,
                                     "owner_sequence": str(owner_seq), "owner_count": str(len(owners)),
                                     "owner_attribution_status": "joint_owner_unallocated",
                                     "officer_title": "", "is_officer": None, "is_director": None,
                                     "is_ten_percent_owner": None, "is_other": None,
                                     "legacy_owner_metadata": json.dumps(metadata, sort_keys=True)})
            rows = expanded
    digest = hashlib.sha256(content).hexdigest()
    acceptance = submission_acceptance(content)
    for row in rows:
        row["source_format"] = fmt
        row["parse_status"] = status
        row["source_sha256"] = digest
        row["acceptance_datetime"] = acceptance
        row["acceptance_source"] = "submission_header" if acceptance else "missing"
    return assign_row_ids(rows, digest), fmt, status
