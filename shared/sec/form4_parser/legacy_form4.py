#!/usr/bin/env python3
"""Parse Form 4 documents across EDGAR-native historical formats.

Inputs include ownership XML, legacy text tables, rendered HTML, and complete
SGML submissions. Parsed document and transaction dictionaries feed the
canonical normalization layer in ``formats.py``.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


class Form4ParseError(ValueError):
    """Raised when an input is not a supported, recognizable Form 4."""


DOCUMENT_FIELDS = [
    "source_format", "form", "schema_version", "period_of_report",
    "issuer_cik", "issuer_name", "ticker", "owner_cik", "owner_name",
    "is_director", "is_officer", "is_ten_percent_owner", "is_other",
    "officer_title", "aff10b5_1", "accession_number", "filing_date",
    "acceptance_datetime",
    "filing_footnotes",
]

TRANSACTION_FIELDS = [
    "record_kind", "sequence_in_filing", "security_title", "transaction_date",
    "deemed_execution_date", "transaction_form_type", "transaction_code",
    "equity_swap_involved", "transaction_shares", "transaction_price_per_share",
    "acquired_disposed", "shares_owned_after", "direct_or_indirect",
    "nature_of_ownership", "conversion_or_exercise_price", "exercise_date",
    "expiration_date", "underlying_security_title", "underlying_security_shares",
    "footnotes", "footnote_ids", "row_kind",
]


def _blank_document(source_format: str) -> dict[str, str]:
    """Return an empty filing-level record tagged with its source format."""
    output = {field: "" for field in DOCUMENT_FIELDS}
    output.update({"source_format": source_format, "form": "4"})
    return output


def _blank_transaction(kind: str, sequence: int) -> dict[str, str | int]:
    """Return an empty transaction record of the given kind and position in the filing."""
    output: dict[str, str | int] = {field: "" for field in TRANSACTION_FIELDS}
    output.update({"record_kind": kind, "sequence_in_filing": sequence})
    return output


def _local_name(tag: str) -> str:
    """Strip an XML namespace: ``{ns}tag`` becomes ``tag``."""
    return tag.rsplit("}", 1)[-1]


def _nodes(element: ET.Element, name: str) -> list[ET.Element]:
    """Return every descendant of ``element`` whose local tag name is ``name``."""
    return [node for node in element.iter() if _local_name(node.tag) == name]


def _first(element: ET.Element, name: str) -> ET.Element | None:
    """Return the first descendant of ``element`` whose local tag name is ``name``, or None."""
    return next((node for node in element.iter() if _local_name(node.tag) == name), None)


def _text(element: ET.Element | None) -> str:
    """Return an element's whitespace-collapsed text, reading its ``<value>`` child when present."""
    if element is None:
        return ""
    value = _first(element, "value")
    target = value if value is not None and value is not element else element
    return " ".join("".join(target.itertext()).split())


def _field(element: ET.Element, name: str) -> str:
    """Return the text of the first ``name`` element under ``element``."""
    return _text(_first(element, name))


def _bool(value: str) -> str:
    """Normalize SEC checkbox values to ``"true"``, ``"false"`` or ``""``."""
    value = value.strip().lower()
    if value in {"1", "true", "yes", "x"}:
        return "true"
    if value in {"0", "false", "no"}:
        return "false"
    return ""


def _clean_number(value: str) -> str:
    """Strip footnote markers like ``(1)``, dollar signs, commas and spaces from a number."""
    value = re.sub(r"\(\d+\)", "", value or "")
    value = re.sub(r"[$,\s]", "", value)
    return value


def _normalize_date(value: str) -> str:
    """Convert the date formats seen in legacy filings to ISO ``YYYY-MM-DD``.

    Values that match no known format are returned unchanged.
    """
    value = " ".join((value or "").replace("\xa0", " ").split()).strip(".,")
    if not value:
        return ""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return value


def _footnote_ids(value: str) -> list[str]:
    """Return the footnote numbers referenced as ``(n)`` in a text cell."""
    return re.findall(r"\((\d+)\)", value or "")


def _footnote_text(values: Iterable[str], notes: dict[str, str]) -> str:
    """Join the text of every footnote referenced in ``values`` as ``F1: ... | F2: ...``."""
    identifiers: list[str] = []
    for value in values:
        for identifier in _footnote_ids(value):
            if identifier not in identifiers:
                identifiers.append(identifier)
    return " | ".join(f"F{identifier}: {notes.get(identifier, '')}" for identifier in identifiers)


def _xml_footnotes(root: ET.Element) -> dict[str, str]:
    """Map footnote ids (without the ``F`` prefix) to their text in an ownership XML."""
    output: dict[str, str] = {}
    for node in _nodes(root, "footnote"):
        identifier = node.attrib.get("id", "")
        if identifier:
            output[identifier.lstrip("Ff")] = _text(node)
    return output


def _xml_referenced_footnotes(node: ET.Element, notes: dict[str, str]) -> str:
    """Join the text of the footnotes an XML node references as ``F1: ... | F2: ...``."""
    identifiers: list[str] = []
    for ref in _nodes(node, "footnoteId"):
        identifier = ref.attrib.get("id", "").lstrip("Ff")
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)
    return " | ".join(f"F{identifier}: {notes.get(identifier, '')}" for identifier in identifiers)


def parse_xml(text: str, source_format: str = "xml") -> dict:
    """Parse native ownership XML into document and transaction fields."""
    start = text.find("<ownershipDocument")
    end_match = re.search(r"</(?:[A-Za-z0-9_]+:)?ownershipDocument\s*>", text, re.I)
    if start < 0 or end_match is None:
        raise Form4ParseError("No ownershipDocument element was found in the XML input.")
    xml_text = text[start:end_match.end()]
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise Form4ParseError(f"Invalid Ownership XML: {exc}") from exc

    owners = _nodes(root, "reportingOwner")

    def joined(name: str, transform=lambda value: value) -> str:
        """Join one field across all reporting owners with `` | ``, skipping blanks."""
        values = [transform(_field(owner, name)) for owner in owners]
        return " | ".join(value for value in values if value)

    document = _blank_document(source_format)
    document.update(
        {
            "form": _field(root, "documentType") or "4",
            "schema_version": _field(root, "schemaVersion"),
            "period_of_report": _normalize_date(_field(root, "periodOfReport")),
            "issuer_cik": _field(root, "issuerCik"),
            "issuer_name": _field(root, "issuerName"),
            "ticker": _field(root, "issuerTradingSymbol"),
            "owner_cik": joined("rptOwnerCik"),
            "owner_name": joined("rptOwnerName"),
            "is_director": joined("isDirector", _bool),
            "is_officer": joined("isOfficer", _bool),
            "is_ten_percent_owner": joined("isTenPercentOwner", _bool),
            "is_other": joined("isOther", _bool),
            "officer_title": joined("officerTitle"),
            "aff10b5_1": _bool(_field(root, "aff10b5One")),
        }
    )

    notes = _xml_footnotes(root)
    transactions: list[dict] = []
    for tag, kind in (("nonDerivativeTransaction", "non_derivative"), ("derivativeTransaction", "derivative")):
        for sequence, node in enumerate(_nodes(root, tag), start=1):
            row = _blank_transaction(kind, sequence)
            row.update(
                {
                    "security_title": _field(node, "securityTitle"),
                    "transaction_date": _normalize_date(_field(node, "transactionDate")),
                    "deemed_execution_date": _normalize_date(_field(node, "deemedExecutionDate")),
                    "transaction_form_type": _field(node, "transactionFormType"),
                    "transaction_code": _field(node, "transactionCode"),
                    "equity_swap_involved": _bool(_field(node, "equitySwapInvolved")),
                    "transaction_shares": _clean_number(_field(node, "transactionShares")),
                    "transaction_price_per_share": _clean_number(_field(node, "transactionPricePerShare")),
                    "acquired_disposed": _field(node, "transactionAcquiredDisposedCode"),
                    "shares_owned_after": _clean_number(_field(node, "sharesOwnedFollowingTransaction")),
                    "direct_or_indirect": _field(node, "directOrIndirectOwnership"),
                    "nature_of_ownership": _field(node, "natureOfOwnership"),
                    "conversion_or_exercise_price": _clean_number(_field(node, "conversionOrExercisePrice")),
                    "exercise_date": _normalize_date(_field(node, "exerciseDate")),
                    "expiration_date": _normalize_date(_field(node, "expirationDate")),
                    "underlying_security_title": _field(node, "underlyingSecurityTitle"),
                    "underlying_security_shares": _clean_number(_field(node, "underlyingSecurityShares")),
                    "footnotes": _xml_referenced_footnotes(node, notes),
                }
            )
            transactions.append(row)
    return {"document": document, "transactions": transactions, "warnings": []}


@dataclass
class _HtmlNode:
    """Store a minimal HTML tree node for rendered filing tables."""
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["_HtmlNode | str"] = field(default_factory=list)

    def text(self) -> str:
        """Return the node's visible text, HTML-unescaped and whitespace-collapsed."""
        pieces: list[str] = []

        def visit(node: "_HtmlNode | str") -> None:
            """Collect text pieces depth-first."""
            if isinstance(node, str):
                pieces.append(node)
            else:
                if node.tag == "br":
                    pieces.append(" ")
                    return
                if node.tag in {"p", "div", "tr", "td", "th", "li"}:
                    pieces.append(" ")
                start = len(pieces)
                for child in node.children:
                    visit(child)
                if node.tag == "sup":
                    superscript = "".join(pieces[start:]).strip()
                    if superscript.isdigit():
                        pieces[start:] = [f"({superscript})"]
                if node.tag in {"p", "div", "tr", "td", "th", "li"}:
                    pieces.append(" ")

        visit(self)
        return " ".join(html.unescape("".join(pieces)).replace("\xa0", " ").split())


class _TreeParser(HTMLParser):
    """Build the minimal node tree used by the rendered-table parser."""
    def __init__(self) -> None:
        """Start with an empty root node on the open-element stack."""
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("root")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Add an element as a child of the current one; void tags (br, img, ...) are not opened."""
        node = _HtmlNode(tag.lower(), {key.lower(): value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag.lower() not in {"br", "hr", "img", "meta", "link", "input"}:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle a self-closing tag ``<tag/>`` as a start tag that closes at once."""
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag.lower():
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        """Close the most recent open element with this tag (tolerates unclosed children)."""
        target = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == target:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        """Append text to the current element."""
        self.stack[-1].children.append(data)


def _descendants(node: _HtmlNode, tag: str) -> list[_HtmlNode]:
    """Return every descendant of ``node`` with the given tag, in document order."""
    output: list[_HtmlNode] = []
    for child in node.children:
        if isinstance(child, _HtmlNode):
            if child.tag == tag:
                output.append(child)
            output.extend(_descendants(child, tag))
    return output


def _direct(node: _HtmlNode, tag: str) -> list[_HtmlNode]:
    """Return the direct children of ``node`` with the given tag."""
    return [child for child in node.children if isinstance(child, _HtmlNode) and child.tag == tag]


def _table_rows(table: _HtmlNode) -> list[list[str]]:
    """Return the text of each ``<td>`` cell, row by row, for an HTML table."""
    bodies = _direct(table, "tbody")
    containers = bodies or [table]
    output: list[list[str]] = []
    for container in containers:
        for row in _direct(container, "tr"):
            cells = _direct(row, "td")
            if cells:
                output.append([cell.text() for cell in cells])
    return output


def _extract_link(block: str) -> tuple[str, str]:
    """Return ``(CIK, name)`` from the first EDGAR company link in an HTML block."""
    match = re.search(r'<a[^>]+CIK=(\d+)[^>]*>(.*?)</a>', block, re.I | re.S)
    if not match:
        return "", ""
    return match.group(1).zfill(10), " ".join(html.unescape(re.sub(r"<[^>]+>", " ", match.group(2))).split())


def _rendered_notes(root: _HtmlNode) -> dict[str, str]:
    """Read numbered response notes from the rendered explanation section."""
    notes = {}
    for table in _descendants(root, "table"):
        if "explanation of responses:" not in table.text().lower():
            continue
        in_notes = False
        for cells in _table_rows(table):
            text = " ".join(cells)
            marker = re.search(r"Explanation of Responses\s*:\s*", text, re.I)
            if marker:
                in_notes = True
                text = text[marker.end():]
            if not in_notes or re.match(r"Remarks\s*:", text, re.I):
                continue
            match = re.match(r"(?:\((\d+)\)|(\d+)\.)\s*(.*)", text)
            if match:
                notes[match.group(1) or match.group(2)] = match.group(3).strip()
    return notes


def _finish_legacy(document: dict, rows: list[dict], notes: dict[str, str], warnings: list[str]) -> dict:
    """Classify economic rows and retain notes and extraction diagnostics."""
    document["filing_footnotes"] = json.dumps(notes, sort_keys=True, ensure_ascii=False)
    for row in rows:
        row["row_kind"] = (
            "transaction" if any(row.get(key) for key in (
                "transaction_date", "transaction_code", "transaction_shares", "acquired_disposed"))
            else "holding" if row.get("shares_owned_after") else "unparsed_row")
        if row["row_kind"] == "unparsed_row":
            warnings.append("A table row has no recoverable transaction or holding; review its wrapped text.")
        if not row.get("security_title"):
            warnings.append("A parsed row has no security title.")
        for identifier in str(row.get("footnote_ids", "")).split(";"):
            if identifier and identifier.lstrip("Ff") not in notes:
                warnings.append(f"Referenced footnote {identifier} was not recovered.")
    return {"document": document, "transactions": rows, "warnings": list(dict.fromkeys(warnings))}


def _cell_notes(values: Iterable[str], notes: dict[str, str]) -> dict[str, str]:
    """Return ordered footnote identifiers and resolved text for table cells."""
    values = list(values)
    ids = list(dict.fromkeys(identifier for value in values for identifier in _footnote_ids(value)))
    return {"footnote_ids": ";".join(f"F{i}" for i in ids), "footnotes": _footnote_text(values, notes)}


def parse_rendered_html(text: str, source_format: str = "rendered_html") -> dict:
    """Parse an SEC-rendered ownership table from HTML text."""
    parser = _TreeParser()
    parser.feed(text)
    document = _blank_document(source_format)
    warnings = []
    notes = _rendered_notes(parser.root)

    owner_start = re.search(r"Name and Address of Reporting Person", text, re.I)
    issuer_start = re.search(r"Issuer Name", text, re.I)
    owner_block = text[owner_start.start():issuer_start.start()] if owner_start and issuer_start else text
    issuer_end = re.search(r"Date of Earliest Transaction", text[issuer_start.start():], re.I) if issuer_start else None
    issuer_block = (
        text[issuer_start.start():issuer_start.start() + issuer_end.start()]
        if issuer_start and issuer_end
        else text
    )
    owner_cik, owner_name = _extract_link(owner_block)
    owner_links = list(dict.fromkeys(re.findall(r'CIK=(\d+)', owner_block, re.I)))
    if len(owner_links) > 1 or re.search(r'FormData[^>]*>\s*X\s*</span>[\s\S]{0,150}?Form filed by More than One', text, re.I):
        warnings.append("Rendered joint-filing owner metadata may be incomplete; use native XML or the submission header.")
    issuer_cik, issuer_name = _extract_link(issuer_block)
    ticker_match = re.search(r"Trading Symbol[\s\S]{0,500}?class=[\"']FormData[\"'][^>]*>([^<]*)", issuer_block, re.I)
    document.update(
        {
            "owner_cik": owner_cik,
            "owner_name": owner_name,
            "issuer_cik": issuer_cik,
            "issuer_name": issuer_name,
            "ticker": html.unescape(ticker_match.group(1)).strip() if ticker_match else "",
            "is_director": (
                "true"
                if re.search(
                    r'FormData[^>]*>\s*X\s*</span>[^<]*(?:</td>\s*<td[^>]*>)?\s*Director', text, re.I
                )
                else ""
            ),
            "is_officer": (
                "true"
                if re.search(r'FormData[^>]*>\s*X\s*</span>[^<]*(?:</td>\s*<td[^>]*>)?\s*Officer', text, re.I)
                else ""
            ),
            "is_ten_percent_owner": (
                "true"
                if re.search(
                    r'FormData[^>]*>\s*X\s*</span>[^<]*(?:</td>\s*<td[^>]*>)?\s*10% Owner', text, re.I
                )
                else ""
            ),
            "is_other": (
                "true"
                if re.search(r'FormData[^>]*>\s*X\s*</span>[^<]*(?:</td>\s*<td[^>]*>)?\s*Other', text, re.I)
                else ""
            ),
            "aff10b5_1": (
                "true"
                if re.search(r"transaction was made pursuant[\s\S]{0,300}?FormData[^>]*>\s*X", text, re.I)
                else "false"
            ),
        }
    )

    relationship = next((table for table in _descendants(parser.root, "table")
                         if "Officer (give title below)" in table.text()
                         and "Name and Address" not in table.text()), None)
    if relationship is not None and document["is_officer"] == "true":
        relationship_rows = _table_rows(relationship)
        for index, cells in enumerate(relationship_rows[:-1]):
            if any("Officer (give title below)" in cell for cell in cells):
                following = relationship_rows[index + 1]
                if len(following) > 1:
                    document["officer_title"] = following[1]
                break

    transactions: list[dict] = []
    tables = _descendants(parser.root, "table")
    for table in tables:
        table_text = table.text().lower()
        if "table i -" in table_text and "non-derivative securities" in table_text:
            for sequence, cells in enumerate(_table_rows(table), start=1):
                if len(cells) < 11:
                    if any(cells):
                        warnings.append("Skipped a non-derivative HTML row with fewer than 11 cells.")
                    continue
                row = _blank_transaction("non_derivative", sequence)
                row.update(
                    {
                        "security_title": cells[0],
                        "transaction_date": _normalize_date(cells[1]),
                        "deemed_execution_date": _normalize_date(cells[2]),
                        "transaction_form_type": "4",
                        "transaction_code": cells[3],
                        "transaction_shares": _clean_number(cells[5]),
                        "acquired_disposed": cells[6],
                        "transaction_price_per_share": _clean_number(cells[7]),
                        "shares_owned_after": _clean_number(cells[8]),
                        "direct_or_indirect": cells[9],
                        "nature_of_ownership": cells[10],
                    }
                )
                row.update(_cell_notes(cells, notes))
                transactions.append(row)
        elif "table ii -" in table_text and "derivative securities" in table_text:
            for sequence, cells in enumerate(_table_rows(table), start=1):
                if len(cells) < 16:
                    if any(cells):
                        warnings.append("Skipped a derivative HTML row with fewer than 16 cells.")
                    continue
                acquired = _clean_number(cells[6])
                disposed = _clean_number(cells[7])
                row = _blank_transaction("derivative", sequence)
                row.update(
                    {
                        "security_title": cells[0],
                        "conversion_or_exercise_price": _clean_number(cells[1]),
                        "transaction_date": _normalize_date(cells[2]),
                        "deemed_execution_date": _normalize_date(cells[3]),
                        "transaction_form_type": "4",
                        "transaction_code": cells[4],
                        "transaction_shares": acquired or disposed,
                        "acquired_disposed": "A" if acquired else ("D" if disposed else ""),
                        "exercise_date": _normalize_date(cells[8]),
                        "expiration_date": _normalize_date(cells[9]),
                        "underlying_security_title": cells[10],
                        "underlying_security_shares": _clean_number(cells[11]),
                        "transaction_price_per_share": _clean_number(cells[12]),
                        "shares_owned_after": _clean_number(cells[13]),
                        "direct_or_indirect": cells[14],
                        "nature_of_ownership": cells[15],
                    }
                )
                row.update(_cell_notes(cells, notes))
                transactions.append(row)
    dates = sorted(row["transaction_date"] for row in transactions if row["transaction_date"])
    if dates:
        document["period_of_report"] = dates[0]
    if not transactions:
        raise Form4ParseError("Rendered HTML was recognized, but no Form 4 transaction rows were found.")
    return _finish_legacy(document, transactions, notes, warnings)


def _legacy_notes(text: str) -> dict[str, str]:
    """Parse the "Explanation of Responses" footnotes of a text filing into ``{number: text}``."""
    start = re.search(r"Explanation of Responses\s*:", text, re.I)
    if not start:
        return {}
    block = text[start.end():]
    end = re.search(r"(?:</TABLE>|\n/s/|Signature of Reporting Person)", block, re.I)
    if end:
        block = block[:end.start()]
    matches = list(re.finditer(r"^\s*\((\d+)\)\s*", block, re.M))
    output: dict[str, str] = {}
    for index, match in enumerate(matches):
        finish = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        output[match.group(1)] = " ".join(block[match.end():finish].split())
    return output


def _fixed_width_groups(section: str) -> list[list[list[str]]]:
    """Split the fixed-width tables of a legacy text filing into rows of cells.

    Each table starts at a ``<S>`` / ``<C>`` column-marker line whose marker
    positions give the column boundaries; pipe-delimited lines are split on
    ``|`` instead. A table ends at a dashed rule. Returns one list of rows per
    table.
    """
    lines = section.splitlines()
    groups: list[list[list[str]]] = []
    for index, line in enumerate(lines):
        if not re.match(r"^\s*<S>", line):
            continue
        starts = [match.start() for match in re.finditer(r"<[SC]>", line)]
        rows: list[list[str]] = []
        for data_line in lines[index + 1:]:
            if re.match(r"^\s*(?:-\s)?[+-][+-]{8,}", data_line):
                break
            if re.match(r"^</?", data_line) or not data_line.strip():
                continue
            if data_line.lstrip().startswith("|"):
                cells = [cell.strip() for cell in data_line.strip().strip("|").split("|")]
                cells = (cells + [""] * len(starts))[:len(starts)]
            else:
                cells = [data_line[start:end].strip().strip("|") for start, end in zip(starts, starts[1:] + [None])]
            joined = "".join(cells)
            if any(cells) and not joined.replace("+", "").replace("-", "").strip() == "":
                rows.append(cells)
        groups.append(rows)
    return groups


def _coalesce_rows(rows: list[list[str]], width: int, indicator_indexes: tuple[int, ...]) -> list[list[str]]:
    """Join physical fixed-width lines that make up one logical table row."""
    output: list[list[str]] = []
    pending = [""] * width
    for raw in rows:
        cells = (raw + [""] * width)[:width]
        if not any(cells):
            continue
        for index, value in enumerate(cells):
            if value:
                pending[index] = " ".join(part for part in (pending[index], value) if part).strip()
        if any(cells[index] for index in indicator_indexes):
            output.append(pending)
            pending = [""] * width
    if any(pending):
        output.append(pending)
    return output


def _legacy_value(text: str, label: str, next_label: str) -> str:
    """Return the first non-blank line between two labels in a legacy text form."""
    match = re.search(label + r"([^\n]*)\n([\s\S]*?)(?=^\s*" + next_label + r")", text, re.I | re.M)
    if not match:
        return ""
    for line in match.group(2).splitlines():
        value = " ".join(line.split())
        if value:
            return value
    return ""


def parse_legacy_text(text: str, source_format: str = "legacy_text") -> dict:
    """Parse fixed-width, pipe-delimited, or wrapped legacy Form 4 text."""
    if not re.search(r"\bFORM\s+4\b", text, re.I) or not re.search(
        r"STATEMENT OF CHANGES IN BENEFICIAL OWNERSHIP", text, re.I
    ):
        raise Form4ParseError("Input does not look like a legacy Form 4.")
    document = _blank_document(source_format)
    warnings = []
    owner = _legacy_value(text, r"1\.\s*Name.*Reporting Person\*?", r"2\.")
    issuer_line = _legacy_value(text, r"2\.\s*Issuer Name.*Trading Symbol", r"3\.")
    issuer_match = re.match(r"(.+?)\s*\(([^()]+)\)\s*$", issuer_line)
    period = _legacy_value(text, r"4\.\s*Statement for Month/Day/Year", r"5\.")
    document.update(
        {
            "owner_name": owner,
            "issuer_name": issuer_match.group(1).strip() if issuer_match else issuer_line,
            "ticker": issuer_match.group(2).strip() if issuer_match else "",
            "period_of_report": _normalize_date(period),
            "is_director": "true" if re.search(r"\(\s*[Xx]\s*\)\s*Director", text) else "false",
            "is_officer": "true" if re.search(r"\(\s*[Xx]\s*\)\s*Officer", text) else "false",
            "is_ten_percent_owner": "true" if re.search(r"\(\s*[Xx]\s*\)\s*10% Owner", text) else "false",
            "is_other": "true" if re.search(r"\(\s*[Xx]\s*\)\s*Other", text) else "false",
        }
    )
    if document["is_officer"] == "true":
        title = re.search(r"\(\s*[Xx]\s*\)\s*Officer[^\n]*\n([\s\S]*?)(?=\n\s*7\.)", text)
        if title:
            document["officer_title"] = " ".join(
                line.strip() for line in title.group(1).splitlines()
                if line.strip() and not re.fullmatch(r"[_\s]+", line))
    notes = _legacy_notes(text)
    transactions: list[dict] = []

    table_i = re.search(r"Table I[\s-]*Non-Derivative([\s\S]*?)(?=FORM 4 \(Continued\)|TABLE II|Table II)", text, re.I)
    if table_i:
        groups = _fixed_width_groups(table_i.group(1))
        rows = [row for group in groups for row in _coalesce_rows(group, 11, (1, 3, 5, 6, 8, 9))]
        for sequence, cells in enumerate(rows, start=1):
            if len(cells) < 11:
                continue
            row = _blank_transaction("non_derivative", sequence)
            row.update(
                {
                    "security_title": re.sub(r"\(\d+\)", "", cells[0]).strip(),
                    "transaction_date": _normalize_date(cells[1]),
                    "deemed_execution_date": _normalize_date(cells[2]),
                    "transaction_form_type": "4",
                    "transaction_code": cells[3],
                    "equity_swap_involved": _bool(cells[4]),
                    "transaction_shares": _clean_number(cells[5]),
                    "acquired_disposed": cells[6],
                    "transaction_price_per_share": _clean_number(cells[7]),
                    "shares_owned_after": _clean_number(cells[8]),
                    "direct_or_indirect": cells[9],
                    "nature_of_ownership": re.sub(r"\(\d+\)", "", cells[10]).strip(),
                    **_cell_notes(cells, notes),
                }
            )
            transactions.append(row)

    table_ii = re.search(r"TABLE II[\s-]*Derivative([\s\S]*?)(?=Explanation of Responses)", text, re.I)
    if table_ii:
        groups = _fixed_width_groups(table_ii.group(1))
        first_half = _coalesce_rows(groups[0] if groups else [], 10, (2, 4, 6, 7))
        second_half = _coalesce_rows(groups[1] if len(groups) > 1 else [], 6, (1, 2, 3, 4, 5))
        if len(groups) > 2:
            warnings.append("Additional derivative table blocks were not parsed.")
        if len(first_half) != len(second_half):
            warnings.append("Derivative table halves have different row counts; review row alignment.")
        for sequence, first_cells in enumerate(first_half, start=1):
            second_cells = second_half[sequence - 1] if sequence <= len(second_half) else [""] * 6
            first_cells += [""] * max(0, 10 - len(first_cells))
            second_cells += [""] * max(0, 6 - len(second_cells))
            acquired = _clean_number(first_cells[6])
            disposed = _clean_number(first_cells[7])
            all_cells = first_cells + second_cells
            row = _blank_transaction("derivative", sequence)
            row.update(
                {
                    "security_title": re.sub(r"\(\d+\)", "", first_cells[0]).strip(),
                    "conversion_or_exercise_price": _clean_number(first_cells[1]),
                    "transaction_date": _normalize_date(first_cells[2]),
                    "deemed_execution_date": _normalize_date(first_cells[3]),
                    "transaction_form_type": "4",
                    "transaction_code": first_cells[4],
                    "equity_swap_involved": _bool(first_cells[5]),
                    "transaction_shares": acquired or disposed,
                    "acquired_disposed": "A" if acquired else ("D" if disposed else ""),
                    "exercise_date": _normalize_date(re.sub(r"\(\d+\)", "", first_cells[8]).strip()),
                    "expiration_date": _normalize_date(re.sub(r"\(\d+\)", "", first_cells[9]).strip()),
                    "underlying_security_title": re.sub(r"\(\d+\)", "", second_cells[0]).strip(),
                    "underlying_security_shares": _clean_number(second_cells[1]),
                    "transaction_price_per_share": _clean_number(second_cells[2]),
                    "shares_owned_after": _clean_number(second_cells[3]),
                    "direct_or_indirect": second_cells[4],
                    "nature_of_ownership": re.sub(r"\(\d+\)", "", second_cells[5]).strip(),
                    **_cell_notes(all_cells, notes),
                }
            )
            transactions.append(row)
    if not transactions:
        raise Form4ParseError("Legacy Form 4 was recognized, but no transaction rows were parsed.")
    return _finish_legacy(document, transactions, notes, warnings)


def _submission_header(text: str) -> dict[str, str]:
    """Read filer and issuer identity from the ``<SEC-HEADER>`` of a full submission."""
    header_match = re.search(r"<SEC-HEADER>([\s\S]*?)</SEC-HEADER>", text, re.I)
    header = header_match.group(1) if header_match else text[:10000]

    def value(label: str) -> str:
        """Return the value after ``label:`` in the header, or ``""``."""
        match = re.search(r"^\s*" + label + r":\s*(.+)$", header, re.I | re.M)
        return match.group(1).strip() if match else ""

    owner_block = re.search(
        r"REPORTING-OWNER:\s*([\s\S]*?)(?=\n(?:ISSUER|SUBJECT COMPANY):|\Z)",
        header,
        re.I,
    )
    issuer_block = re.search(
        r"(?:ISSUER|SUBJECT COMPANY):\s*([\s\S]*?)(?=\n(?:REPORTING-OWNER|FILED BY):|\Z)",
        header,
        re.I,
    )

    def company(block: re.Match | None) -> tuple[str, str]:
        """Return ``(CIK, conformed name)`` from a header company block."""
        if not block:
            return "", ""
        name = re.search(r"COMPANY CONFORMED NAME:\s*(.+)", block.group(1), re.I)
        cik = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", block.group(1), re.I)
        return (cik.group(1).zfill(10) if cik else "", name.group(1).strip() if name else "")

    owner_cik, owner_name = company(owner_block)
    issuer_cik, issuer_name = company(issuer_block)
    acceptance_match = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d+)", text, re.I)
    acceptance = acceptance_match.group(1) if acceptance_match else value("ACCEPTANCE-DATETIME")
    if re.fullmatch(r"\d{14}", acceptance):
        acceptance = datetime.strptime(acceptance, "%Y%m%d%H%M%S").isoformat()
    return {
        "accession_number": value("ACCESSION NUMBER"),
        "filing_date": _normalize_date(value("FILED AS OF DATE")),
        "acceptance_datetime": acceptance,
        "period_of_report": _normalize_date(value("CONFORMED PERIOD OF REPORT")),
        "form": value("CONFORMED SUBMISSION TYPE") or "4",
        "owner_cik": owner_cik,
        "owner_name": owner_name,
        "issuer_cik": issuer_cik,
        "issuer_name": issuer_name,
    }


def parse_submission_text(text: str) -> dict:
    """Select and parse the ownership document inside an SGML submission."""
    header = _submission_header(text)
    documents = re.findall(r"<DOCUMENT>([\s\S]*?)</DOCUMENT>", text, re.I)
    candidates: list[str] = []
    for document in documents:
        type_match = re.search(r"<TYPE>\s*([^\n<]+)", document, re.I)
        if type_match and type_match.group(1).strip().upper() in {"4", "4/A"}:
            text_match = re.search(r"<TEXT>([\s\S]*?)</TEXT>", document, re.I)
            candidates.append(text_match.group(1) if text_match else document)
    if not candidates:
        candidates = [text]

    errors: list[str] = []
    for candidate in candidates:
        candidate = re.sub(r"^\s*<XML>\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*</XML>\s*$", "", candidate, flags=re.I)
        try:
            if "<ownershipDocument" in candidate:
                result = parse_xml(candidate, "submission_txt+xml")
            elif re.search(r"<!DOCTYPE\s+html|<html\b", candidate, re.I):
                result = parse_rendered_html(candidate, "submission_txt+html")
            else:
                result = parse_legacy_text(candidate, "submission_txt+legacy_text")
            for key, value in header.items():
                if value and not result["document"].get(key):
                    result["document"][key] = value
            return result
        except Form4ParseError as exc:
            errors.append(str(exc))
    raise Form4ParseError("No supported Form 4 document was found in the submission: " + " | ".join(errors))


def detect_format(data: bytes | str, source_name: str = "") -> str:
    """Identify the legacy filing representation from content and source name."""
    if isinstance(data, bytes):
        if data.startswith(b"%PDF"):
            return "pdf"
        text = data.decode("utf-8", errors="replace")
    else:
        text = data
    lowered = text.lower()
    if "<sec-document" in lowered or "<document>" in lowered:
        return "submission_txt"
    if "<ownershipdocument" in lowered:
        return "xml"
    if "<!doctype html" in lowered or "<html" in lowered or source_name.lower().endswith((".htm", ".html")):
        return "rendered_html"
    if re.search(r"\bFORM\s+4\b", text, re.I) and "STATEMENT OF CHANGES IN BENEFICIAL OWNERSHIP" in text.upper():
        return "legacy_text"
    return "unknown"


def parse_form4(data: bytes | str, source_name: str = "") -> dict:
    """Dispatch Form 4 content to the matching legacy parser."""
    detected = detect_format(data, source_name)
    if detected == "pdf":
        raise Form4ParseError("PDF Form 4 parsing is unsupported; inspect whether text extraction or OCR is needed.")
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if detected == "submission_txt":
        return parse_submission_text(text)
    if detected == "xml":
        return parse_xml(text)
    if detected == "rendered_html":
        return parse_rendered_html(text)
    if detected == "legacy_text":
        return parse_legacy_text(text)
    raise Form4ParseError(f"Unrecognized Form 4 format for {source_name or 'input'}.")


def parse_path(path: Path) -> dict:
    """Read and parse a Form 4 filing from ``path``."""
    return parse_form4(path.read_bytes(), path.name)
