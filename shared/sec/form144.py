"""Discover, fetch, parse, and store electronic SEC Form 144 filings.

Quarterly discovery uses EDGAR ``master.idx`` and therefore includes both 144
and 144/A. Network access is isolated behind ``EdgarClient``; parsing and index
tests are fully offline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import io
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

SUBMISSION = Path(__file__).resolve().parents[2]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

from shared.config import DATA_ROOT, DEFAULT_CONFIG
from shared.sec.form4_parser.client import EdgarClient

MASTER_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
ARCHIVES_URL = "https://www.sec.gov/Archives/"
DEFAULT_OUTPUT = DATA_ROOT / "sec/form144"


@dataclass(frozen=True)
class Filing144:
    """Identify one Form 144 filing discovered in an SEC master index."""
    cik: int
    company_name: str
    form: str
    filing_date: str
    submission_path: str
    accession: str


def _local(tag: str) -> str:
    """Strip an XML namespace: ``{ns}tag`` becomes ``tag``."""
    return tag.rsplit("}", 1)[-1]


def _nodes(root: ET.Element, name: str) -> list[ET.Element]:
    """Return every descendant element of ``root`` whose local tag name is ``name``."""
    return [node for node in root.iter() if _local(node.tag) == name]


def _text(node: ET.Element | None, name: str) -> str:
    """Return the whitespace-collapsed text of the first ``name`` element under ``node``, or ``""``."""
    if node is None:
        return ""
    for child in node.iter():
        if _local(child.tag) == name:
            return " ".join("".join(child.itertext()).split())
    return ""


def parse_master_index(content: bytes | str, ciks: set[int]) -> list[Filing144]:
    """Return requested 144/144-A records from a quarterly master index."""
    text = content.decode("latin-1") if isinstance(content, bytes) else content
    wanted = {int(cik) for cik in ciks}
    result = []
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) != 5 or fields[0] == "CIK":
            continue
        cik_text, company, form, filing_date, path = fields
        try:
            cik = int(cik_text)
        except ValueError:
            continue
        if cik not in wanted or form.strip().upper() not in {"144", "144/A"}:
            continue
        match = re.search(r"(\d{10}-\d{2}-\d{6})\.txt$", path)
        accession = match.group(1) if match else Path(path).stem
        result.append(Filing144(cik, company.strip(), form.strip().upper(), filing_date, path, accession))
    return result


def extract_primary_xml(submission: bytes, form: str = "144") -> bytes:
    """Extract the primary 144 XML document from an EDGAR complete submission."""
    text = submission.decode("utf-8", errors="replace")
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, flags=re.I | re.S)
    candidates = []
    for document in documents:
        type_match = re.search(r"<TYPE>\s*([^\r\n<]+)", document, flags=re.I)
        filename_match = re.search(r"<FILENAME>\s*([^\r\n<]+)", document, flags=re.I)
        xml_match = re.search(r"<XML>(.*?)</XML>", document, flags=re.I | re.S)
        doc_type = type_match.group(1).strip().upper() if type_match else ""
        filename = filename_match.group(1).strip().lower() if filename_match else ""
        if xml_match and (doc_type in {"144", "144/A"} or filename.endswith(".xml")):
            candidates.append((doc_type in {form.upper(), "144", "144/A"}, xml_match.group(1)))
    if candidates:
        preferred = next((xml for is_primary, xml in candidates if is_primary), candidates[0][1])
        return preferred.strip().encode("utf-8")
    stripped = submission.lstrip()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<edgarSubmission"):
        return submission
    raise ValueError("Complete submission contains no Form 144 XML document")


def parse_form144_xml(xml: bytes | str, filing: Filing144 | None = None) -> pd.DataFrame:
    """Parse one public Form 144 XML into one row per securitiesInformation."""
    root = ET.fromstring(xml)
    issuer = next(iter(_nodes(root, "issuerInfo")), root)
    filer = _text(issuer, "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold")
    relationships = " | ".join(
        " ".join("".join(node.itertext()).split()) for node in _nodes(issuer, "relationshipToIssuer")
    )
    plan_values = []
    for name in (
        "is10b5OnePlan", "is10b5OneTradingArrangement", "dateOfAdoptionOfTradingPlan",
        "dateOfAdoptionOfTradingArrangement", "planAdoptionDate",
    ):
        for node in _nodes(root, name):
            value = " ".join("".join(node.itertext()).split())
            if value:
                plan_values.append(f"{name}={value}")
    rows = []
    for sequence, security in enumerate(_nodes(root, "securitiesInformation"), start=1):
        broker = next(iter(_nodes(security, "brokerOrMarketmakerDetails")), None)
        rows.append({
            "filing_date": filing.filing_date if filing else "",
            "accession": filing.accession if filing else "",
            "form": filing.form if filing else "144",
            "issuer_cik": _text(issuer, "issuerCik"),
            "issuer_name": _text(issuer, "issuerName"),
            "filer": filer,
            "relationship_to_issuer": relationships,
            "sequence_in_filing": sequence,
            "securities_class": _text(security, "securitiesClassTitle"),
            "shares_to_be_sold": pd.to_numeric(_text(security, "noOfUnitsSold"), errors="coerce"),
            "aggregate_market_value": pd.to_numeric(_text(security, "aggregateMarketValue"), errors="coerce"),
            "approximate_sale_date": pd.to_datetime(_text(security, "approxSaleDate"), errors="coerce"),
            "broker": _text(broker, "name"),
            "plan_10b5_1": " | ".join(dict.fromkeys(plan_values)),
        })
    return pd.DataFrame(rows)


def default_ciks(
    crosswalk_path: Path = DEFAULT_CONFIG.crosswalk_path,
    universe_path: Path = DEFAULT_CONFIG.universe_path,
) -> set[int]:
    """Return CIKs in the configured point-in-time investment universe."""
    intervals = set(pd.read_parquet(universe_path, columns=["interval_id"])["interval_id"])
    crosswalk = pd.read_parquet(crosswalk_path, columns=["interval_id", "cik"])
    values = crosswalk.loc[crosswalk["interval_id"].isin(intervals), "cik"]
    return set(pd.to_numeric(values, errors="coerce").dropna().astype(int))


def collect_form144(
    start_year: int, start_quarter: int, end_year: int, end_quarter: int,
    ciks: set[int] | None = None, output_dir: Path = DEFAULT_OUTPUT,
    client: EdgarClient | None = None,
) -> pd.DataFrame:
    """Discover and download Form 144 filings for an inclusive quarter range."""
    client = client or EdgarClient(min_interval=0.125, cache_dir=output_dir / "raw_cache")
    ciks = ciks or default_ciks()
    discovered: list[Filing144] = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if (year, quarter) < (start_year, start_quarter) or (year, quarter) > (end_year, end_quarter):
                continue
            index = client.get_bytes(MASTER_URL.format(year=year, quarter=quarter))
            discovered.extend(parse_master_index(index, ciks))
    frames = []
    for filing in discovered:
        submission = client.get_bytes(ARCHIVES_URL + filing.submission_path)
        frame = parse_form144_xml(extract_primary_xml(submission, filing.form), filing)
        frame["submission_path"] = filing.submission_path
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_dir / "form144.parquet", index=False)
    pd.DataFrame([asdict(item) for item in discovered]).to_parquet(
        output_dir / "discovered_filings.parquet", index=False,
    )
    return result


def main() -> int:
    """Collect Form 144 rows for the requested inclusive quarter range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023Q2")
    parser.add_argument("--end", default="2026Q2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not os.environ.get("SEC_USER_AGENT", "").strip():
        parser.error("SEC_USER_AGENT is required (for example: 'Organization contact@example.com')")
    start_year, start_quarter = int(args.start[:4]), int(args.start[-1])
    end_year, end_quarter = int(args.end[:4]), int(args.end[-1])
    result = collect_form144(start_year, start_quarter, end_year, end_quarter, output_dir=args.output_dir)
    print(f"wrote {len(result)} Form 144 security rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
