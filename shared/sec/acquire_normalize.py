"""Acquire and normalize official SEC quarterly Form 3/4/5 ZIP data sets.

The standard path writes submissions, transactions, owners, and footnotes for
signal loaders; the lossless path writes all seven source tables separately.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import pandas as pd
import requests

SUBMISSION = Path(__file__).resolve().parents[2]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

from shared.config import DATA_ROOT


BASE_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
DEFAULT_ROOT = DATA_ROOT / "sec"
VERSION = 1
ALL_VERSION = 1
ALL_TABLES = {
    "submissions": "SUBMISSION",
    "owners": "REPORTINGOWNER",
    "nonderiv_transactions": "NONDERIV_TRANS",
    "nonderiv_holdings": "NONDERIV_HOLDING",
    "deriv_transactions": "DERIV_TRANS",
    "deriv_holdings": "DERIV_HOLDING",
    "footnotes": "FOOTNOTES",
}
ALL_OUTPUT_ROOT = DATA_ROOT / "sec/all_forms"


def quarter_url(year: int, quarter: int) -> str:
    """Return the SEC data-set URL for a calendar quarter."""
    if quarter not in {1, 2, 3, 4}:
        raise ValueError("quarter must be 1, 2, 3, or 4")
    return f"{BASE_URL}/{int(year)}q{int(quarter)}_form345.zip"


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(archive: zipfile.ZipFile, name: str, usecols=None) -> pd.DataFrame:
    """Read one tab-separated table from an SEC quarterly ZIP, keeping every value as text."""
    with archive.open(f"{name}.tsv") as handle:
        return pd.read_csv(
            handle, sep="\t", dtype="string", keep_default_na=False,
            usecols=usecols, on_bad_lines="error",
        )


def _date(values: pd.Series) -> pd.Series:
    """Parse SEC dates written like ``05-Jul-2021``; bad values become NaT."""
    return pd.to_datetime(values, format="%d-%b-%Y", errors="coerce")


class SECDownloader:
    """Sequential downloader with a polite minimum interval and no 403 bypass."""

    def __init__(self, user_agent: str, interval: float = 0.35):
        """Create a polite SEC downloader.

        ``user_agent`` must contain a contact email (SEC policy). Requests are
        spaced at least ``interval`` seconds apart, with a floor of 0.2 s.
        """
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must identify an organization and contact email")
        self.interval = max(0.2, float(interval))
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def download(self, url: str, destination: Path) -> Path:
        """Download and validate an SEC quarterly ZIP at ``destination``."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(5):
            time.sleep(max(0.0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, timeout=(30, 120))
                if response.status_code == 403:
                    raise PermissionError(f"SEC denied access; stopped without bypass: {url}")
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 4:
                        response.raise_for_status()
                    time.sleep(max(2 ** (attempt + 3), int(response.headers.get("Retry-After", "0") or 0)))
                    continue
                response.raise_for_status()
                temporary.write_bytes(response.content)
                with zipfile.ZipFile(temporary) as archive:
                    if "SUBMISSION.tsv" not in archive.namelist() or archive.testzip() is not None:
                        raise ValueError("Invalid SEC quarterly ZIP")
                temporary.replace(destination)
                return destination
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 2))
        raise RuntimeError(f"Failed download: {url}")


def normalize_archive(archive_path: Path, output_dir: Path, source_quarter: str) -> dict:
    """Normalize one ZIP into the exact schemas read by the signal loaders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        sub = _read(archive, "SUBMISSION")
        form_counts = sub["DOCUMENT_TYPE"].value_counts().to_dict()
        sub = sub.loc[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])].copy()
        sub = sub.rename(columns={
            "ACCESSION_NUMBER": "accession", "FILING_DATE": "filing_date",
            "PERIOD_OF_REPORT": "period_of_report", "DATE_OF_ORIG_SUB": "date_of_original_submission",
            "DOCUMENT_TYPE": "form", "ISSUERCIK": "issuer_cik", "ISSUERNAME": "issuer_name",
            "ISSUERTRADINGSYMBOL": "ticker", "REMARKS": "remarks",
        })
        sub.columns = sub.columns.str.lower()
        for name in ("filing_date", "period_of_report", "date_of_original_submission"):
            if name in sub:
                sub[name] = _date(sub[name])
        sub["issuer_cik"] = pd.to_numeric(sub["issuer_cik"], errors="coerce").astype("Int64")
        sub["ticker"] = sub["ticker"].str.strip().str.upper()
        sub["is_amendment"] = sub["form"].eq("4/A")
        sub["source_quarter"] = source_quarter
        sub["filing_url"] = (
            "https://www.sec.gov/Archives/edgar/data/" + sub["issuer_cik"].astype("string") + "/"
            + sub["accession"].str.replace("-", "", regex=False) + "/" + sub["accession"] + "-index.html"
        )
        if sub["accession"].duplicated().any():
            raise ValueError("Duplicate submission primary key within quarter")
        accessions = set(sub["accession"])

        owners = _read(archive, "REPORTINGOWNER", usecols=[
            "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP",
            "RPTOWNER_TITLE", "RPTOWNER_TXT",
        ])
        owners = owners.loc[owners["ACCESSION_NUMBER"].isin(accessions)].rename(columns={
            "ACCESSION_NUMBER": "accession", "RPTOWNERCIK": "owner_cik", "RPTOWNERNAME": "owner_name",
            "RPTOWNER_RELATIONSHIP": "relationship", "RPTOWNER_TITLE": "officer_title",
            "RPTOWNER_TXT": "other_role_text",
        })
        owners["owner_cik"] = pd.to_numeric(owners["owner_cik"], errors="coerce").astype("Int64")
        relationship = owners["relationship"].str.lower().str.replace(r"[^a-z]", "", regex=True)
        for name, phrase in (("is_officer", "officer"), ("is_director", "director"),
                             ("is_ten_percent_owner", "tenpercentowner"), ("is_other", "other")):
            owners[name] = relationship.str.contains(phrase, regex=False)
        title = owners["officer_title"].str.lower()
        owners["is_ceo"] = title.str.contains(r"\bceo\b|chief executive", regex=True)
        owners["is_cfo"] = title.str.contains(r"\bcfo\b|chief financial", regex=True)
        owners["is_coo"] = title.str.contains(r"\bcoo\b|chief operating", regex=True)
        owners["is_chair"] = title.str.contains(r"\bchair(?:man|woman|person)?\b", regex=True)
        owners["is_president"] = title.str.contains(r"\bpresident\b", regex=True) & ~title.str.contains(
            r"vice|\bvp\b", regex=True
        )
        owners["role"] = "Other / unknown"
        owners.loc[owners["is_ten_percent_owner"], "role"] = "10% owner"
        owners.loc[owners["is_director"], "role"] = "Director"
        owners.loc[owners["is_officer"], "role"] = "Other officer"
        owners.loc[owners[["is_ceo", "is_cfo", "is_coo", "is_chair", "is_president"]].any(axis=1), "role"] = (
            "Top executive / chair"
        )
        if owners.duplicated(["accession", "owner_cik"]).any():
            raise ValueError("Duplicate reporting-owner primary key")
        sub = sub.join(owners.groupby("accession")["owner_cik"].nunique().rename("owner_count"), on="accession")

        trans = _read(archive, "NONDERIV_TRANS")
        trans = trans.loc[
            trans["ACCESSION_NUMBER"].isin(accessions) & trans["TRANS_CODE"].isin(["P", "S"])
        ].copy().rename(columns={
            "ACCESSION_NUMBER": "accession", "NONDERIV_TRANS_SK": "transaction_id",
            "TRANS_DATE": "transaction_date", "TRANS_CODE": "transaction_code",
            "TRANS_SHARES": "shares", "TRANS_PRICEPERSHARE": "price",
            "TRANS_ACQUIRED_DISP_CD": "acquired_disposed", "SHRS_OWND_FOLWNG_TRANS": "shares_after",
            "DIRECT_INDIRECT_OWNERSHIP": "direct_indirect", "NATURE_OF_OWNERSHIP": "ownership_nature",
        })
        trans.columns = trans.columns.str.lower()
        trans["transaction_date"] = _date(trans["transaction_date"])
        if "deemed_execution_date" in trans:
            trans["deemed_execution_date"] = _date(trans["deemed_execution_date"])
        for name in ("shares", "price", "shares_after", "valu_ownd_folwng_trans"):
            if name in trans:
                trans[name] = pd.to_numeric(trans[name], errors="coerce")
        trans["trade_value"] = trans["shares"] * trans["price"]
        trans["side"] = trans["transaction_code"].map({"P": 1, "S": -1}).astype("int8")
        trans["transaction_key"] = trans["accession"] + ":" + trans["transaction_id"]
        if trans["transaction_key"].duplicated().any():
            raise ValueError("Duplicate transaction primary key")

        foot = _read(archive, "FOOTNOTES")
        foot = foot.loc[foot["ACCESSION_NUMBER"].isin(set(trans["accession"]))].rename(columns={
            "ACCESSION_NUMBER": "accession", "FOOTNOTE_ID": "footnote_id", "FOOTNOTE_TXT": "footnote_text",
        })
        flags = foot.assign(
            mentions_10b5_1=foot["footnote_text"].str.contains(r"10b5[\s\-–]*1", case=False, regex=True),
            mentions_private=foot["footnote_text"].str.contains(r"\bprivate(?:ly)?\b", case=False, regex=True),
        ).groupby("accession")[["mentions_10b5_1", "mentions_private"]].max()
        sub = sub.join(flags, on="accession")
        for name in ("mentions_10b5_1", "mentions_private"):
            sub[name] = sub[name].astype("boolean")

        tables = {"submissions": sub, "transactions": trans, "owners": owners, "footnotes": foot}
        for name, frame in tables.items():
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        manifest = {
            "version": VERSION, "quarter": source_quarter, "raw_sha256": _sha256(archive_path),
            "all_form_counts": form_counts, "form4_submissions": len(sub),
            "transactions": len(trans), "owners": len(owners), "footnotes": len(foot),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


def _all_table(archive: zipfile.ZipFile, source_name: str, source_quarter: str) -> pd.DataFrame:
    """Read one SEC table without filtering and apply stable storage types."""
    frame = _read(archive, source_name)
    frame.columns = frame.columns.str.lower()
    frame = frame.rename(columns={"accession_number": "accession"})
    for column in frame.columns:
        if column.endswith("_date") or column in {
            "filing_date", "period_of_report", "date_of_orig_sub", "trans_date",
            "deemed_execution_date", "exercise_date", "excercise_date", "expiration_date",
        }:
            frame[column] = _date(frame[column])
    frame["source_quarter"] = source_quarter
    return frame


def normalize_all_archive(archive_path: Path, output_dir: Path, source_quarter: str) -> dict:
    """Write the seven complete SEC tables for one quarterly ZIP.

    Source field names are retained in lower case (apart from the common
    ``ACCESSION_NUMBER`` -> ``accession`` key). Empty strings remain strings;
    only SEC date fields are converted to timestamps. This avoids lossy numeric
    coercion of identifiers and footnote-linked values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        frames = {
            output_name: _all_table(archive, source_name, source_quarter)
            for output_name, source_name in ALL_TABLES.items()
        }
        submissions = frames["submissions"]
        forms = submissions["document_type"].astype("string").str.strip()
        keep = forms.str.fullmatch(r"[345](?:/A)?", na=False)
        submissions = submissions.loc[keep].copy()
        frames["submissions"] = submissions
        accessions = set(submissions["accession"])
        for name, frame in frames.items():
            if name != "submissions":
                frame = frame.loc[frame["accession"].isin(accessions)].copy()
                frames[name] = frame
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
            counts[name] = int(len(frame))
    manifest = {
        "schema_version": ALL_VERSION,
        "quarter": source_quarter,
        "raw_file": archive_path.name,
        "raw_sha256": _sha256(archive_path),
        "forms": {str(k): int(v) for k, v in submissions["document_type"].value_counts().sort_index().items()},
        "tables": counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def normalize_all(raw_dir: Path, output_root: Path = ALL_OUTPUT_ROOT) -> dict:
    """Normalize every locally available quarterly ZIP and write a root manifest."""
    archives = sorted(raw_dir.glob("*_form345.zip"))
    if not archives:
        raise FileNotFoundError(f"No *_form345.zip archives under {raw_dir}")
    quarters = []
    total_forms: dict[str, int] = {}
    total_tables = {name: 0 for name in ALL_TABLES}
    for archive in archives:
        stem = archive.name.split("_", 1)[0].lower()
        if not (len(stem) == 6 and stem[4] == "q"):
            raise ValueError(f"Unexpected quarterly archive name: {archive.name}")
        year, quarter = int(stem[:4]), int(stem[-1])
        manifest = normalize_all_archive(
            archive, output_root / f"year={year}" / f"quarter={quarter}", stem,
        )
        quarters.append(manifest)
        for form, count in manifest["forms"].items():
            total_forms[form] = total_forms.get(form, 0) + count
        for table, count in manifest["tables"].items():
            total_tables[table] += count
    root_manifest = {
        "schema_version": ALL_VERSION,
        "quarter_count": len(quarters),
        "forms": dict(sorted(total_forms.items())),
        "tables": total_tables,
        "quarters": quarters,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8")
    return root_manifest


def acquire_quarter(year: int, quarter: int, root: Path = DEFAULT_ROOT) -> dict:
    """Download and normalize one quarter, returning its manifest."""
    user_agent = os.environ.get("SEC_USER_AGENT", "")
    raw = root / "raw" / f"{year}q{quarter}_form345.zip"
    SECDownloader(user_agent).download(quarter_url(year, quarter), raw)
    out = root / "normalized" / f"year={year}" / f"quarter={quarter}"
    return normalize_archive(raw, out, f"{year}q{quarter}")


def synthetic_zip_bytes() -> bytes:
    """Return a tiny deterministic ZIP used by the offline self-test."""
    accession = "0000000001-21-000001"
    tables = {
        "SUBMISSION.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "FILING_DATE": "05-Jul-2021", "PERIOD_OF_REPORT": "02-Jul-2021",
            "DATE_OF_ORIG_SUB": "", "DOCUMENT_TYPE": "4", "ISSUERCIK": "1234",
            "ISSUERNAME": "Example Bio", "ISSUERTRADINGSYMBOL": "EXB", "REMARKS": "",
        }]),
        "REPORTINGOWNER.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "RPTOWNERCIK": "5678", "RPTOWNERNAME": "A Owner",
            "RPTOWNER_RELATIONSHIP": "Officer", "RPTOWNER_TITLE": "CEO", "RPTOWNER_TXT": "",
        }]),
        "NONDERIV_TRANS.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "NONDERIV_TRANS_SK": "1", "TRANS_DATE": "02-Jul-2021",
            "TRANS_CODE": "P", "TRANS_SHARES": "100", "TRANS_PRICEPERSHARE": "5",
            "TRANS_ACQUIRED_DISP_CD": "A", "SHRS_OWND_FOLWNG_TRANS": "100",
            "DIRECT_INDIRECT_OWNERSHIP": "D", "NATURE_OF_OWNERSHIP": "",
            "DEEMED_EXECUTION_DATE": "", "VALU_OWND_FOLWNG_TRANS": "500",
        }]),
        "FOOTNOTES.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "FOOTNOTE_ID": "F1", "FOOTNOTE_TXT": "Open-market purchase.",
        }]),
        "NONDERIV_HOLDING.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "NONDERIV_HOLDING_SK": "2", "SECURITY_TITLE": "Common Stock",
            "SHRS_OWND_FOLWNG_TRANS": "100", "DIRECT_INDIRECT_OWNERSHIP": "D",
        }]),
        "DERIV_TRANS.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "DERIV_TRANS_SK": "3", "SECURITY_TITLE": "Option",
            "TRANS_DATE": "02-Jul-2021", "TRANS_CODE": "M", "TRANS_SHARES": "10",
            "TRANS_ACQUIRED_DISP_CD": "A", "SHRS_OWND_FOLWNG_TRANS": "10",
            "DIRECT_INDIRECT_OWNERSHIP": "D",
        }]),
        "DERIV_HOLDING.tsv": pd.DataFrame([{
            "ACCESSION_NUMBER": accession, "DERIV_HOLDING_SK": "4", "SECURITY_TITLE": "Option",
            "SHRS_OWND_FOLWNG_TRANS": "10", "DIRECT_INDIRECT_OWNERSHIP": "D",
        }]),
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            archive.writestr(name, frame.to_csv(sep="\t", index=False))
    return payload.getvalue()


def offline_self_test() -> None:
    """Exercise both normalization paths with a synthetic quarterly ZIP."""
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        archive = base / "synthetic.zip"
        archive.write_bytes(synthetic_zip_bytes())
        result = normalize_archive(archive, base / "normalized", "2021q3")
        assert result["form4_submissions"] == 1 and result["transactions"] == 1
        expected = {"submissions.parquet", "transactions.parquet", "owners.parquet", "footnotes.parquet"}
        assert expected.issubset({path.name for path in (base / "normalized").iterdir()})
        all_result = normalize_all_archive(archive, base / "sec_all", "2021q3")
        assert all_result["tables"]["nonderiv_transactions"] == 1
        assert set(ALL_TABLES).issubset({path.stem for path in (base / "sec_all").glob("*.parquet")})


def main() -> int:
    """Run the acquisition, full-normalization, or offline-test CLI mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int)
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--normalize-all", action="store_true")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=ALL_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.self_test:
        offline_self_test()
        print("PASS SEC synthetic ZIP normalization")
        return 0
    if args.normalize_all:
        raw_dir = args.raw_dir or args.root / "raw"
        print(json.dumps(normalize_all(raw_dir, args.output_root), indent=2))
        return 0
    if args.year is None or args.quarter is None:
        parser.error("--year and --quarter are required unless --self-test is used")
    print(json.dumps(acquire_quarter(args.year, args.quarter, args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
