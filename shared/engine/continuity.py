"""Audit reported ownership continuity across the normalized SEC Form 3/4/5 archive.

The module maps rows to the biotech universe, applies CRSP split adjustments, and writes
row-level Parquet results plus a CSV status summary for the research pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SUBMISSION = Path(__file__).resolve().parents[2]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

from shared.config import DATA_ROOT, DEFAULT_CONFIG
from shared.sec.form4_parser.quality_checks import check_ownership_continuity

SEC_ALL_ROOT = DATA_ROOT / "sec/all_forms"
OUTPUT_ROOT = DATA_ROOT / "outputs/continuity"
SUMMARY_PATH = SUBMISSION / "results/continuity_summary.csv"


def _number(values: pd.Series) -> pd.Series:
    """Convert values to numbers; anything unparseable becomes NaN."""
    return pd.to_numeric(values, errors="coerce")


def map_permno(rows: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time issuer-CIK mapping; ambiguous matches remain missing."""
    base = rows.reset_index(drop=True).copy()
    base["_row"] = np.arange(len(base))
    cross = crosswalk.copy()
    cross["issuer_cik"] = _number(cross["cik"]).astype("Int64")
    cross["effective_start"] = pd.to_datetime(cross["effective_start"])
    cross["effective_end"] = pd.to_datetime(cross["effective_end"])
    candidates = base[["_row", "issuer_cik", "event_date"]].merge(
        cross[["issuer_cik", "permno", "effective_start", "effective_end", "cik_is_point_in_time"]],
        on="issuer_cik", how="left",
    )
    candidates = candidates.loc[
        candidates["event_date"].between(candidates["effective_start"], candidates["effective_end"])
    ]
    candidates["_pit"] = candidates["cik_is_point_in_time"].fillna(False)
    candidates["_has_pit"] = candidates.groupby("_row")["_pit"].transform("any")
    # Prefer historical CIK links whenever an event has any point-in-time candidate.
    candidates = candidates.loc[~candidates["_has_pit"] | candidates["_pit"]]
    unique_count = candidates.groupby("_row")["permno"].transform("nunique")
    selected = (
        candidates.loc[unique_count.eq(1)].sort_values(["_row", "effective_start"])
        .drop_duplicates("_row", keep="last")[["_row", "permno"]]
    )
    return base.merge(selected, on="_row", how="left").drop(columns="_row")


def split_events(price_dir: Path, permnos: set[int]) -> pd.DataFrame:
    """Convert CRSP cumulative share-factor changes to effective split events."""
    parts = []
    for path in sorted(price_dir.glob("daily_*.parquet")):
        frame = pd.read_parquet(path, columns=["permno", "dlycaldt", "dlycumfacshr"])
        frame = frame.loc[frame["permno"].isin(permnos)]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=["permno", "effective_date", "split_factor"])
    daily = pd.concat(parts, ignore_index=True).sort_values(["permno", "dlycaldt"])
    daily["dlycumfacshr"] = _number(daily["dlycumfacshr"])
    daily["previous_factor"] = daily.groupby("permno")["dlycumfacshr"].shift()
    changed = daily.loc[
        daily["previous_factor"].notna() & daily["dlycumfacshr"].notna()
        & ~np.isclose(daily["previous_factor"], daily["dlycumfacshr"])
        & daily["dlycumfacshr"].ne(0)
    ].copy()
    changed["split_factor"] = changed["previous_factor"] / changed["dlycumfacshr"]
    return changed.rename(columns={"dlycaldt": "effective_date"})[["permno", "effective_date", "split_factor"]]


def build_continuity_rows(
    sec_root: Path = SEC_ALL_ROOT,
    crosswalk_path: Path = DEFAULT_CONFIG.crosswalk_path,
    universe_path: Path = DEFAULT_CONFIG.universe_path,
    start_year: int = 2006,
    end_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load complete SEC rows for biotech issuers and return audit input + splits."""
    intervals = set(pd.read_parquet(universe_path, columns=["interval_id"])["interval_id"])
    cross = pd.read_parquet(crosswalk_path)
    cross = cross.loc[cross["interval_id"].isin(intervals)].copy()
    target = set(_number(cross["cik"]).dropna().astype(int))
    frames = []
    for sub_path in sorted(sec_root.glob("year=*/quarter=*/submissions.parquet")):
        year = int(sub_path.parent.parent.name.split("=", 1)[1])
        if not start_year <= year <= end_year:
            continue
        sub = pd.read_parquet(sub_path, columns=[
            "accession", "filing_date", "period_of_report", "document_type",
            "issuercik", "issuername", "issuertradingsymbol",
        ])
        sub["issuer_cik"] = _number(sub["issuercik"]).astype("Int64")
        sub["filing_date"] = pd.to_datetime(sub["filing_date"], errors="coerce")
        sub = sub.loc[sub["issuer_cik"].isin(target)].copy()
        if sub.empty:
            continue
        sub = sub[
            [
                "accession",
                "filing_date",
                "period_of_report",
                "document_type",
                "issuer_cik",
                "issuername",
                "issuertradingsymbol",
            ]
        ]
        sub = sub.rename(
            columns={
                "document_type": "form",
                "issuername": "issuer_name",
                "issuertradingsymbol": "issuer_ticker",
            }
        )
        owners = pd.read_parquet(
            sub_path.parent / "owners.parquet",
            columns=["accession", "rptownercik", "rptownername"],
        )
        owners = owners.loc[owners["accession"].isin(set(sub["accession"]))].copy()
        owners["owner_cik"] = _number(owners["rptownercik"]).astype("Int64")
        owners = owners.rename(columns={"rptownername": "owner_name"})
        owners["owner_count"] = owners.groupby("accession")["owner_cik"].transform("nunique")
        owners["owner_attribution_status"] = np.where(
            owners["owner_count"].eq(1), "single_owner", "joint_owner_unallocated",
        )
        owners = owners[["accession", "owner_cik", "owner_name", "owner_attribution_status"]]
        row_sources = (
            ("nonderiv_transactions.parquet", "transaction", "non_derivative"),
            ("nonderiv_holdings.parquet", "holding", "non_derivative"),
            ("deriv_transactions.parquet", "transaction", "derivative"),
            ("deriv_holdings.parquet", "holding", "derivative"),
        )
        for filename, kind, table in row_sources:
            columns = [
                "accession", "security_title", "shrs_ownd_folwng_trans",
                "direct_indirect_ownership", "nature_of_ownership",
            ]
            if kind == "transaction":
                columns += ["trans_date", "trans_code", "trans_shares", "trans_acquired_disp_cd"]
            if table == "derivative":
                columns += ["conv_exercise_price", "expiration_date", "undlyng_sec_title"]
                columns += ["excercise_date" if kind == "transaction" else "exercise_date"]
            raw = pd.read_parquet(sub_path.parent / filename, columns=columns)
            raw = raw.loc[raw["accession"].isin(set(sub["accession"]))].copy()
            if raw.empty:
                continue
            raw["row_kind"] = kind
            raw["table"] = table
            raw["event_date"] = pd.to_datetime(
                raw["trans_date"] if kind == "transaction" else pd.NaT, errors="coerce",
            )
            merged = raw.merge(sub, on="accession", how="inner", validate="many_to_one")
            merged["event_date"] = merged["event_date"].fillna(merged["period_of_report"]).fillna(merged["filing_date"])
            merged = merged.merge(owners, on="accession", how="left", validate="many_to_many")
            merged = merged.rename(columns={
                "trans_date": "transaction_date", "trans_code": "transaction_code", "trans_shares": "shares",
                "trans_acquired_disp_cd": "acquired_disposed",
                "shrs_ownd_folwng_trans": "shares_owned_following",
                "direct_indirect_ownership": "direct_or_indirect",
                "conv_exercise_price": "conversion_or_exercise_price",
                "excercise_date": "exercise_date",
                "undlyng_sec_title": "underlying_security_title",
            })
            merged["is_amendment"] = merged["form"].astype(str).str.endswith("/A")
            merged["sequence_in_filing"] = np.arange(len(merged)) + 1
            frames.append(merged)
    if not frames:
        raise RuntimeError(f"No biotech continuity rows found under {sec_root}")
    rows = pd.concat(frames, ignore_index=True)
    rows = map_permno(rows, cross)
    splits = split_events(DEFAULT_CONFIG.universe_price_dir, set(rows["permno"].dropna().astype(int)))
    return rows, splits


def audit_continuity(rows: pd.DataFrame, splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the canonical quality check and map it to the research status schema."""
    checked = check_ownership_continuity(rows, split_adjustments=splits)
    source = rows.reset_index().rename(columns={"index": "_source_index"})
    source["_source_index"] += 1
    checked = checked.merge(
        source[["_source_index", "transaction_code", "permno"]],
        left_on="input_row_number", right_on="_source_index", how="left",
    ).drop(columns="_source_index")
    checked["year"] = pd.to_datetime(checked["filing_date"], errors="coerce").dt.year.astype("Int64")
    status = checked["continuity_status"]
    checked["status"] = np.select(
        [status.eq("PASS"), status.str.startswith("BASELINE"), status.str.startswith("FLAG")],
        ["consistent", "first_observation", "inconsistent"], default="not_comparable",
    )
    checked["reason"] = np.select(
        [status.eq("PASS"), status.str.startswith("BASELINE"), status.eq("SKIP_AMENDMENT"),
         status.str.contains("OWNER_ATTRIBUTION"), status.str.startswith("FLAG") & checked["permno"].isna()],
        ["reconciled", "missing_prior", "amendment", "owner_attribution", "split_unmatched"],
        default=np.where(status.str.startswith("FLAG"), "arithmetic_mismatch", checked["continuity_reason"]),
    )
    checked.loc[checked["reason"].eq("split_unmatched"), "status"] = "not_comparable"
    chronological = checked.sort_values(["owner_cik", "issuer_cik", "event_date", "input_row_number"])
    has_prior_issuer_observation = chronological.groupby(["owner_cik", "issuer_cik"], dropna=False).cumcount().gt(0)
    security_change = chronological["status"].eq("first_observation") & has_prior_issuer_observation
    checked.loc[chronological.index[security_change], ["status", "reason"]] = ["not_comparable", "security_change"]
    return checked


def continuity_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize continuity statuses overall, by year, and by transaction code."""
    output = []
    for dimension, columns in (("overall", []), ("year", ["year"]), ("transaction_code", ["transaction_code"])):
        grouped = rows.groupby(columns + ["status"], dropna=False).size().rename("count").reset_index()
        grouped["total"] = grouped.groupby(columns, dropna=False)["count"].transform("sum") if columns else len(rows)
        grouped["percent"] = 100 * grouped["count"] / grouped["total"]
        grouped["dimension"] = dimension
        output.append(grouped)
    return pd.concat(output, ignore_index=True, sort=False)[
        ["dimension", "year", "transaction_code", "status", "count", "total", "percent"]
    ]


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, audit, and persist continuity rows and their aggregate summary."""
    rows, splits = build_continuity_rows()
    audited = audit_continuity(rows, splits)
    summary = continuity_summary(audited)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    audited.to_parquet(OUTPUT_ROOT / "continuity_rows.parquet", index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    return audited, summary


def main() -> int:
    """Run the continuity pipeline and print status counts."""
    audited, _ = run()
    print(audited["status"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
