"""Build version 5 effective-dated CRSP common-stock and biotech tables.

This opt-in builder rejects malformed identifiers, dates and overlapping
classification histories. Historical results continue to use the version 4
builder and saved inputs. Classification histories can contain later revisions.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import inspect
import json
from numbers import Number
import os
import fcntl
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Optional
import uuid

import pandas as pd

if __package__:
    from .wrds_cloud import open_wrds_connection
else:
    from wrds_cloud import open_wrds_connection


RULE_VERSION = "crsp-us-common-10-11-v5"
DEFAULT_START_DATE = date(2003, 6, 30)
DEFAULT_END_DATE = date(2025, 12, 31)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(os.environ.get("BRIDGEWAY_DATA", PROJECT_ROOT.parent / "data")) / "universe_v5"
OUTPUT_PRODUCER = "shared.universe.universe_v5"
OUTPUT_SCHEMA_VERSION = 4

# Keep retrospective outcome fields separate from membership inputs.
RETROSPECTIVE_PREFIXES = ("delisting_", "compustat_deletion_")


def _retrospective_columns(frame: pd.DataFrame) -> list[str]:
    """Return retrospective outcome columns that must not determine membership."""
    return [
        column for column in frame.columns
        if str(column).lower().startswith(RETROSPECTIVE_PREFIXES)
    ]


ELIGIBLE_EXCHANGES = frozenset({"N", "A", "Q"})
EXCHANGE_NAMES = {
    "N": "NYSE",
    "A": "NYSE American/AMEX",
    "Q": "NASDAQ",
}
ELIGIBLE_ISSUER_TYPES = frozenset({"ACOR", "CORP"})

STRICT_BIOTECH_GICS = "35201010"
PHARMA_GICS = "35202010"
LIFE_SCIENCE_TOOLS_GICS = "35203010"
BIOTECH_SIC = 2836
PHARMA_SIC = 2834

SECURITY_REQUIRED = {
    "permno",
    "permco",
    "secinfostartdt",
    "secinfoenddt",
    "primaryexch",
    "securitytype",
    "securitysubtype",
    "sharetype",
    "issuertype",
    "usincflg",
    "conditionaltype",
    "tradingstatusflg",
    "siccd",
    "naics",
    "cusip",
    "ticker",
    "issuernm",
    "securitynm",
    "shareclass",
    "exchangetier",
}

DELIST_REQUIRED = {
    "permno",
    "delistingdt",
    "delactiontype",
    "delstatustype",
    "delreasontype",
    "delpaymenttype",
    "delpermno",
    "delpermco",
    "delret",
}

WRDS_CONTRACT = {
    ("crsp", "stksecurityinfohist"): SECURITY_REQUIRED,
    ("crsp", "stkdlysecuritydata"): {"permno", "dlycaldt"},
    ("crsp", "stkdelists"): DELIST_REQUIRED,
    ("crsp", "ccmxpf_lnkhist"): {
        "gvkey",
        "lpermno",
        "lpermco",
        "linkdt",
        "linkenddt",
        "linkprim",
        "linktype",
    },
    ("crsp", "comphist"): {
        "gvkey",
        "hcik",
        "hconm",
        "hipodate",
        "hdldte",
        "hdlrsn",
        "hchgdt",
        "hchgenddt",
    },
    ("comp", "company"): {"gvkey", "cik", "conm"},
    ("comp", "co_hgic"): {
        "gvkey",
        "indtype",
        "gsubind",
        "indfrom",
        "indthru",
    },
    ("comp", "co_industry"): {
        "gvkey",
        "datadate",
        "sich",
        "consol",
        "popsrc",
    },
}


def _lower_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with lower-case column names."""
    result = frame.copy()
    result.columns = result.columns.astype(str).str.lower()
    return result


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    """Raise ValueError naming any ``required`` column missing from ``frame``."""
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")


def _as_timestamp(value: Any, label: str) -> pd.Timestamp:
    """Parse ``value`` to a date at midnight, raising ValueError if it is not a date."""
    if isinstance(value, Number) or pd.api.types.is_bool(value):
        raise ValueError(f"{label} must be a calendar date, not a number")
    result = pd.to_datetime(value, errors="coerce")
    if pd.isna(result):
        raise ValueError(f"{label} must be a valid date")
    if result.tzinfo is not None:
        raise ValueError(f"{label} must be a timezone-naive calendar date")
    return pd.Timestamp(result).normalize()


def _normalize_text(series: pd.Series) -> pd.Series:
    """Upper-case and strip text; blank strings become NA."""
    return series.astype("string").str.strip().str.upper().replace("", pd.NA)


def _normalize_code(value: Any) -> Optional[str]:
    """Normalize a classification code to upper-case text without a trailing ``.0``."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return re.sub(r"\.0+$", "", text)


def _normalize_integer(value: Any) -> Optional[int]:
    """Return an integer or None for missing input; reject other values."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    if pd.api.types.is_bool(value):
        raise ValueError(f"expected an integer, got {value!r}")
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"expected a finite integer, got {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"expected a finite integer, got {value!r}")
    return int(number)


def _gics_code(value: Any) -> Optional[str]:
    code = _normalize_code(value)
    if code is not None and not re.fullmatch(r"\d{8}", code):
        raise ValueError(f"GICS subindustry must contain eight digits, got {value!r}")
    return code


def _sic_code(value: Any) -> Optional[int]:
    number = _normalize_integer(value)
    if number is not None and not 0 <= number <= 9999:
        raise ValueError(f"SIC must be an integer between 0 and 9999, got {value!r}")
    return number or None


def _identifiers(values: pd.Series, label: str, *, required: bool = False) -> pd.Series:
    """Normalize positive integral identifiers without rounding or truncation."""
    normalized = pd.Series([_normalize_integer(value) for value in values], index=values.index, dtype=object)
    if normalized.dropna().map(lambda value: not 0 < value <= 2**63 - 1).any() or (required and normalized.isna().any()):
        raise ValueError(f"{label} must contain positive integer identifiers")
    return normalized.astype("Int64")


def _dates(values: pd.Series, label: str) -> pd.Series:
    """Parse calendar dates, rejecting nonempty invalid or timezone-aware input."""
    missing = values.isna() | values.astype("string").str.strip().eq("").fillna(False)
    if any(isinstance(value, Number) or pd.api.types.is_bool(value) for value in values.loc[~missing]):
        raise ValueError(f"{label} must contain calendar dates, not numbers")
    parsed = pd.to_datetime(values.mask(missing), errors="coerce", format="mixed")
    if (parsed.isna() & ~missing).any():
        raise ValueError(f"{label} contains an invalid date")
    if not pd.api.types.is_datetime64_dtype(parsed.dtype):
        raise ValueError(f"{label} must contain timezone-naive calendar dates")
    return parsed.dt.normalize()


def _normalize_cik(value: Any) -> Optional[str]:
    """Return a valid CIK zero-padded to 10 digits, or None."""
    code = _normalize_code(value)
    if not code:
        return None
    if not re.fullmatch(r"\d{1,10}", code) or int(code) == 0:
        return None
    return code.zfill(10)


def build_membership_intervals(
    raw: pd.DataFrame,
    start_date: Any,
    end_date: Any,
) -> pd.DataFrame:
    """Normalize and validate eligible CRSP CIZ security-history intervals."""

    frame = _lower_columns(raw)
    _require_columns(frame, SECURITY_REQUIRED, "CRSP security history")
    start = _as_timestamp(start_date, "start_date")
    end = _as_timestamp(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    for column in (
        "primaryexch",
        "securitytype",
        "securitysubtype",
        "sharetype",
        "issuertype",
        "usincflg",
        "conditionaltype",
        "tradingstatusflg",
    ):
        frame[column] = _normalize_text(frame[column])

    frame["secinfostartdt"] = _dates(frame["secinfostartdt"], "security start")
    frame["secinfoenddt"] = _dates(frame["secinfoenddt"], "security end").fillna(end)
    frame["permno"] = _identifiers(frame["permno"], "PERMNO")
    frame["permco"] = _identifiers(frame["permco"], "PERMCO")

    eligible = (
        frame["permno"].notna()
        & frame["permco"].notna()
        & frame["secinfostartdt"].notna()
        & frame["primaryexch"].isin(ELIGIBLE_EXCHANGES)
        & frame["securitytype"].eq("EQTY")
        & frame["securitysubtype"].eq("COM")
        & frame["sharetype"].eq("NS")
        & frame["issuertype"].isin(ELIGIBLE_ISSUER_TYPES)
        & frame["usincflg"].eq("Y")
        & frame["conditionaltype"].eq("RW")
        & frame["tradingstatusflg"].eq("A")
        & frame["secinfostartdt"].le(end)
        & frame["secinfoenddt"].ge(start)
    )
    frame = frame.loc[eligible].copy()
    if frame.duplicated().any():
        raise ValueError("CRSP security history contains duplicate interval rows")

    frame["permno"] = frame["permno"].astype("int64")
    frame["permco"] = frame["permco"].astype("int64")
    frame["effective_start"] = frame["secinfostartdt"].clip(lower=start)
    frame["effective_end"] = frame["secinfoenddt"].clip(upper=end)
    frame["crsp_share_code"] = frame["issuertype"].map({"ACOR": 10, "CORP": 11})
    frame["exchange_name"] = frame["primaryexch"].map(EXCHANGE_NAMES)
    frame["is_major_exchange_common_stock"] = True
    frame["universe_rule_version"] = RULE_VERSION
    frame["source_table"] = "crsp.stksecurityinfohist"

    frame = frame.drop(columns=["secinfostartdt", "secinfoenddt"])
    frame = frame.sort_values(
        ["permno", "effective_start", "effective_end"], kind="mergesort"
    ).reset_index(drop=True)
    frame["source_membership_id"] = (
        frame["permno"].astype(str)
        + ":"
        + frame["effective_start"].dt.strftime("%Y%m%d")
        + ":"
        + frame["effective_end"].dt.strftime("%Y%m%d")
    )

    validate_universe(frame)
    return frame


def _prepare_history(
    frame: Optional[pd.DataFrame],
    value_column: str,
    label: str,
) -> pd.DataFrame:
    """Clean an effective-dated classification history (e.g. GICS or SIC by PERMNO).

    Keep rows with a PERMNO, start date and value. Reject malformed bounds,
    reversed intervals and conflicting overlaps. When the input omits end dates,
    each change lasts until the next change in that PERMNO/GVKEY history. Explicit
    end dates retain their meaning. Merge overlaps with the same value, then sort.
    """
    columns = ["permno", "effective_date", "effective_end", value_column, "gvkey"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = _lower_columns(frame)
    _require_columns(result, {"permno", "effective_date", value_column}, label)
    infer_ends = "effective_end" not in result
    if "gvkey" not in result:
        result["gvkey"] = pd.NA
    if "effective_end" not in result:
        result["effective_end"] = pd.NaT
    result = result[columns].copy()
    result["permno"] = _identifiers(result["permno"], f"{label} PERMNO")
    result["effective_date"] = _dates(result["effective_date"], f"{label} start")
    result["effective_end"] = _dates(result["effective_end"], f"{label} end")
    result[value_column] = result[value_column].map(_normalize_code)
    if value_column == "gics_code" or label.startswith("GICS"):
        result[value_column] = result[value_column].map(_gics_code)
    else:
        result[value_column] = result[value_column].map(
            lambda value: None if (number := _sic_code(value)) is None else str(number)
        )
    result["gvkey"] = result["gvkey"].astype("string").str.strip().replace("", pd.NA)
    result = result.dropna(subset=["permno", "effective_date", value_column])
    if (result["effective_date"] > result["effective_end"]).any():
        raise ValueError(f"{label} has a reversed interval")
    result["permno"] = result["permno"].astype("int64")

    conflicts = (
        result.groupby(["permno", "gvkey", "effective_date"], dropna=False)[value_column]
        .nunique(dropna=True)
        .gt(1)
    )
    if conflicts.any():
        raise ValueError(f"{label} has conflicting values on one effective date")

    result = result.drop_duplicates(
        ["permno", "gvkey", "effective_date", "effective_end", value_column]
    ).sort_values(["permno", "gvkey", "effective_date"], kind="mergesort")
    if infer_ends:
        next_change = result.groupby(["permno", "gvkey"], dropna=False)["effective_date"].shift(-1)
        result["effective_end"] = next_change - pd.Timedelta(days=1)
    merged = []
    for _, group in result.groupby(["permno", "gvkey"], dropna=False, sort=False):
        current = None
        for row in group.to_dict("records"):
            overlaps = current is not None and (
                pd.isna(current["effective_end"])
                or row["effective_date"] <= current["effective_end"]
            )
            if overlaps:
                if row[value_column] != current[value_column]:
                    raise ValueError(f"{label} has conflicting overlapping intervals")
                current["effective_end"] = (
                    pd.NaT if pd.isna(current["effective_end"]) or pd.isna(row["effective_end"])
                    else max(current["effective_end"], row["effective_end"])
                )
            else:
                if current is not None:
                    merged.append(current)
                current = row
        if current is not None:
            merged.append(current)
    return pd.DataFrame(merged, columns=columns).reset_index(drop=True)


def _prepare_crosswalk(
    frame: Optional[pd.DataFrame],
    default_start: Any = "1900-01-01",
    default_end: Any = "2262-04-10",
) -> pd.DataFrame:
    """Clean the CRSP-Compustat link table used to attach GVKEY, CIK and names.

    Missing bounds use ``default_start`` / ``default_end``. Malformed bounds
    and reversed intervals raise ValueError; rows without PERMNO or GVKEY drop.
    """
    columns = [
        "permno",
        "permco",
        "link_start",
        "link_end",
        "gvkey",
        "cik",
        "company_name",
        "cik_source",
        "ipo_date",
        "linkprim",
        "linktype",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = _lower_columns(frame)
    _require_columns(result, {"permno", "link_start", "link_end", "gvkey"}, "crosswalk")
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    result = result[columns].copy()
    result["permno"] = _identifiers(result["permno"], "crosswalk PERMNO")
    result["permco"] = _identifiers(result["permco"], "crosswalk PERMCO")
    result["link_start"] = _dates(result["link_start"], "crosswalk start")
    result["link_end"] = _dates(result["link_end"], "crosswalk end")
    result["link_start"] = result["link_start"].fillna(
        _as_timestamp(default_start, "crosswalk default_start")
    )
    result["link_end"] = result["link_end"].fillna(
        _as_timestamp(default_end, "crosswalk default_end")
    )
    result["gvkey"] = result["gvkey"].astype("string").str.strip().replace("", pd.NA)
    result["cik"] = result["cik"].map(_normalize_cik)
    result["company_name"] = result["company_name"].astype("string").str.strip()
    result["cik_source"] = result["cik_source"].astype("string").str.strip()
    result["ipo_date"] = pd.to_datetime(result["ipo_date"], errors="coerce").dt.normalize()
    result["linkprim"] = _normalize_text(result["linkprim"])
    result["linktype"] = _normalize_text(result["linktype"])
    result = result.dropna(subset=["permno", "link_start", "link_end", "gvkey"])
    if (result["link_start"] > result["link_end"]).any():
        raise ValueError("crosswalk has a reversed interval")
    result["permno"] = result["permno"].astype("int64")
    result["permco"] = result["permco"].astype("Int64")
    return result.drop_duplicates().sort_values(
        ["permno", "link_start", "link_end", "gvkey"], kind="mergesort"
    ).reset_index(drop=True)


def _history_lookup(
    history: pd.DataFrame,
    value_column: str,
) -> dict[
    tuple[int, Optional[str]],
    tuple[list[pd.Timestamp], list[Optional[pd.Timestamp]], list[str]],
]:
    """Index a classification history by (PERMNO, GVKEY) for fast date lookups.

    Each entry holds the sorted start dates, end dates and values.
    """
    lookup: dict[
        tuple[int, Optional[str]],
        tuple[list[pd.Timestamp], list[Optional[pd.Timestamp]], list[str]],
    ] = {}
    if history.empty:
        return lookup
    for (permno, gvkey), group in history.groupby(["permno", "gvkey"], dropna=False, sort=False):
        normalized_gvkey = None if pd.isna(gvkey) else str(gvkey)
        lookup[(int(permno), normalized_gvkey)] = (
            group["effective_date"].tolist(),
            [None if pd.isna(value) else pd.Timestamp(value) for value in group["effective_end"]],
            group[value_column].tolist(),
        )
    return lookup


def _latest_value(
    lookup: Mapping[
        tuple[int, Optional[str]],
        tuple[list[pd.Timestamp], list[Optional[pd.Timestamp]], list[str]],
    ],
    permno: int,
    gvkey: Optional[str],
    when: pd.Timestamp,
) -> Optional[str]:
    """Return the classification value in effect for a security on ``when``.

    Use PERMNO-only history only when the exact (PERMNO, GVKEY) key is absent.
    Returns None before the first record or after the record's end date.
    """
    candidate = lookup.get((permno, gvkey)) or lookup.get((permno, None))
    if not candidate:
        return None
    dates, end_dates, values = candidate
    position = bisect_right(dates, when) - 1
    if position < 0:
        return None
    effective_end = end_dates[position]
    if effective_end is not None and when > effective_end:
        return None
    return values[position]


def _crosswalk_priority(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Sort key for competing links: CRSP historical CIK first, then primary link, then link type."""
    cik_source = row.get("cik_source")
    source = 0 if isinstance(cik_source, str) and cik_source == "crsp.comphist_hcik" else 1
    prim = {"P": 0, "C": 1}.get(_normalize_code(row.get("linkprim")), 9)
    link = {"LC": 0, "LU": 1, "LS": 2}.get(_normalize_code(row.get("linktype")), 9)
    return source, prim, link, str(row.get("gvkey") or "")


def _active_crosswalk(
    records: list[dict[str, Any]],
    when: pd.Timestamp,
) -> tuple[Optional[dict[str, Any]], str]:
    """Pick the crosswalk link active on ``when``.

    Returns ``(link, "MATCHED")``, ``(None, "UNMATCHED")`` when none is active,
    or ``(None, "AMBIGUOUS")`` when equally ranked links point to different
    companies.
    """
    active = [
        row for row in records
        if row["link_start"] <= when <= row["link_end"]
    ]
    if not active:
        return None, "UNMATCHED"
    active.sort(key=_crosswalk_priority)
    best = active[0]
    best_priority = _crosswalk_priority(best)[:3]
    tied = {
        (str(row.get("gvkey")), str(row.get("cik"))) for row in active
        if _crosswalk_priority(row)[:3] == best_priority
    }
    if len(tied) > 1:
        return None, "AMBIGUOUS"
    return best, "MATCHED"


def _classification(
    gics_code: Optional[str],
    crsp_sic: Any,
    compustat_sic: Optional[str],
) -> dict[str, Any]:
    """Classify a security as biotech, pharma or other.

    GICS decides when present (biotech = 35201010, pharma = 35202010). Without
    GICS, SIC is the labelled fallback (2836 biotech, 2834 pharma), taking CRSP
    SIC before Compustat SIC. Returns the codes, the rule used and the flags.
    """
    gics = _gics_code(gics_code)
    crsp = _sic_code(crsp_sic)
    comp = _sic_code(compustat_sic)
    if gics:
        is_biotech = gics == STRICT_BIOTECH_GICS
        is_pharma = gics == PHARMA_GICS
        source = "GICS"
        code = gics
        if is_biotech:
            rule = "GICS_STRICT_BIOTECH_EXACT"
        elif is_pharma:
            rule = "GICS_PHARMA_EXACT"
        else:
            rule = "GICS_OTHER"
    else:
        sic = crsp if crsp not in (None, 0) else comp
        is_biotech = sic == BIOTECH_SIC
        is_pharma = sic == PHARMA_SIC
        source = "CRSP_SIC" if crsp not in (None, 0) else (
            "COMPUSTAT_SIC" if comp not in (None, 0) else "UNCLASSIFIED"
        )
        code = str(sic) if sic is not None else None
        if is_biotech:
            rule = "SIC_2836_FALLBACK"
        elif is_pharma:
            rule = "SIC_2834_FALLBACK"
        elif sic is not None:
            rule = "SIC_OTHER_FALLBACK"
        else:
            rule = "NO_CLASSIFICATION"
    return {
        "gics_code": gics,
        "crsp_sic": crsp,
        "compustat_sic": comp,
        "classification_source": source,
        "classification_code": code,
        "classification_rule": rule,
        "is_biotech": bool(is_biotech),
        "is_pharma": bool(is_pharma),
        "is_biopharma": bool(is_biotech or is_pharma),
        "is_life_science_tools": bool(
            gics == LIFE_SCIENCE_TOOLS_GICS
        ),
    }


def build_classified_intervals(
    membership: pd.DataFrame,
    gics_history: Optional[pd.DataFrame] = None,
    sic_history: Optional[pd.DataFrame] = None,
    crosswalk: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Split membership at identifier/classification changes and classify it."""

    base = _lower_columns(membership)
    validate_universe(base)
    gics = _prepare_history(gics_history, "gics_code", "GICS history")
    sic = _prepare_history(sic_history, "compustat_sic", "SIC history")
    links = _prepare_crosswalk(
        crosswalk,
        base["effective_start"].min(),
        base["effective_end"].max(),
    )
    gics_lookup = _history_lookup(gics, "gics_code")
    sic_lookup = _history_lookup(sic, "compustat_sic")
    links_by_permno = {
        int(permno): group.to_dict("records")
        for permno, group in links.groupby("permno", sort=False)
    }

    gics_dates = {
        int(permno): sorted(
            set(group["effective_date"])
            | {
                pd.Timestamp(value) + pd.Timedelta(days=1)
                for value in group["effective_end"].dropna()
            }
        )
        for permno, group in gics.groupby("permno", sort=False)
    }
    sic_dates = {
        int(permno): sorted(
            set(group["effective_date"])
            | {
                pd.Timestamp(value) + pd.Timedelta(days=1)
                for value in group["effective_end"].dropna()
            }
        )
        for permno, group in sic.groupby("permno", sort=False)
    }

    rows: list[dict[str, Any]] = []
    links_were_supplied = crosswalk is not None and not crosswalk.empty
    for source in base.to_dict("records"):
        permno = int(source["permno"])
        start = pd.Timestamp(source["effective_start"])
        end = pd.Timestamp(source["effective_end"])
        link_records = links_by_permno.get(permno, [])
        boundaries = {start}
        for change_date in gics_dates.get(permno, []):
            if start < change_date <= end:
                boundaries.add(change_date)
        for change_date in sic_dates.get(permno, []):
            if start < change_date <= end:
                boundaries.add(change_date)
        for link in link_records:
            link_start = link["link_start"]
            after_link = link["link_end"] + pd.Timedelta(days=1)
            if start < link_start <= end:
                boundaries.add(link_start)
            if start < after_link <= end:
                boundaries.add(after_link)

        ordered = sorted(boundaries)
        for index, segment_start in enumerate(ordered):
            segment_end = (
                ordered[index + 1] - pd.Timedelta(days=1)
                if index + 1 < len(ordered)
                else end
            )
            link, match_status = _active_crosswalk(link_records, segment_start)
            gvkey = str(link["gvkey"]) if link else None
            permit_compustat = not links_were_supplied or link is not None
            current_gics = (
                _latest_value(gics_lookup, permno, gvkey, segment_start)
                if permit_compustat else None
            )
            current_sic = (
                _latest_value(sic_lookup, permno, gvkey, segment_start)
                if permit_compustat else None
            )
            output = {
                key: value for key, value in source.items()
                if not key.lower().startswith(RETROSPECTIVE_PREFIXES)
            }
            output["effective_start"] = segment_start
            output["effective_end"] = segment_end
            output["gvkey"] = gvkey
            linked_cik = link.get("cik") if link else None
            has_cik = linked_cik is not None and not pd.isna(linked_cik)
            cik_source = link.get("cik_source") if link else None
            is_historical_cik = (
                has_cik
                and cik_source is not None
                and not pd.isna(cik_source)
                and str(cik_source) == "crsp.comphist_hcik"
            )
            is_current_fallback = (
                has_cik
                and cik_source is not None
                and not pd.isna(cik_source)
                and str(cik_source) == "comp.company_current"
            )
            output["cik"] = linked_cik
            output["company_name"] = link.get("company_name") if link else None
            output["cik_source"] = cik_source
            output["cik_is_point_in_time"] = bool(is_historical_cik)
            output["cik_match_status"] = (
                "POINT_IN_TIME"
                if is_historical_cik
                else "CURRENT_FALLBACK"
                if is_current_fallback
                else "UNVERIFIED_SOURCE"
                if has_cik
                else "MISSING"
            )
            output["ipo_date"] = link.get("ipo_date") if link else None
            output["ccm_linkprim"] = link.get("linkprim") if link else None
            output["ccm_linktype"] = link.get("linktype") if link else None
            output["identifier_match_status"] = match_status
            output.update(_classification(current_gics, source.get("siccd"), current_sic))
            rows.append(output)

    result = pd.DataFrame(rows)
    result = _coalesce_adjacent(result)
    result["interval_id"] = (
        result["permno"].astype(str)
        + ":"
        + result["effective_start"].dt.strftime("%Y%m%d")
        + ":"
        + result["effective_end"].dt.strftime("%Y%m%d")
    )
    validate_universe(result)
    return result


def _same_value(left: Any, right: Any) -> bool:
    """Compare two values, treating two missing values as equal."""
    left_missing, right_missing = bool(pd.isna(left)), bool(pd.isna(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(left == right)


def _coalesce_adjacent(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge back-to-back intervals of a security that carry identical attributes."""
    ordered = frame.sort_values(
        ["permno", "effective_start", "effective_end"], kind="mergesort"
    ).reset_index(drop=True)
    compare_columns = [
        column for column in ordered.columns
        if column not in {"effective_start", "effective_end"}
    ]
    output: list[dict[str, Any]] = []
    for row in ordered.to_dict("records"):
        if output:
            previous = output[-1]
            adjacent = (
                previous["permno"] == row["permno"]
                and previous["effective_end"] + pd.Timedelta(days=1)
                == row["effective_start"]
            )
            same = adjacent and all(
                _same_value(previous.get(column), row.get(column))
                for column in compare_columns
            )
            if same:
                previous["effective_end"] = row["effective_end"]
                continue
        output.append(row)
    return pd.DataFrame(output, columns=ordered.columns)


def validate_universe(frame: pd.DataFrame) -> None:
    """Reject invalid identifiers and missing, reversed or overlapping intervals."""

    _require_columns(
        frame,
        {"permno", "effective_start", "effective_end"},
        "universe",
    )
    if frame.empty:
        raise ValueError("universe contains no eligible rows")
    identifiers = _identifiers(frame["permno"], "universe PERMNO", required=True)
    if "permco" in frame:
        _identifiers(frame["permco"], "universe PERMCO", required=True)
    starts = _dates(frame["effective_start"], "universe start")
    ends = _dates(frame["effective_end"], "universe end")
    if starts.isna().any() or ends.isna().any() or starts.gt(ends).any():
        raise ValueError("universe has invalid effective-date intervals")
    if frame.duplicated(["permno", "effective_start", "effective_end"]).any():
        raise ValueError("universe has duplicate PERMNO intervals")
    ordered = frame.assign(permno=identifiers, _start=starts, _end=ends).sort_values(
        ["permno", "_start", "_end"], kind="mergesort"
    )
    prior_end = ordered.groupby("permno", sort=False)["_end"].transform(
        lambda values: values.cummax().shift()
    )
    if ordered["_start"].le(prior_end).fillna(False).any():
        raise ValueError("universe has overlapping intervals for a PERMNO")


def prepare_delisting_history(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize the one-record-per-PERMNO CRSP CIZ delisting table."""

    columns = [
        "permno",
        "delisting_date",
        "delisting_action",
        "delisting_status",
        "delisting_reason",
        "delisting_payment",
        "delisting_linked_permno",
        "delisting_linked_permco",
        "delisting_return",
        "delisting_source",
        "universe_rule_version",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = _lower_columns(frame)
    _require_columns(result, DELIST_REQUIRED, "CRSP delisting history")
    result = result.rename(
        columns={
            "delistingdt": "delisting_date",
            "delactiontype": "delisting_action",
            "delstatustype": "delisting_status",
            "delreasontype": "delisting_reason",
            "delpaymenttype": "delisting_payment",
            "delpermno": "delisting_linked_permno",
            "delpermco": "delisting_linked_permco",
            "delret": "delisting_return",
        }
    )
    result["permno"] = _identifiers(result["permno"], "delisting PERMNO")
    result["delisting_date"] = _dates(result["delisting_date"], "delisting date")
    for column in (
        "delisting_action",
        "delisting_status",
        "delisting_reason",
        "delisting_payment",
    ):
        result[column] = _normalize_text(result[column])
    for column in ("delisting_linked_permno", "delisting_linked_permco"):
        result[column] = _identifiers(result[column].replace(0, pd.NA), column)
    result["delisting_return"] = pd.to_numeric(
        result["delisting_return"], errors="coerce"
    )
    result = result.dropna(subset=["permno", "delisting_date"]).copy()
    result["permno"] = result["permno"].astype("int64")
    if result.duplicated("permno").any():
        raise ValueError("CRSP delisting history has multiple records for one PERMNO")
    result["delisting_source"] = "crsp.stkdelists"
    result["universe_rule_version"] = RULE_VERSION
    return result[columns].sort_values(
        ["permno", "delisting_date"], kind="mergesort"
    ).reset_index(drop=True)


def prepare_compustat_deletion_history(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Keep company-deletion outcomes separate from analytical identifiers.

    These are Compustat research-company outcomes, not CRSP security delistings.
    Link dates below describe source lineage, not availability to a trader.
    """
    columns = [
        "permno", "gvkey", "link_start", "link_end", "compustat_deletion_date",
        "compustat_deletion_reason", "source", "retrospective_only",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    result = _lower_columns(frame)
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    result["permno"] = _identifiers(result["permno"], "company deletion PERMNO", required=True)
    for column in ("compustat_deletion_date", "link_start", "link_end"):
        result[column] = _dates(result[column], column)
    if result["link_start"].gt(result["link_end"]).any():
        raise ValueError("company deletion history has reversed link dates")
    result["compustat_deletion_reason"] = _normalize_text(result["compustat_deletion_reason"])
    result = result.loc[
        result["compustat_deletion_date"].notna()
        | result["compustat_deletion_reason"].notna()
    ].copy()
    result["source"] = "crsp.comphist"
    result["retrospective_only"] = True
    return result[columns].drop_duplicates().reset_index(drop=True)


def _fetch_frame(connection: Any, query: str, params: Any = ()) -> pd.DataFrame:
    """Run a SQL query on an open connection and return the result as a DataFrame."""
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        columns = [description.name for description in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def validate_wrds_contract(connection: Any) -> None:
    """Verify that the connected WRDS schemas expose every required field."""
    query = """
        select table_schema, table_name, column_name
        from information_schema.columns
        where table_schema in ('crsp', 'comp')
    """
    available = _fetch_frame(connection, query)
    for (schema, table), required in WRDS_CONTRACT.items():
        columns = set(
            available.loc[
                available["table_schema"].eq(schema)
                & available["table_name"].eq(table),
                "column_name",
            ]
        )
        missing = required.difference(columns)
        if missing:
            raise RuntimeError(
                f"WRDS contract failed for {schema}.{table}: "
                f"{', '.join(sorted(missing))}"
            )


def latest_crsp_date(connection: Any) -> pd.Timestamp:
    """Return the most recent daily security date available in CRSP."""
    frame = _fetch_frame(
        connection,
        "select max(dlycaldt) as latest_date from crsp.stkdlysecuritydata",
    )
    return _as_timestamp(frame.at[0, "latest_date"], "latest CRSP date")


def extract_security_history(
    connection: Any,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Extract eligible CRSP security-history rows for the requested period."""
    columns = ", ".join(sorted(SECURITY_REQUIRED))
    query = f"""
        select {columns}
        from crsp.stksecurityinfohist
        where secinfostartdt <= %s
          and coalesce(secinfoenddt, %s) >= %s
          and primaryexch in ('N', 'A', 'Q')
          and securitytype = 'EQTY'
          and securitysubtype = 'COM'
          and sharetype = 'NS'
          and issuertype in ('ACOR', 'CORP')
          and usincflg = 'Y'
          and conditionaltype = 'RW'
          and tradingstatusflg = 'A'
        order by permno, secinfostartdt, secinfoenddt
    """
    return _fetch_frame(
        connection,
        query,
        (end_date.date(), end_date.date(), start_date.date()),
    )


def extract_delisting_history(
    connection: Any,
    permnos: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Extract and normalize delistings for selected securities and dates."""
    columns = ", ".join(sorted(DELIST_REQUIRED))
    query = f"""
        select {columns}
        from crsp.stkdelists
        where permno = any(%s)
          and delistingdt between %s and %s
        order by permno, delistingdt
    """
    raw = _fetch_frame(
        connection,
        query,
        (permnos, start_date.date(), end_date.date()),
    )
    return prepare_delisting_history(raw)


def extract_crosswalk(
    connection: Any,
    permnos: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Extract dated CRSP–Compustat identifier candidates for selected PERMNOs."""
    query = """
        with links as (
            select l.*
            from crsp.ccmxpf_lnkhist l
            where l.lpermno = any(%(permnos)s)
              and l.linkprim in ('P', 'C')
              and l.linktype in ('LC', 'LU', 'LS')
              and coalesce(l.linkdt, %(start)s) <= %(end)s
              and coalesce(l.linkenddt, %(end)s) >= %(start)s
        ), fallback as (
            select distinct
                l.lpermno::bigint as permno,
                l.lpermco::bigint as permco,
                greatest(coalesce(l.linkdt, %(start)s), %(start)s)::date as link_start,
                least(coalesce(l.linkenddt, %(end)s), %(end)s)::date as link_end,
                l.gvkey,
                c.cik,
                c.conm as company_name,
                'comp.company_current'::text as cik_source,
                null::date as ipo_date,
                null::date as compustat_deletion_date,
                null::text as compustat_deletion_reason,
                l.linkprim,
                l.linktype
            from links l
            left join comp.company c on c.gvkey = l.gvkey
        ), historical as (
            select distinct
                l.lpermno::bigint as permno,
                l.lpermco::bigint as permco,
                greatest(
                    coalesce(l.linkdt, %(start)s),
                    coalesce(h.hchgdt, %(start)s),
                    %(start)s
                )::date as link_start,
                least(
                    coalesce(l.linkenddt, %(end)s),
                    coalesce(h.hchgenddt, %(end)s),
                    %(end)s
                )::date as link_end,
                l.gvkey,
                h.hcik as cik,
                h.hconm as company_name,
                'crsp.comphist_hcik'::text as cik_source,
                h.hipodate as ipo_date,
                h.hdldte as compustat_deletion_date,
                h.hdlrsn as compustat_deletion_reason,
                l.linkprim,
                l.linktype
            from links l
            join crsp.comphist h on h.gvkey = l.gvkey
             and coalesce(h.hchgdt, %(start)s) <= coalesce(l.linkenddt, %(end)s)
             and coalesce(h.hchgenddt, %(end)s) >= coalesce(l.linkdt, %(start)s)
            where h.hcik is not null
              and coalesce(h.hchgdt, %(start)s) <= %(end)s
              and coalesce(h.hchgenddt, %(end)s) >= %(start)s
        )
        select * from fallback
        union all
        select * from historical
        order by permno, link_start, link_end, gvkey, cik_source
    """
    start = start_date.date()
    end = end_date.date()
    return _fetch_frame(connection, query, {"permnos": permnos, "start": start, "end": end})


def extract_gics_history(
    connection: Any,
    permnos: list[int],
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Extract effective-dated GICS history for selected securities."""
    query = """
        select distinct
            l.lpermno::bigint as permno,
            l.gvkey,
            greatest(h.indfrom, coalesce(l.linkdt, h.indfrom))::date
                as effective_date,
            least(
                coalesce(h.indthru, %s),
                coalesce(l.linkenddt, %s),
                %s
            )::date as effective_end,
            trim(h.gsubind::text) as gics_code
        from comp.co_hgic h
        join crsp.ccmxpf_lnkhist l on l.gvkey = h.gvkey
        where l.lpermno = any(%s)
          and h.indtype = 'GICS'
          and h.gsubind is not null
          and h.indfrom <= %s
          and coalesce(h.indthru, %s) >= coalesce(l.linkdt, h.indfrom)
          and coalesce(l.linkenddt, %s) >= h.indfrom
          and l.linkprim in ('P', 'C')
          and l.linktype in ('LC', 'LU', 'LS')
        order by permno, l.gvkey, effective_date, effective_end
    """
    end = end_date.date()
    return _fetch_frame(
        connection,
        query,
        (end, end, end, permnos, end, end, end),
    )


def extract_sic_history(
    connection: Any,
    permnos: list[int],
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Extract monthly effective Compustat SIC changes for selected securities."""
    query = """
        with ranked as (
            select
                l.lpermno::bigint as permno,
                l.gvkey,
                (date_trunc('month', c.datadate) + interval '1 month')::date
                    as effective_date,
                c.sich::text as compustat_sic,
                row_number() over (
                    partition by l.lpermno, l.gvkey, c.datadate
                    order by case l.linkprim when 'P' then 0 else 1 end,
                             case l.linktype when 'LC' then 0 when 'LU' then 1 else 2 end
                ) as choice
            from comp.co_industry c
            join crsp.ccmxpf_lnkhist l on l.gvkey = c.gvkey
            where l.lpermno = any(%s)
              and c.consol = 'C'
              and c.popsrc = 'D'
              and c.sich is not null
              and c.datadate <= %s
              and c.datadate between coalesce(l.linkdt, date '1900-01-01')
                                     and coalesce(l.linkenddt, %s)
              and l.linkprim in ('P', 'C')
              and l.linktype in ('LC', 'LU', 'LS')
        ), chosen as (
            select permno, gvkey, effective_date, compustat_sic
            from ranked where choice = 1
        ), changes as (
            select *, lag(compustat_sic) over (
                partition by permno, gvkey order by effective_date
            ) as prior_code
            from chosen
        )
        select permno, gvkey, effective_date, compustat_sic
        from changes
        where prior_code is distinct from compustat_sic
        order by permno, gvkey, effective_date
    """
    return _fetch_frame(connection, query, (permnos, end_date.date(), end_date.date()))


def build_classification_history(
    gics_history: pd.DataFrame,
    sic_history: pd.DataFrame,
) -> pd.DataFrame:
    """Combine normalized GICS and SIC changes into one classification history."""
    gics = _prepare_history(gics_history, "gics_code", "GICS history").rename(
        columns={"gics_code": "classification_code"}
    )
    gics["classification_system"] = "GICS"
    sic = _prepare_history(sic_history, "compustat_sic", "SIC history").rename(
        columns={"compustat_sic": "classification_code"}
    )
    sic["classification_system"] = "COMPUSTAT_SIC"
    result = pd.concat([gics, sic], ignore_index=True).sort_values(
        ["permno", "effective_date", "classification_system"], kind="mergesort"
    ).reset_index(drop=True)
    applied = []
    for row in result.to_dict("records"):
        if row["classification_system"] == "GICS":
            applied.append(_classification(row["classification_code"], None, None))
        else:
            applied.append(_classification(None, None, row["classification_code"]))
    if applied:
        mapped = pd.DataFrame(applied)
        for column in (
            "classification_rule",
            "is_biotech",
            "is_pharma",
            "is_biopharma",
            "is_life_science_tools",
        ):
            result[column] = mapped[column]
    result["universe_rule_version"] = RULE_VERSION
    return result


def validate_pipeline_outputs(outputs: Mapping[str, pd.DataFrame]) -> list[str]:
    """Validate cross-output invariants and return the recorded check names."""

    required = {
        "source_membership_intervals",
        "master_universe_intervals",
        "biotech_universe_intervals",
        "biopharma_universe_intervals",
        "identifier_crosswalk",
        "identifier_link_candidates",
        "classification_history",
        "delisting_history",
        "compustat_deletion_history",
    }
    missing = required.difference(outputs)
    if missing:
        raise ValueError(f"pipeline outputs are missing: {', '.join(sorted(missing))}")

    master = outputs["master_universe_intervals"]
    membership = outputs["source_membership_intervals"]
    biotech = outputs["biotech_universe_intervals"]
    biopharma = outputs["biopharma_universe_intervals"]
    crosswalk = outputs["identifier_crosswalk"]
    validate_universe(master)
    validate_universe(membership)
    _require_columns(
        master,
        {"interval_id", "source_membership_id"},
        "master universe",
    )
    if master["interval_id"].astype("string").str.strip().fillna("").eq("").any() or master["interval_id"].duplicated().any():
        raise ValueError("master universe interval_id is not unique and non-missing")
    _require_columns(membership, {"source_membership_id"}, "source membership")
    if membership["source_membership_id"].astype("string").str.strip().fillna("").eq("").any() or membership["source_membership_id"].duplicated().any():
        raise ValueError("source membership identifiers must be unique and non-missing")
    if master["source_membership_id"].astype("string").str.strip().fillna("").eq("").any():
        raise ValueError("master universe has missing source lineage")
    _require_columns(master, membership.columns, "master source attributes")
    if set(master["source_membership_id"]) != set(membership["source_membership_id"]):
        raise ValueError("master universe does not cover every source membership")
    lineage = master.merge(
        membership,
        on="source_membership_id", how="left", validate="many_to_one", suffixes=("", "_source"),
    )
    for column in set(membership.columns) - {"source_membership_id", "effective_start", "effective_end"}:
        same = lineage[column].eq(lineage[column + "_source"]) | (
            lineage[column].isna() & lineage[column + "_source"].isna()
        )
        if not same.fillna(False).all():
            raise ValueError(f"master source attribute differs: {column}")
    for column in ("effective_start", "effective_end", "effective_start_source", "effective_end_source"):
        lineage[column] = _dates(lineage[column], column)
    if not (
        lineage["effective_start"].ge(lineage["effective_start_source"])
        & lineage["effective_end"].le(lineage["effective_end_source"])
    ).all():
        raise ValueError("master intervals do not match their source membership")
    for _, group in lineage.groupby("source_membership_id", sort=False):
        ordered = group.sort_values("effective_start").reset_index(drop=True)
        complete = (
            ordered.iloc[0]["effective_start"] == ordered.iloc[0]["effective_start_source"]
            and ordered.iloc[-1]["effective_end"] == ordered.iloc[-1]["effective_end_source"]
        )
        adjacent = ordered["effective_start"].iloc[1:].reset_index(drop=True).eq(
            ordered["effective_end"].iloc[:-1].reset_index(drop=True) + pd.Timedelta(days=1)
        ).all()
        if not complete or not adjacent:
            raise ValueError("master intervals do not completely cover their source membership")
    for label, frame in (
        ("source membership", membership),
        ("master", master), ("biotech", biotech), ("biopharma", biopharma),
        ("identifier_crosswalk", crosswalk),
    ):
        leaked_columns = _retrospective_columns(frame)
        if leaked_columns:
            raise ValueError(
                f"retrospective fields leaked into {label}: "
                + ", ".join(sorted(leaked_columns))
            )
    for label, frame in (("biotech", biotech), ("biopharma", biopharma)):
        if not frame.empty:
            validate_universe(frame)
        flag = "is_biotech" if label == "biotech" else "is_biopharma"
        if flag not in frame or not frame[flag].fillna(False).all():
            raise ValueError(f"{label} output contains an ineligible classification")

    filter_checks = {
        "primaryexch": master["primaryexch"].isin(ELIGIBLE_EXCHANGES),
        "securitytype": master["securitytype"].eq("EQTY"),
        "securitysubtype": master["securitysubtype"].eq("COM"),
        "sharetype": master["sharetype"].eq("NS"),
        "issuertype": master["issuertype"].isin(ELIGIBLE_ISSUER_TYPES),
        "usincflg": master["usincflg"].eq("Y"),
        "conditionaltype": master["conditionaltype"].eq("RW"),
        "tradingstatusflg": master["tradingstatusflg"].eq("A"),
    }
    failed_filters = [name for name, passed in filter_checks.items() if not passed.all()]
    if failed_filters:
        raise ValueError(
            f"master universe violates eligibility fields: {', '.join(failed_filters)}"
        )

    keys = ["permno", "effective_start", "effective_end"]
    master_keys = set(map(tuple, master[keys].itertuples(index=False, name=None)))
    for label, frame in (("biotech", biotech), ("biopharma", biopharma)):
        view_keys = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
        expected = master.loc[master["is_" + label].eq(True)]
        expected_keys = set(map(tuple, expected[keys].itertuples(index=False, name=None)))
        if view_keys != expected_keys or set(frame.columns) != set(master.columns):
            raise ValueError(f"{label} output does not equal its master filter")
        try:
            pd.testing.assert_frame_equal(
                frame[master.columns].sort_values(keys).reset_index(drop=True),
                expected.sort_values(keys).reset_index(drop=True), check_dtype=False,
            )
        except AssertionError as exc:
            raise ValueError(f"{label} attributes differ from master") from exc
    crosswalk_keys = set(map(tuple, crosswalk[keys].itertuples(index=False, name=None)))
    if crosswalk.duplicated(keys).any():
        raise ValueError("identifier crosswalk has duplicate interval keys")
    if crosswalk_keys != master_keys:
        raise ValueError("identifier crosswalk interval keys do not match the master universe")
    _require_columns(
        crosswalk,
        {"permco", "interval_id", "source_membership_id", "cik", "gvkey", "cik_source", "cik_is_point_in_time", "cik_match_status"},
        "identifier crosswalk",
    )
    if not set(crosswalk.columns).issubset(master.columns):
        raise ValueError("identifier crosswalk has fields absent from master")
    try:
        pd.testing.assert_frame_equal(
            crosswalk.sort_values(keys).reset_index(drop=True),
            master[crosswalk.columns].sort_values(keys).reset_index(drop=True), check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError("identifier crosswalk values differ from master") from exc

    history = outputs["classification_history"]
    _require_columns(history, {"permno", "gvkey", "effective_date", "effective_end", "classification_code", "classification_system"}, "classification history")
    if not history["classification_system"].isin(["GICS", "COMPUSTAT_SIC"]).all():
        raise ValueError("classification history has an unknown classification system")
    for label, group in history.groupby("classification_system"):
        cleaned = _prepare_history(group, "classification_code", label)
        if len(cleaned) != len(group):
            raise ValueError("classification history is incomplete or not normalized")
    candidates = outputs["identifier_link_candidates"]
    cleaned_candidates = _prepare_crosswalk(candidates)
    if len(cleaned_candidates) != len(candidates):
        raise ValueError("identifier link candidates are incomplete or duplicated")
    for label, table in (("classification history", history), ("identifier link candidates", candidates)):
        if not set(table.get("permno", [])).issubset(set(master["permno"])):
            raise ValueError(f"{label} contains a PERMNO outside the master universe")

    delist = outputs["delisting_history"]
    _require_columns(delist, prepare_delisting_history(None).columns, "delisting history")
    _identifiers(delist["permno"], "delisting PERMNO", required=True)
    if _dates(delist["delisting_date"], "delisting date").isna().any():
        raise ValueError("delisting history has a missing date")
    for column in ("delisting_linked_permno", "delisting_linked_permco"):
        _identifiers(delist[column], column)
    if not delist.empty and delist.duplicated("permno").any():
        raise ValueError("delisting output has duplicate PERMNO rows")
    if not set(delist.get("permno", [])).issubset(set(master["permno"])):
        raise ValueError("delisting output contains a PERMNO outside the master universe")
    deletion = outputs["compustat_deletion_history"]
    _require_columns(deletion, prepare_compustat_deletion_history(None).columns, "company deletion history")
    _identifiers(deletion["permno"], "company deletion PERMNO", required=True)
    deletion_dates = _dates(deletion["compustat_deletion_date"], "company deletion date")
    starts = _dates(deletion["link_start"], "company deletion link start")
    ends = _dates(deletion["link_end"], "company deletion link end")
    if starts.gt(ends).any():
        raise ValueError("company deletion history has reversed link dates")
    if (deletion_dates.isna() & _normalize_text(deletion["compustat_deletion_reason"]).isna()).any():
        raise ValueError("company deletion history has no deletion date or reason")
    if not deletion["retrospective_only"].map(lambda value: pd.api.types.is_bool(value) and bool(value)).all():
        raise ValueError("company deletion outcomes must be marked retrospective_only")
    if not set(deletion.get("permno", [])).issubset(set(master["permno"])):
        raise ValueError("company deletion output contains a PERMNO outside the master universe")
    return [
        "nonempty_master",
        "valid_nonoverlapping_intervals",
        "unique_final_interval_ids_with_source_lineage",
        "source_attributes_and_complete_interval_coverage",
        "major_exchange_us_common_stock_filters",
        "strict_and_broad_view_flags",
        "views_equal_master_filters",
        "identifier_crosswalk_referential_integrity",
        "classification_history_and_link_candidates_valid",
        "one_delisting_record_per_permno",
        "retrospective_delisting_outcomes_kept_separate",
        "retrospective_company_deletion_outcomes_kept_separate",
    ]


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_destination(output_dir: Path, overwrite: bool) -> Path:
    """Allow a new external path or replacement of this writer's own pointer."""
    requested = Path(output_dir).expanduser().absolute()
    destination = requested.parent.resolve() / requested.name
    if destination == PROJECT_ROOT or destination.is_relative_to(PROJECT_ROOT):
        raise ValueError("licensed universe outputs must be outside the repository")
    exists = destination.exists() or destination.is_symlink()
    if exists and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    if exists:
        if not destination.is_symlink():
            raise ValueError("version 5 never replaces existing directories; choose a new output path")
        versions = destination.parent / f".{destination.name}.versions"
        if not destination.resolve().is_relative_to(versions.resolve()):
            raise ValueError("output pointer does not refer to this writer's version directory")
        try:
            manifest = json.loads((destination / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("output pointer has no valid manifest") from exc
        if (manifest.get("producer") != OUTPUT_PRODUCER
                or manifest.get("output_schema_version") != OUTPUT_SCHEMA_VERSION
                or manifest.get("rule_version") != RULE_VERSION):
            raise ValueError("output pointer belongs to a different universe version")
    return destination


def write_outputs_atomic(
    outputs: Mapping[str, pd.DataFrame],
    output_dir: Path,
    manifest: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Publish an immutable version and atomically replace its public symlink.

    The public directory path and filenames stay stable for consumers. Previous
    versions are retained. Existing physical directories are never replaced.
    """
    destination = _safe_output_destination(output_dir, overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.publish.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        destination = _safe_output_destination(output_dir, overwrite)
        versions = destination.parent / f".{destination.name}.versions"
        if versions.is_symlink():
            raise ValueError("the universe versions directory cannot be a symlink")
        versions.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=versions))
        version = versions / uuid.uuid4().hex
        pointer = destination.parent / f".{destination.name}.pointer-{uuid.uuid4().hex}"
        published = False
        try:
            files = {}
            for stem, frame in outputs.items():
                if not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
                    raise ValueError(f"unsafe output stem: {stem}")
                parquet, csv = staging / f"{stem}.parquet", staging / f"{stem}.csv"
                frame.to_parquet(parquet, index=False, compression="zstd")
                frame.to_csv(csv, index=False)
                for path in (parquet, csv):
                    files[path.name] = {"rows": len(frame), "sha256": _sha256(path)}
            complete = dict(manifest, producer=OUTPUT_PRODUCER,
                            output_schema_version=OUTPUT_SCHEMA_VERSION,
                            rule_version=RULE_VERSION, files=files)
            (staging / "manifest.json").write_text(
                json.dumps(complete, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            staging.rename(version)
            pointer.symlink_to(os.path.relpath(version, destination.parent), target_is_directory=True)
            os.replace(pointer, destination)
            published = True
            return destination
        finally:
            if pointer.is_symlink():
                pointer.unlink()
            if staging.exists():
                shutil.rmtree(staging)
            if not published and version.exists():
                shutil.rmtree(version)


def run_pipeline(
    connection: Any,
    start_date: Any = DEFAULT_START_DATE,
    end_date: Any = DEFAULT_END_DATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract, validate, classify, and atomically publish the universe."""

    destination = _safe_output_destination(Path(output_dir), overwrite)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {destination}")
    validate_wrds_contract(connection)
    start = _as_timestamp(start_date, "start_date")
    available_end = latest_crsp_date(connection)
    requested_end = _as_timestamp(
        DEFAULT_END_DATE if end_date is None else end_date, "end_date"
    )
    end = min(requested_end, available_end)
    if end < start:
        raise ValueError("CRSP coverage ends before the requested start date")

    raw = extract_security_history(connection, start, end)
    membership = build_membership_intervals(raw, start, end)
    permnos = sorted(membership["permno"].unique().tolist())
    crosswalk = extract_crosswalk(connection, permnos, start, end)
    company_deletions = prepare_compustat_deletion_history(crosswalk)
    gics = extract_gics_history(connection, permnos, end)
    sic = extract_sic_history(connection, permnos, end)
    classified = build_classified_intervals(membership, gics, sic, crosswalk)
    delisting_history = extract_delisting_history(connection, permnos, start, end)
    biotech = classified.loc[classified["is_biotech"]].reset_index(drop=True)
    biopharma = classified.loc[classified["is_biopharma"]].reset_index(drop=True)
    classification_history = build_classification_history(gics, sic)
    identifier_columns = [
        "permno",
        "permco",
        "interval_id",
        "source_membership_id",
        "effective_start",
        "effective_end",
        "ticker",
        "cusip",
        "issuernm",
        "securitynm",
        "shareclass",
        "primaryexch",
        "exchangetier",
        "gvkey",
        "cik",
        "company_name",
        "cik_source",
        "cik_is_point_in_time",
        "cik_match_status",
        "ipo_date",
        "ccm_linkprim",
        "ccm_linktype",
        "identifier_match_status",
        "universe_rule_version",
    ]
    identifier_crosswalk = classified[identifier_columns].drop_duplicates().reset_index(drop=True)

    outputs = {
        "source_membership_intervals": membership,
        "master_universe_intervals": classified,
        "biotech_universe_intervals": biotech,
        "biopharma_universe_intervals": biopharma,
        "identifier_crosswalk": identifier_crosswalk,
        "identifier_link_candidates": _prepare_crosswalk(crosswalk),
        "classification_history": classification_history,
        "delisting_history": delisting_history,
        "compustat_deletion_history": company_deletions,
    }
    passed_checks = validate_pipeline_outputs(outputs)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "rule_version": RULE_VERSION,
        "generated_at_utc": generated_at,
        "source_code_sha256": _sha256(Path(__file__)),
        "query_function_sha256": {
            function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for function in (
                extract_security_history, extract_crosswalk, extract_gics_history,
                extract_sic_history, extract_delisting_history,
            )
        },
        "output_dir": str(destination),
        "requested_start": start.date().isoformat(),
        "requested_end": requested_end.date().isoformat(),
        "crsp_available_end": available_end.date().isoformat(),
        "effective_end": end.date().isoformat(),
        "sources": [
            "crsp.stksecurityinfohist",
            "crsp.stkdlysecuritydata",
            "crsp.stkdelists",
            "crsp.ccmxpf_lnkhist",
            "crsp.comphist",
            "comp.company",
            "comp.co_hgic",
            "comp.co_industry",
        ],
        "eligibility": {
            "legacy_share_codes": [10, 11],
            "primary_exchanges": sorted(ELIGIBLE_EXCHANGES),
            "includes_nasdaq_capital_market": True,
            "market_cap_cutoff": None,
        },
        "classification": {
            "strict_biotech_gics_subindustry": STRICT_BIOTECH_GICS,
            "pharma_gics_subindustry": PHARMA_GICS,
            "sic_fallback": {"biotech": BIOTECH_SIC, "pharma": PHARMA_SIC},
            "gics_history_is_effective_dated_not_vintage_as_known": True,
        },
        "identifier_provenance": {
            "current_compustat_cik_is_historical_fallback_only": True,
            "point_in_time_cik_source": "crsp.comphist_hcik",
            "current_cik_fallback_quarantined_by_identifier_mapper": False,
            "ambiguous_ccm_links_are_not_selected": True,
        },
        "delisting_provenance": {
            "output": "delisting_history",
            "joined_to_membership_intervals": False,
            "reason": "retrospective outcome data is kept separate to prevent look-ahead",
        },
        "compustat_deletion_provenance": {
            "output": "compustat_deletion_history",
            "joined_to_membership_intervals": False,
            "retrospective_only": True,
        },
        "counts": {
            "master_intervals": int(len(classified)),
            "strict_biotech_intervals": int(len(biotech)),
            "biopharma_intervals": int(len(biopharma)),
            "distinct_permnos": int(classified["permno"].nunique()),
            "distinct_permcos": int(classified["permco"].nunique()),
            "cik_matched_intervals": int(classified["cik"].notna().sum()),
            "point_in_time_cik_intervals": int(
                classified["cik_is_point_in_time"].fillna(False).sum()
            ),
            "current_cik_fallback_intervals": int(
                classified["cik_match_status"].eq("CURRENT_FALLBACK").sum()
            ),
            "delisting_records": int(len(delisting_history)),
            "compustat_deletion_records": int(len(company_deletions)),
        },
        "validation": {"status": "passed", "checks": passed_checks},
    }
    published = write_outputs_atomic(outputs, destination, manifest, overwrite)
    return json.loads((published / "manifest.json").read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the universe builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument(
        "--end", default=DEFAULT_END_DATE.isoformat(),
        help="Requested end date (default 2025-12-31); capped at latest CRSP date",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--username", help="WRDS username; inferred from ~/.pgpass when unique")
    parser.add_argument("--connector", choices=("legacy", "v2"), default="legacy",
                        help="Use the existing WRDS login helper or opt in to verified-host connector v2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print requested settings without connecting or writing",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the universe pipeline from command-line arguments."""
    args = _parser().parse_args(argv)
    start = _as_timestamp(args.start, "start")
    end = _as_timestamp(args.end, "end")
    if end < start:
        raise ValueError("end must be on or after start")
    _safe_output_destination(args.output_dir, args.overwrite)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rule_version": RULE_VERSION,
                    "start": args.start,
                    "end": args.end,
                    "output_dir": str(args.output_dir),
                    "connector": args.connector,
                    "share_codes": [10, 11],
                    "primary_exchanges": sorted(ELIGIBLE_EXCHANGES),
                    "market_cap_cutoff": None,
                },
                indent=2,
            )
        )
        return 0
    connect = open_wrds_connection
    if args.connector == "v2":
        if __package__:
            from .wrds_cloud_v2 import open_wrds_connection as connect
        else:
            from wrds_cloud_v2 import open_wrds_connection as connect
    with connect(args.username) as connection:
        manifest = run_pipeline(
            connection,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
