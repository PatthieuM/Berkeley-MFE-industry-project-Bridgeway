"""Load the validated biotech universe, SEC events, and daily market series.

No function writes row-level data.  The normalized ``transactions.parquet``
tables contain the non-derivative Form 4 table; derivative holdings are stored
separately upstream and are therefore not eligible. Outputs feed signal construction
and event-return evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from shared.config import Config, DEFAULT_CONFIG


def dates(values) -> pd.Series:
    """Coerce values to normalized, timezone-naive pandas dates."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def numeric_cik(values) -> pd.Series:
    """Coerce SEC CIK values to pandas nullable integers."""
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def normalize_ticker(value: object) -> str:
    """Normalize a ticker to uppercase alphanumeric characters for matching."""
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value).upper().strip() if ch.isalnum())


def load_universe(config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load validated strict-biotech intervals with point-in-time identifiers."""
    manifest = json.loads(config.universe_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("validation", {}).get("status") != "passed":
        raise RuntimeError("Universe manifest is not validated")
    if manifest.get("rule_version") != "crsp-us-common-10-11-v4":
        raise RuntimeError("Unexpected universe rule version")

    interval_cols = ["interval_id", "permno", "permco", "effective_start", "effective_end"]
    biotech = pd.read_parquet(config.universe_path, columns=interval_cols)
    cross_cols = [
        "interval_id", "permno", "permco", "effective_start", "effective_end",
        "ticker", "cik", "cik_source", "cik_is_point_in_time", "cik_match_status",
        "identifier_match_status",
    ]
    cross = pd.read_parquet(config.crosswalk_path, columns=cross_cols)
    if cross["interval_id"].duplicated().any():
        raise RuntimeError("Identifier crosswalk interval_id is not unique")
    universe = biotech.merge(
        cross, on="interval_id", how="left", validate="one_to_one", suffixes=("_bio", "")
    )
    for column in ("permno", "permco"):
        if not universe[f"{column}_bio"].eq(universe[column]).all():
            raise RuntimeError(f"Crosswalk {column} disagrees with biotech interval")
    for column in ("effective_start", "effective_end"):
        left, right = dates(universe[f"{column}_bio"]), dates(universe[column])
        if not left.eq(right).all():
            raise RuntimeError(f"Crosswalk {column} disagrees with biotech interval")
        universe[column] = right
    universe["issuer_cik"] = numeric_cik(universe["cik"])
    universe["permno"] = pd.to_numeric(universe["permno"], errors="coerce").astype("Int64")
    universe["permco"] = pd.to_numeric(universe["permco"], errors="coerce").astype("Int64")
    universe["ticker_norm"] = universe["ticker"].map(normalize_ticker)
    universe["cik_is_point_in_time"] = universe["cik_is_point_in_time"].fillna(False).astype(bool)
    keep = [
        "interval_id", "permno", "permco", "effective_start", "effective_end",
        "ticker", "ticker_norm", "issuer_cik", "cik_source", "cik_is_point_in_time",
        "cik_match_status", "identifier_match_status",
    ]
    return universe[keep].copy()


def _load_raw_issuer_days(
    target_ciks: set[int], config: Config
) -> Tuple[pd.DataFrame, dict]:
    """Read the normalized SEC quarters and build purchase and sale issuer-days.

    Keeps original (non-amendment) Form 4 filings by target issuers, filed
    within the study window, that contain a code-P or code-S transaction and
    have a valid reporting-owner CIK. Collapses them to one row per issuer x
    filing date x direction. Returns the events and an audit dict of counts.
    """
    rows = []
    audit = {
        "quarters_read": 0,
        "original_target_form4_submissions": 0,
        "submissions_with_purchase": 0,
        "submissions_with_sale": 0,
        "submissions_with_valid_owner": 0,
    }
    history_start = pd.Timestamp(config.breakpoint_history_start)
    study_end = pd.Timestamp(config.study_end)
    transaction_end = pd.Timestamp(config.transaction_date_end)
    sub_cols = ["accession", "filing_date", "form", "issuer_cik", "ticker", "is_amendment"]
    tx_cols = ["accession", "transaction_key", "transaction_date", "transaction_code"]
    owner_cols = ["accession", "owner_cik"]

    for sub_path in sorted(config.normalized_sec_dir.glob("year=*/quarter=*/submissions.parquet")):
        year = int(sub_path.parent.parent.name.split("=", 1)[1])
        if year < history_start.year or year > study_end.year:
            continue
        audit["quarters_read"] += 1
        submissions = pd.read_parquet(sub_path, columns=sub_cols)
        submissions["filing_date"] = dates(submissions["filing_date"])
        submissions["issuer_cik"] = numeric_cik(submissions["issuer_cik"])
        submissions = submissions.loc[
            submissions["form"].eq("4")
            & ~submissions["is_amendment"].fillna(False)
            & submissions["issuer_cik"].isin(target_ciks)
            & submissions["filing_date"].between(history_start, study_end)
        ].copy()
        audit["original_target_form4_submissions"] += len(submissions)
        if submissions.empty:
            continue

        accessions = set(submissions["accession"])
        transactions = pd.read_parquet(sub_path.parent / "transactions.parquet", columns=tx_cols)
        transactions = transactions.loc[
            transactions["accession"].isin(accessions)
            & transactions["transaction_code"].isin(["P", "S"])
        ].copy()
        transactions["transaction_date"] = dates(transactions["transaction_date"])
        transactions = transactions.loc[
            transactions["transaction_date"].between(history_start, transaction_end)
        ].drop_duplicates("transaction_key", keep="last")
        if transactions.empty:
            continue

        owners = pd.read_parquet(sub_path.parent / "owners.parquet", columns=owner_cols)
        owners["owner_cik"] = numeric_cik(owners["owner_cik"])
        valid_owner_accessions = set(
            owners.loc[
                owners["accession"].isin(set(transactions["accession"])) & owners["owner_cik"].notna(),
                "accession",
            ]
        )
        audit["submissions_with_valid_owner"] += len(valid_owner_accessions)
        for direction, label in (("P", "purchase"), ("S", "sale")):
            direction_accessions = set(
                transactions.loc[transactions["transaction_code"].eq(direction), "accession"]
            ) & valid_owner_accessions
            audit[f"submissions_with_{label}"] += len(direction_accessions)
            if direction_accessions:
                piece = submissions.loc[submissions["accession"].isin(direction_accessions)].copy()
                piece["direction"] = direction
                rows.append(piece)

    if not rows:
        raise RuntimeError("No eligible P/S issuer-days found")
    filings = pd.concat(rows, ignore_index=True)
    filings["ticker_norm"] = filings["ticker"].map(normalize_ticker)
    events = filings.groupby(
        ["issuer_cik", "filing_date", "direction"], as_index=False, sort=True
    ).agg(
        filing_count=("accession", "nunique"),
        event_tickers=("ticker_norm", lambda values: tuple(sorted({x for x in values if x}))),
    )
    if events.duplicated(["issuer_cik", "filing_date", "direction"]).any():
        raise RuntimeError("SEC events are not unique issuer-date-direction rows")
    audit["issuer_days_before_universe_mapping"] = len(events)
    audit["purchase_issuer_days_before_universe_mapping"] = int(events["direction"].eq("P").sum())
    audit["sale_issuer_days_before_universe_mapping"] = int(events["direction"].eq("S").sum())
    return events, audit


def _map_events_to_universe(
    events: pd.DataFrame, universe: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    """Map issuer-day events to a PERMNO using the universe intervals.

    Candidates are universe intervals for the event's CIK that contain the
    filing date; point-in-time CIK intervals are preferred over current-CIK
    fallbacks. When several share classes remain, the filing's ticker picks one;
    if it cannot, the event is dropped as ``ambiguous_share_class``. Returns the
    mapped events and an audit dict of mapping outcomes.
    """
    base = events.reset_index(drop=True).copy()
    base["_event_id"] = np.arange(len(base), dtype="int64")
    candidates = base[["_event_id", "issuer_cik", "filing_date", "event_tickers"]].merge(
        universe, on="issuer_cik", how="left"
    )
    candidates = candidates.loc[
        candidates["permno"].notna()
        & candidates["filing_date"].between(candidates["effective_start"], candidates["effective_end"])
    ].copy()
    selected, statuses = [], {}
    for event_id, group in candidates.groupby("_event_id", sort=False):
        current = group.copy()
        if current["cik_is_point_in_time"].any():
            current = current.loc[current["cik_is_point_in_time"]]
            source_status = "point_in_time"
        else:
            source_status = "current_fallback"
        resolution = "unique_permno"
        if pd.unique(current["permno"].dropna()).size > 1:
            tickers = set(current["event_tickers"].iloc[0])
            exact = current.loc[current["ticker_norm"].isin(tickers)] if tickers else current.iloc[0:0]
            if pd.unique(exact["permno"].dropna()).size == 1:
                current, resolution = exact, "ticker"
            else:
                statuses[int(event_id)] = "ambiguous_share_class"
                continue
        row = current.sort_values(
            ["effective_start", "effective_end", "permno"],
            ascending=[False, False, True], kind="mergesort",
        ).iloc[0]
        selected.append({
            "_event_id": int(event_id), "permno": int(row["permno"]),
            "permco": int(row["permco"]), "mapping_source": source_status,
            "mapping_resolution": resolution,
        })
        statuses[int(event_id)] = f"mapped_{source_status}"

    mapping = pd.DataFrame(selected)
    mapped = base.merge(mapping, on="_event_id", how="inner", validate="one_to_one")
    mapped["event_id"] = mapped["direction"] + ":" + mapped["_event_id"].astype(str)
    mapped["filing_year"] = mapped["filing_date"].dt.year.astype(int)
    mapped = mapped.drop(columns=["_event_id", "event_tickers"])
    counts = pd.Series(statuses, dtype="object").value_counts().to_dict()
    audit = {str(key): int(value) for key, value in counts.items()}
    audit["unmapped_outside_or_missing_interval"] = int(len(base) - len(statuses))
    audit["mapped_issuer_days"] = len(mapped)
    audit["mapped_purchase_issuer_days"] = int(mapped["direction"].eq("P").sum())
    audit["mapped_sale_issuer_days"] = int(mapped["direction"].eq("S").sum())
    return mapped, audit


def load_insider_events(
    config: Config = DEFAULT_CONFIG,
    universe: Optional[pd.DataFrame] = None,
    return_audit: bool = False,
):
    """Return mapped P/S issuer-days using the frozen v6 construction."""
    universe = load_universe(config) if universe is None else universe
    target_ciks = set(universe["issuer_cik"].dropna().astype(int))
    raw, sec_audit = _load_raw_issuer_days(target_ciks, config)
    mapped, map_audit = _map_events_to_universe(raw, universe)
    mapped = mapped.sort_values(["filing_date", "issuer_cik", "direction"], kind="mergesort").reset_index(drop=True)
    if return_audit:
        return mapped, {**sec_audit, **map_audit}
    return mapped


def _standardize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename CRSP ``dlycaldt`` to ``date``, normalize dates and make price columns numeric."""
    rename = {"dlycaldt": "date"}
    result = frame.rename(columns=rename).copy()
    result["date"] = dates(result["date"])
    for column in ["permno", "dlyret", "dlyclose", "dlyvol", "shrout", "dlycap", "dlyprc"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_prices(
    source: Optional[str] = None,
    config: Config = DEFAULT_CONFIG,
    permnos: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Load and standardize a configured CRSP daily source."""
    source = source or config.price_source
    if source == "universe_full":
        directory = config.universe_price_dir
    elif source == "v6_cache":
        directory = config.v6_market_dir
    else:
        raise ValueError("Unknown price source")
    relevant = None if permnos is None else {int(x) for x in permnos}
    date_column = "dlycaldt" if source == "universe_full" else "date"
    columns = ["permno", date_column, "dlyret", "dlyclose", "dlyvol", "shrout", "dlycap", "dlyprc", "dlydelflg"]
    parts = []
    for path in sorted(directory.glob("daily_*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = _standardize_daily(frame)
        if relevant is not None:
            frame = frame.loc[frame["permno"].isin(relevant)]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        raise RuntimeError(f"No price rows found for source {source}")
    daily = pd.concat(parts, ignore_index=True)
    daily = daily.loc[daily["date"].le(pd.Timestamp(config.study_end))]
    daily = daily.sort_values(["permno", "date"], kind="mergesort").drop_duplicates(["permno", "date"], keep="last")
    return daily.reset_index(drop=True)


def total_return_index(frame: pd.DataFrame, return_col: str, id_col: str) -> pd.DataFrame:
    """Compound valid returns into a reset-safe level index for each identifier."""
    pieces = []
    for security, group in frame.groupby(id_col, sort=False):
        current = group.sort_values("date").copy()
        values = pd.to_numeric(current[return_col], errors="coerce").to_numpy(float)
        level, levels = 1.0, np.full(len(current), np.nan)
        for i, value in enumerate(values):
            if np.isfinite(value) and value > -1:
                level *= 1 + value
                if not np.isfinite(level) or level <= 1e-100 or level >= 1e100:
                    level = 1.0
                levels[i] = level
        current["close"] = levels
        current["security_id"] = security
        pieces.append(current)
    return pd.concat(pieces, ignore_index=True)


def price_levels(daily: pd.DataFrame) -> pd.DataFrame:
    """Convert standardized daily returns to event-evaluation price levels."""
    indexed = total_return_index(daily, "dlyret", "permno")
    indexed["delisted"] = indexed.get("dlydelflg", pd.Series(False, index=indexed.index)).astype("string").eq("Y")
    return indexed[["date", "security_id", "close", "delisted"]]


def load_xbi(config: Config = DEFAULT_CONFIG) -> pd.Series:
    """Load XBI on the common cached exchange calendar as a total-return index."""
    etf = pd.read_parquet(config.v6_market_dir / "etf_daily.parquet", columns=["ticker", "date", "dlyret"])
    etf["date"] = dates(etf["date"])
    xbi = etf.loc[
        etf["ticker"].eq(config.benchmark) & etf["date"].le(pd.Timestamp(config.study_end)),
        ["date", "dlyret"],
    ]
    calendar = pd.DataFrame({"date": dates(pd.read_parquet(config.v6_market_dir / "calendar.parquet")["date"])})
    calendar = calendar.loc[calendar["date"].le(pd.Timestamp(config.study_end))].drop_duplicates().sort_values("date")
    xbi = calendar.merge(xbi, on="date", how="left")
    xbi["benchmark"] = config.benchmark
    return total_return_index(xbi, "dlyret", "benchmark").set_index("date")["close"]
