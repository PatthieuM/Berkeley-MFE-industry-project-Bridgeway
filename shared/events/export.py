"""Export stable identifiers for research events without changing result tables.

Signal event tables and the identifier crosswalk are enriched on filing date and
written as one Parquet file per S1-S4 signal for downstream delivery.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SUBMISSION = Path(__file__).resolve().parents[2]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

from shared.config import DATA_ROOT, DEFAULT_CONFIG

OUTPUT_ROOT = DATA_ROOT / "outputs/events"
EXPORT_COLUMNS = [
    "event_id",
    "signal",
    "permno",
    "permco",
    "cusip",
    "ticker",
    "issuer_cik",
    "filing_date",
    "entry_date",
]


def _enrich(frame: pd.DataFrame, signal: str, crosswalk: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Add stable identifiers (PERMCO, CUSIP, ticker) and the entry date to one signal's events.

    Identifiers come from the crosswalk interval containing the filing date
    (latest-starting one if several). A missing entry date is set to the first
    session strictly after the filing date. Returns ``EXPORT_COLUMNS``.
    """
    x = frame.copy().reset_index(drop=True)
    x["filing_date"] = pd.to_datetime(x["filing_date"], errors="coerce").dt.normalize()
    if "entry_date" not in x:
        positions = calendar.searchsorted(x["filing_date"].to_numpy(dtype="datetime64[ns]"), side="right")
        x["entry_date"] = [calendar[pos] if pos < len(calendar) else pd.NaT for pos in positions]
    x["_row"] = np.arange(len(x))
    cross = crosswalk.copy()
    cross["effective_start"] = pd.to_datetime(cross["effective_start"])
    cross["effective_end"] = pd.to_datetime(cross["effective_end"])
    candidates = x[["_row", "permno", "filing_date"]].merge(
        cross[["permno", "permco", "cusip", "ticker", "effective_start", "effective_end"]],
        on="permno", how="left", suffixes=("", "_cross"),
    )
    candidates = candidates.loc[
        candidates["filing_date"].between(candidates["effective_start"], candidates["effective_end"])
    ]
    identifiers = candidates.sort_values(["_row", "effective_start"]).drop_duplicates("_row", keep="last")
    identifiers = identifiers[["_row", "permco", "cusip", "ticker"]]
    for column in ("permco", "ticker"):
        x = x.drop(columns=column, errors="ignore")
    x = x.merge(identifiers, on="_row", how="left", validate="one_to_one").drop(columns="_row")
    x["signal"] = signal
    return x.reindex(columns=EXPORT_COLUMNS)


def export_events(
    signals: dict[str, pd.DataFrame], s4_events: pd.DataFrame,
    calendar: pd.DatetimeIndex, crosswalk_path: Path = DEFAULT_CONFIG.crosswalk_path,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, int]:
    """Write S1-S4 Parquet files with stable security/company identifiers."""
    crosswalk = pd.read_parquet(crosswalk_path)
    tables = {**signals, "S4": s4_events}
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for signal in ("S1", "S2", "S3", "S4"):
        frame = _enrich(tables[signal], signal, crosswalk, calendar)
        frame.to_parquet(output_root / f"{signal.lower()}_events.parquet", index=False)
        counts[signal] = len(frame)
    return counts
