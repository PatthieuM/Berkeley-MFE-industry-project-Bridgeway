"""Build S1–S3 insider signals from mapped events and daily market data.

The pipeline adds pre-entry market features, assigns prior-years-only size
breakpoints, and returns signal event tables plus breakpoint audit data.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from shared.config import Config, DEFAULT_CONFIG


def next_session(signal_dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Map each signal date to the first strictly later trading session."""
    values = pd.to_datetime(signal_dates, errors="coerce").dt.normalize()
    positions = calendar.searchsorted(values.to_numpy(), side="right")
    result = pd.Series(pd.NaT, index=signal_dates.index, dtype="datetime64[ns]")
    good = values.notna().to_numpy() & (positions < len(calendar))
    result.iloc[np.flatnonzero(good)] = calendar.take(positions[good]).to_numpy()
    return result


def add_market_features(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add lagged cap and 60-session turnover ending before entry."""
    out = events.copy()
    if "entry_date" not in out:
        out["entry_date"] = next_session(out["filing_date"], calendar)
    positions = calendar.get_indexer(pd.DatetimeIndex(out["entry_date"]))
    usable = positions > 0
    usable_events = out.loc[usable].copy()
    usable_positions = positions[usable]
    offsets = np.arange(config.turnover_sessions, dtype=int)
    indices = usable_positions[:, None] - 1 - offsets[None, :]
    if (indices < 0).any():
        raise RuntimeError("Insufficient calendar for turnover lookback")
    windows = pd.DataFrame({
        "event_id": np.repeat(usable_events["event_id"].to_numpy(), config.turnover_sessions),
        "permno": np.repeat(usable_events["permno"].to_numpy(), config.turnover_sessions),
        "feature_date": calendar.to_numpy()[indices.ravel()],
        "offset": np.tile(offsets, len(usable_events)),
    })
    market = daily[["permno", "date", "dlyvol", "shrout", "dlycap"]].copy()
    market = market.rename(columns={"date": "feature_date"}).drop_duplicates(["permno", "feature_date"])
    joined = windows.merge(market, on=["permno", "feature_date"], how="left", validate="many_to_one")
    volume = pd.to_numeric(joined["dlyvol"], errors="coerce")
    shares = pd.to_numeric(joined["shrout"], errors="coerce")
    valid = volume.ge(0) & shares.gt(0) & np.isfinite(volume) & np.isfinite(shares)
    joined["turn_value"] = np.where(valid, volume / shares, np.nan)
    features = joined.groupby("event_id", sort=False).agg(
        turnover=("turn_value", "mean"), turnover_n=("turn_value", "count")
    )
    lagged = joined.loc[joined["offset"].eq(0), ["event_id", "feature_date", "dlycap"]].copy()
    lagged = lagged.rename(columns={"feature_date": "feature_end_date", "dlycap": "lagged_size"}).set_index("event_id")
    out = out.join(features, on="event_id").join(lagged, on="event_id")
    out["turnover"] = out["turnover"].where(out["turnover_n"].ge(config.turnover_min_valid))
    out["lagged_size"] = pd.to_numeric(out["lagged_size"], errors="coerce").where(lambda values: values.gt(0))
    dated = out["feature_end_date"].notna()
    if not out.loc[dated, "feature_end_date"].lt(out.loc[dated, "entry_date"]).all():
        raise RuntimeError("Market feature endpoint is not strictly before entry")
    return out


def assign_small_cap(
    purchase_events: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the exact v6 expanding prior-years-only lower-tercile rule."""
    out = purchase_events.copy()
    out["size_bucket"] = pd.NA
    breakpoint_rows = []
    for year in range(pd.Timestamp(config.study_start).year, pd.Timestamp(config.study_end).year + 1):
        training = out.loc[
            out["filing_year"].lt(year)
            & out["lagged_size"].notna()
            & out["turnover"].notna()
        ]
        test_index = out.index[
            out["filing_year"].eq(year)
            & out["lagged_size"].notna()
            & out["turnover"].notna()
        ]
        if training.empty or len(test_index) == 0:
            continue
        if not training["filing_year"].lt(year).all():
            raise AssertionError("Breakpoint training includes current/future filing year")
        q1, q2 = training["lagged_size"].quantile([1 / 3, 2 / 3]).to_numpy(float)
        values = out.loc[test_index, "lagged_size"]
        out.loc[test_index, "size_bucket"] = np.select(
            [values.le(q1), values.le(q2)], ["small", "mid"], default="large"
        )
        breakpoint_rows.append({
            "year": year,
            "training_min_year": int(training["filing_year"].min()),
            "training_max_year": int(training["filing_year"].max()),
            "training_N": len(training),
            "size_q1": q1,
            "size_q2": q2,
        })
    breakpoints = pd.DataFrame(breakpoint_rows)
    if breakpoints.empty or not breakpoints["training_max_year"].lt(breakpoints["year"]).all():
        raise AssertionError("Small-cap breakpoint leakage")
    out["is_small"] = out["size_bucket"].eq("small")
    return out, breakpoints


def s1_small_cap_purchases(featured_events: pd.DataFrame) -> pd.DataFrame:
    """Return purchase issuer-days assigned to the lagged small-cap bucket."""
    return featured_events.loc[
        featured_events["direction"].eq("P") & featured_events["is_small"].eq(True)
    ].copy()


def s2_all_purchases(featured_events: pd.DataFrame) -> pd.DataFrame:
    """Return all purchase issuer-days."""
    return featured_events.loc[featured_events["direction"].eq("P")].copy()


def s3_sales(featured_events: pd.DataFrame) -> pd.DataFrame:
    """Issuer-days with at least one non-derivative open-market S transaction."""
    return featured_events.loc[featured_events["direction"].eq("S")].copy()


def build_signals(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: Config = DEFAULT_CONFIG,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Return S1–S3 tables, annual size breakpoints, and featured events."""
    featured = add_market_features(events, daily, calendar, config)
    purchases, breakpoints = assign_small_cap(featured.loc[featured["direction"].eq("P")].copy(), config)
    featured = featured.drop(columns=["size_bucket", "is_small"], errors="ignore").merge(
        purchases[["event_id", "size_bucket", "is_small"]], on="event_id", how="left", validate="one_to_one"
    )
    signals = {
        "S1": s1_small_cap_purchases(featured),
        "S2": s2_all_purchases(featured),
        "S3": s3_sales(featured),
    }
    return signals, breakpoints, featured
