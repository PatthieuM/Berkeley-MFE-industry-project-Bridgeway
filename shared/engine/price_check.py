"""Compare reported Form 4 trade prices with CRSP price ranges, for every universe trade.

Covers all original Form 4 purchases (code P) and sales (code S) of universe
issuers, 2006-2025, one row per transaction and reporting owner. A reported price
passes the daily check when it lies within the CRSP low-high of the transaction
date (plus or minus 1%), and the weekly check when it lies within the lowest low
and highest high of the five sessions centred on that date. A transaction dated
on a non-trading day is matched to the latest session within the previous four
calendar days.

This is an audit: it writes ``results/price_check_summary.csv`` and changes no
signal sample. The flags used in the signal robustness tables stay in
``shared/engine/sanity.py``.

Run:
    python -m shared.engine.price_check
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import Config, DEFAULT_CONFIG
from shared.engine.sanity import _map_transaction_permno, load_daily_ranges, load_transactions
from shared.events.events import load_universe

OUTPUT = Path(__file__).resolve().parents[2] / "results" / "price_check_summary.csv"
TOLERANCE = 0.01
WEEK_HALF_WIDTH = 2  # sessions on each side of the transaction date


def session_ranges(daily: pd.DataFrame) -> pd.DataFrame:
    """Return daily and five-session low-high ranges per PERMNO and session."""
    d = daily[["permno", "transaction_date", "dlylow", "dlyhigh"]].copy()
    d["low"] = pd.to_numeric(d["dlylow"], errors="coerce").abs()
    d["high"] = pd.to_numeric(d["dlyhigh"], errors="coerce").abs()
    d = d.sort_values(["permno", "transaction_date"]).reset_index(drop=True)
    window = 2 * WEEK_HALF_WIDTH + 1
    grouped = d.groupby("permno", sort=False)
    d["week_low"] = grouped["low"].transform(
        lambda v: v.rolling(window, center=True, min_periods=1).min()
    )
    d["week_high"] = grouped["high"].transform(
        lambda v: v.rolling(window, center=True, min_periods=1).max()
    )
    return d.rename(columns={"transaction_date": "session"})[
        ["permno", "session", "low", "high", "week_low", "week_high"]
    ]


def check_prices(config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Attach the daily and weekly range checks to every comparable universe trade."""
    universe = load_universe(config)
    rows = _map_transaction_permno(load_transactions(universe, config), universe)
    rows = rows.loc[rows["permno"].notna()].copy()
    rows["permno"] = rows["permno"].astype(int)
    ranges = session_ranges(load_daily_ranges(config, set(rows["permno"])))
    ranges["permno"] = ranges["permno"].astype(int)
    merged = pd.merge_asof(
        rows.sort_values("transaction_date"), ranges.sort_values("session"),
        left_on="transaction_date", right_on="session", by="permno",
        direction="backward", tolerance=pd.Timedelta(days=4),
    )
    price = pd.to_numeric(merged["price"], errors="coerce")
    comparable = price.gt(0) & merged["low"].gt(0) & merged["high"].gt(0)
    merged = merged.loc[comparable & merged["transaction_date"].dt.year.between(2006, 2025)].copy()
    price = price.loc[merged.index]
    merged["within_day"] = price.between((1 - TOLERANCE) * merged["low"], (1 + TOLERANCE) * merged["high"])
    merged["within_week"] = price.between(
        (1 - TOLERANCE) * merged["week_low"], (1 + TOLERANCE) * merged["week_high"]
    )
    merged["below_week_low"] = price.lt((1 - TOLERANCE) * merged["week_low"])
    merged["above_week_high"] = price.gt((1 + TOLERANCE) * merged["week_high"])
    merged["ratio_to_week_low"] = np.where(merged["below_week_low"], price / merged["week_low"], np.nan)
    return merged


def summarize(checked: pd.DataFrame) -> pd.DataFrame:
    """Summarize pass rates by transaction code."""
    out = checked.groupby("transaction_code").agg(
        comparable_rows=("within_day", "size"),
        share_within_day_range=("within_day", "mean"),
        share_within_week_range=("within_week", "mean"),
        share_below_week_low=("below_week_low", "mean"),
        share_above_week_high=("above_week_high", "mean"),
        median_price_to_week_low_when_below=("ratio_to_week_low", "median"),
    )
    return out.reset_index()


def main() -> int:
    """Run the check on local data and write the aggregate summary."""
    summary = summarize(check_prices())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
