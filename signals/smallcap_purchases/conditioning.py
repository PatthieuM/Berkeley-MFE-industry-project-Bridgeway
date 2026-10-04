"""Descriptive, point-in-time conditioning tables for S1--S3.

The output deliberately reuses the official event returns and two-way clustered
inference.  SEC attributes are reconstructed from the normalized archive; no
row-level licensed market data are written.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from shared.config import Config, DEFAULT_CONFIG
from shared.engine.evaluate import summary_stats


ROLE_ORDER = ("senior_executive", "other_officers", "directors_only", "ten_percent_owners_only")
SELLER_ORDER = ("routine", "opportunistic", "unclassified")


def _dates(values) -> pd.Series:
    """Parse values to dates normalized to midnight; bad values become NaT."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _senior_title(values: pd.Series) -> pd.Series:
    """Flag officer titles that name a CEO, CFO, COO, chair or president."""
    text = values.fillna("").astype(str).str.lower()
    return text.str.contains(
        r"\bceo\b|chief executive|\bcfo\b|chief financial|\bcoo\b|chief operating|"
        r"chair(?:man|woman|person)?\b|\bpresident\b",
        regex=True,
    )


def _role_bucket(group: pd.DataFrame) -> str | None:
    """Assign an issuer-day's buyers to one exclusive role group.

    Precedence: senior executive (by role or title), then other officer, then
    directors only, then 10% owners only. Returns None for any other mix.
    """
    roles = set(group["role"].fillna("Other / unknown").astype(str))
    if "Top executive / chair" in roles or bool(_senior_title(group["officer_title"]).any()):
        return "senior_executive"
    if "Other officer" in roles:
        return "other_officers"
    known = roles - {"Other / unknown"}
    if known == {"Director"}:
        return "directors_only"
    if known == {"10% owner"}:
        return "ten_percent_owners_only"
    return None


def load_sec_conditioning(
    config: Config = DEFAULT_CONFIG, target_ciks: set[int] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return purchase issuer-day attributes and owner-level sale decisions."""
    start = pd.Timestamp(config.breakpoint_history_start)
    end = pd.Timestamp(config.study_end)
    purchase_amounts, purchase_owners, sales = [], [], []
    for sub_path in sorted(config.normalized_sec_dir.glob("year=*/quarter=*/submissions.parquet")):
        year = int(sub_path.parent.parent.name.split("=", 1)[1])
        if year < start.year or year > end.year:
            continue
        sub = pd.read_parquet(
            sub_path, columns=["accession", "filing_date", "form", "issuer_cik", "is_amendment"]
        )
        sub["filing_date"] = _dates(sub["filing_date"])
        sub["issuer_cik"] = pd.to_numeric(sub["issuer_cik"], errors="coerce").astype("Int64")
        sub = sub.loc[
            sub["form"].eq("4") & ~sub["is_amendment"].fillna(False)
            & sub["issuer_cik"].notna() & sub["filing_date"].between(start, end)
        ]
        if target_ciks is not None:
            sub = sub.loc[sub["issuer_cik"].isin(target_ciks)]
        if sub.empty:
            continue
        accessions = set(sub["accession"])
        tx = pd.read_parquet(
            sub_path.parent / "transactions.parquet",
            columns=["accession", "transaction_key", "transaction_date", "transaction_code",
                     "trade_value", "shares", "price"],
        )
        tx = tx.loc[tx["accession"].isin(accessions) & tx["transaction_code"].isin(["P", "S"])].copy()
        tx["transaction_date"] = _dates(tx["transaction_date"])
        tx = tx.drop_duplicates("transaction_key", keep="last")
        if tx.empty:
            continue
        owners = pd.read_parquet(
            sub_path.parent / "owners.parquet",
            columns=["accession", "owner_cik", "role", "officer_title"],
        )
        owners["owner_cik"] = pd.to_numeric(owners["owner_cik"], errors="coerce").astype("Int64")
        owners = owners.loc[owners["accession"].isin(set(tx["accession"])) & owners["owner_cik"].notna()]
        if owners.empty:
            continue

        ptx = tx.loc[tx["transaction_code"].eq("P")].copy()
        if not ptx.empty:
            value = pd.to_numeric(ptx["trade_value"], errors="coerce")
            fallback = pd.to_numeric(ptx["shares"], errors="coerce") * pd.to_numeric(ptx["price"], errors="coerce")
            ptx["reported_dollars"] = value.fillna(fallback).abs()
            filings = sub.loc[sub["accession"].isin(set(ptx["accession"]))]
            amount = ptx.groupby("accession", as_index=False)["reported_dollars"].sum(min_count=1)
            purchase_amounts.append(filings.merge(amount, on="accession"))
            purchase_owners.append(
                owners.loc[owners["accession"].isin(set(ptx["accession"]))]
                .merge(filings[["accession", "issuer_cik", "filing_date"]], on="accession", validate="many_to_one")
            )

        stx = tx.loc[tx["transaction_code"].eq("S")]
        if not stx.empty:
            sale_filings = sub.loc[sub["accession"].isin(set(stx["accession"]))]
            sales.append(
                stx[["accession", "transaction_key", "transaction_date"]]
                .merge(sale_filings, on="accession", validate="many_to_one")
                .merge(owners[["accession", "owner_cik"]], on="accession", validate="many_to_many")
                .drop_duplicates(["transaction_key", "owner_cik"])
            )

    if not purchase_amounts or not purchase_owners or not sales:
        raise RuntimeError("SEC conditioning inputs are empty")
    p = pd.concat(purchase_amounts, ignore_index=True)
    p["issuer_cik"] = p["issuer_cik"].astype("int64")
    day_amount = p.groupby(["issuer_cik", "filing_date"], as_index=False).agg(
        reported_dollars=("reported_dollars", "sum")
    )
    owner = pd.concat(purchase_owners, ignore_index=True)
    owner["issuer_cik"] = owner["issuer_cik"].astype("int64")
    day_role = (
        owner.groupby(["issuer_cik", "filing_date"], sort=False)
        .apply(_role_bucket, include_groups=False).rename("role_bucket").reset_index()
    )
    purchase_days = day_amount.merge(day_role, on=["issuer_cik", "filing_date"], how="left")

    s = pd.concat(sales, ignore_index=True)
    s["issuer_cik"] = s["issuer_cik"].astype("int64")
    s["owner_cik"] = s["owner_cik"].astype("int64")
    return purchase_days, s


def add_prior_year_dollar_terciles(days: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Assign each study-year event using purchase dollars from earlier years only."""
    out = days.copy()
    out["year"] = out["filing_date"].dt.year
    out["dollar_tercile"] = pd.NA
    for year in range(pd.Timestamp(config.study_start).year, pd.Timestamp(config.study_end).year + 1):
        training = out.loc[out["year"].lt(year), "reported_dollars"].dropna()
        current = out.index[out["year"].eq(year) & out["reported_dollars"].notna()]
        if training.empty or len(current) == 0:
            continue
        q1, q2 = training.quantile([1 / 3, 2 / 3]).to_numpy(float)
        values = out.loc[current, "reported_dollars"]
        out.loc[current, "dollar_tercile"] = np.select(
            [values.le(q1), values.le(q2)], ["low", "middle"], default="high"
        )
    return out


def classify_seller_decisions(sales: pd.DataFrame) -> pd.DataFrame:
    """Apply the CMP three-prior-year same-month rule at each year start."""
    x = sales.copy()
    x["transaction_date"] = _dates(x["transaction_date"])
    x["filing_date"] = _dates(x["filing_date"])
    x["transaction_year"] = x["transaction_date"].dt.year
    x["transaction_month"] = x["transaction_date"].dt.month
    classifications: dict[tuple[int, int, int], str] = {}
    for year in sorted(x["transaction_year"].dropna().astype(int).unique()):
        known = x.loc[x["filing_date"].lt(pd.Timestamp(year=year, month=1, day=1))]
        months: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        for row in known.itertuples(index=False):
            months[(int(row.owner_cik), int(row.issuer_cik), int(row.transaction_year))].add(int(row.transaction_month))
        pairs = x.loc[x["transaction_year"].eq(year), ["owner_cik", "issuer_cik"]].drop_duplicates()
        for owner_cik, issuer_cik in pairs.itertuples(index=False, name=None):
            history = [months.get((int(owner_cik), int(issuer_cik), y)) for y in range(year - 3, year)]
            label = "unclassified"
            if all(value is not None for value in history):
                label = "routine" if set.intersection(*history) else "opportunistic"
            classifications[(int(owner_cik), int(issuer_cik), year)] = label
    x["seller_class"] = [
        classifications[(int(row.owner_cik), int(row.issuer_cik), int(row.transaction_year))]
        for row in x.itertuples(index=False)
    ]
    counts = x.groupby(["issuer_cik", "filing_date", "seller_class"])["owner_cik"].nunique().unstack(fill_value=0)
    for label in SELLER_ORDER:
        if label not in counts:
            counts[label] = 0
    counts["seller_class"] = np.select(
        [counts["opportunistic"].gt(0), counts["routine"].gt(0)],
        ["opportunistic", "routine"], default="unclassified",
    )
    return counts.reset_index()[["issuer_cik", "filing_date", "seller_class"]]


def conditioning_table(
    outcomes: Dict[str, pd.DataFrame], config: Config = DEFAULT_CONFIG
) -> pd.DataFrame:
    """Build the requested full-grid descriptive conditioning table."""
    target_ciks = set().union(*(
        set(pd.to_numeric(frame["issuer_cik"], errors="coerce").dropna().astype(int))
        for frame in outcomes.values()
    ))
    purchase_days, sales = load_sec_conditioning(config, target_ciks)
    purchase_days = add_prior_year_dollar_terciles(purchase_days, config)
    seller_days = classify_seller_decisions(sales)
    rows = []
    for signal in ("S1", "S2"):
        frame = outcomes[signal].merge(purchase_days, on=["issuer_cik", "filing_date"], how="left")
        for dimension, values in (("buyer_role", ROLE_ORDER), ("reported_dollar_tercile", ("low", "middle", "high"))):
            column = "role_bucket" if dimension == "buyer_role" else "dollar_tercile"
            for value in values:
                for horizon in config.horizons:
                    current = frame.loc[frame[column].eq(value) & frame["horizon"].eq(horizon)]
                    rows.append({"signal": signal, "dimension": dimension, "group": value,
                                 "horizon": int(horizon), **summary_stats(current)})
    frame = outcomes["S3"].merge(seller_days, on=["issuer_cik", "filing_date"], how="left")
    frame["seller_class"] = frame["seller_class"].fillna("unclassified")
    for value in SELLER_ORDER:
        for horizon in config.horizons:
            current = frame.loc[frame["seller_class"].eq(value) & frame["horizon"].eq(horizon)]
            rows.append({"signal": "S3", "dimension": "seller_class", "group": value,
                         "horizon": int(horizon), **summary_stats(current)})
    return pd.DataFrame(rows)
