"""Point-in-time Hong--Li seller-silence signal (S4).

The construction is deliberately independent of S1--S3.  It uses original
Form 4 non-derivative code-S rows, expands joint filings to every reporting
owner, and freezes each signal at the second US-federal business day after
month end.  No row-level licensed data are written by this module.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay, MonthEnd
from scipy import stats
import statsmodels.api as sm

from shared.config import Config, DEFAULT_CONFIG
from shared.engine.evaluate import event_returns, summary_stats


SEC_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
S4_HORIZONS = (1, 2, 3, 5, 10, 21, 42, 63, 126)
S4_PERIODS = (("2009-2025", 2009, 2025), ("2009-2019", 2009, 2019), ("2020-2025", 2020, 2025))


def availability_date(year: int, month: int) -> pd.Timestamp:
    """Second US-federal-business day after the calendar month end."""
    return pd.Timestamp(year=int(year), month=int(month), day=1) + MonthEnd(0) + 2 * SEC_BUSINESS_DAY


def _dates(values) -> pd.Series:
    """Parse values to dates normalized to midnight; bad values become NaT."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def load_owner_sales(
    universe: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load original code-S executions and expand each filing to its owner CIKs."""
    target_ciks = set(pd.to_numeric(universe["issuer_cik"], errors="coerce").dropna().astype(int))
    start = pd.Timestamp(config.breakpoint_history_start)
    end = pd.Timestamp(config.transaction_date_end)
    frames: list[pd.DataFrame] = []
    audit = {"owners": 0, "owner_issuer_pairs": 0, "owner_sale_rows": 0, "sale_month_rows": 0}
    for sub_path in sorted(config.normalized_sec_dir.glob("year=*/quarter=*/submissions.parquet")):
        year = int(sub_path.parent.parent.name.split("=", 1)[1])
        if year < start.year or year > end.year:
            continue
        sub = pd.read_parquet(
            sub_path,
            columns=["accession", "filing_date", "form", "issuer_cik", "is_amendment"],
        )
        sub["filing_date"] = _dates(sub["filing_date"])
        sub["issuer_cik"] = pd.to_numeric(sub["issuer_cik"], errors="coerce").astype("Int64")
        sub = sub.loc[
            sub["form"].eq("4")
            & ~sub["is_amendment"].fillna(False)
            & sub["issuer_cik"].isin(target_ciks)
            & sub["filing_date"].between(start, end)
        ].copy()
        if sub.empty:
            continue
        accessions = set(sub["accession"])
        tx = pd.read_parquet(
            sub_path.parent / "transactions.parquet",
            columns=["accession", "transaction_key", "transaction_date", "transaction_code"],
        )
        tx["transaction_date"] = _dates(tx["transaction_date"])
        tx = tx.loc[
            tx["accession"].isin(accessions)
            & tx["transaction_code"].eq("S")
            & tx["transaction_date"].between(start, end)
        ].drop_duplicates("transaction_key", keep="last")
        if tx.empty:
            continue
        owners = pd.read_parquet(sub_path.parent / "owners.parquet", columns=["accession", "owner_cik"])
        owners["owner_cik"] = pd.to_numeric(owners["owner_cik"], errors="coerce").astype("Int64")
        owners = owners.loc[owners["accession"].isin(set(tx["accession"])) & owners["owner_cik"].notna()]
        if owners.empty:
            continue
        frames.append(
            tx.merge(sub, on="accession", how="inner", validate="many_to_one")
            .merge(owners, on="accession", how="inner", validate="many_to_many")
        )
    if not frames:
        raise RuntimeError("No owner-level code-S rows found")
    sales = pd.concat(frames, ignore_index=True)
    sales = sales.drop_duplicates(["transaction_key", "owner_cik"], keep="last")
    sales["issuer_cik"] = sales["issuer_cik"].astype("int64")
    sales["owner_cik"] = sales["owner_cik"].astype("int64")
    sales["transaction_year"] = sales["transaction_date"].dt.year.astype(int)
    sales["transaction_month"] = sales["transaction_date"].dt.month.astype(int)
    audit.update(
        {
            "owners": int(sales["owner_cik"].nunique()),
            "owner_issuer_pairs": int(sales[["owner_cik", "issuer_cik"]].drop_duplicates().shape[0]),
            "owner_sale_rows": int(len(sales)),
            "sale_month_rows": int(
                sales[["owner_cik", "issuer_cik", "transaction_year", "transaction_month"]]
                .drop_duplicates()
                .shape[0]
            ),
        }
    )
    return sales.sort_values(["filing_date", "issuer_cik", "owner_cik"], kind="mergesort").reset_index(drop=True), audit


def build_owner_months(
    sales: pd.DataFrame,
    start_year: int = 2009,
    end_year: int = 2025,
    source_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Classify routine owner-months as SSN or SSS using only then-public filings."""
    required = {"owner_cik", "issuer_cik", "transaction_date", "filing_date"}
    if missing := required.difference(sales.columns):
        raise ValueError(f"sales missing columns: {sorted(missing)}")
    x = sales.copy()
    x["transaction_date"] = _dates(x["transaction_date"])
    x["filing_date"] = _dates(x["filing_date"])
    x["transaction_year"] = x["transaction_date"].dt.year.astype("Int64")
    x["transaction_month"] = x["transaction_date"].dt.month.astype("Int64")
    first = x.groupby(
        ["owner_cik", "issuer_cik", "transaction_year", "transaction_month"], as_index=False
    ).agg(first_filing_date=("filing_date", "min"))
    lookup = {
        (int(r.owner_cik), int(r.issuer_cik), int(r.transaction_year), int(r.transaction_month)): r.first_filing_date
        for r in first.itertuples(index=False)
    }
    pair_months: dict[tuple[int, int], set[int]] = {}
    for r in first.itertuples(index=False):
        pair_months.setdefault((int(r.owner_cik), int(r.issuer_cik)), set()).add(int(r.transaction_month))
    rows: list[dict] = []
    for (owner_cik, issuer_cik), months in pair_months.items():
        for year in range(int(start_year), int(end_year) + 1):
            for month in sorted(months):
                available = availability_date(year, month)
                if source_end is not None and available > pd.Timestamp(source_end):
                    continue
                y1 = lookup.get((owner_cik, issuer_cik, year - 1, month))
                y2 = lookup.get((owner_cik, issuer_cik, year - 2, month))
                if y1 is None or y2 is None or y1 > available or y2 > available:
                    continue
                current = lookup.get((owner_cik, issuer_cik, year, month))
                current_known = current is not None and current <= available
                rows.append({
                    "owner_cik": owner_cik,
                    "issuer_cik": issuer_cik,
                    "signal_month": pd.Timestamp(year=year, month=month, day=1),
                    "availability_date": available,
                    "prior_y1_first_filing": y1,
                    "prior_y2_first_filing": y2,
                    "current_first_filing": current,
                    "owner_state": "SSS" if current_known else "SSN",
                    "late_current_sale": bool(current is not None and current > available),
                })
    owner_months = pd.DataFrame(rows)
    if owner_months.empty:
        raise RuntimeError("No routine seller-months produced")
    return owner_months.sort_values(["signal_month", "issuer_cik", "owner_cik"]).reset_index(drop=True)


def _map_firm_months(firm: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach PERMNO and PERMCO to seller firm-months by CIK and universe interval.

    The interval must contain the month's availability date; point-in-time CIK
    intervals are preferred. Firm-months that still match several PERMNOs are
    dropped and counted. Returns the mapped rows and the number dropped.
    """
    base = firm.reset_index(drop=True).copy()
    base["_row"] = np.arange(len(base), dtype=np.int64)
    candidates = base[["_row", "issuer_cik", "availability_date"]].merge(universe, on="issuer_cik", how="left")
    candidates = candidates.loc[
        candidates["permno"].notna()
        & candidates["availability_date"].between(candidates["effective_start"], candidates["effective_end"])
    ].copy()
    selected: list[dict] = []
    ambiguous = 0
    for row_id, group in candidates.groupby("_row", sort=False):
        current = group.loc[group["cik_is_point_in_time"]] if group["cik_is_point_in_time"].any() else group
        if current["permno"].dropna().nunique() != 1:
            ambiguous += 1
            continue
        chosen = current.sort_values(["effective_start", "permno"], ascending=[False, True]).iloc[0]
        selected.append({"_row": int(row_id), "permno": int(chosen["permno"]), "permco": int(chosen["permco"])})
    mapping = pd.DataFrame(selected, columns=["_row", "permno", "permco"])
    mapped = base.merge(mapping, on="_row", how="inner", validate="one_to_one").drop(columns="_row")
    return mapped, ambiguous


def build_firm_months(owner_months: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate owner states, apply SSN precedence, then map at availability."""
    firm = owner_months.groupby(["issuer_cik", "signal_month", "availability_date"], as_index=False).agg(
        eligible_owner_count=("owner_cik", "nunique"),
        ssn_owner_count=("owner_state", lambda x: int(x.eq("SSN").sum())),
        sss_owner_count=("owner_state", lambda x: int(x.eq("SSS").sum())),
        late_current_sale_owner_count=("late_current_sale", "sum"),
    )
    firm["signal_type"] = np.where(firm["ssn_owner_count"].gt(0), "SSN", "SSS_only")
    pre_counts = firm["signal_type"].value_counts()
    mapped, ambiguous = _map_firm_months(firm, universe)
    mapped["filing_date"] = mapped["availability_date"]
    mapped["event_id"] = (
        "S4:" + mapped["issuer_cik"].astype(str) + ":" + mapped["signal_month"].dt.strftime("%Y-%m")
    )
    audit = {
        "routine_seller_months": int(len(owner_months)),
        "firm_months_before_universe": int(len(firm)),
        "ssn_firm_months_before_universe": int(pre_counts.get("SSN", 0)),
        "sss_only_firm_months_before_universe": int(pre_counts.get("SSS_only", 0)),
        "ambiguous_share_class": int(ambiguous),
        "firm_months_eligible": int(len(mapped)),
        "ssn_firm_months_eligible": int(mapped["signal_type"].eq("SSN").sum()),
        "sss_only_firm_months_eligible": int(mapped["signal_type"].eq("SSS_only").sum()),
    }
    return mapped.sort_values(["signal_month", "issuer_cik"]).reset_index(drop=True), audit


def contrast_stats(frame: pd.DataFrame) -> dict[str, float]:
    """SSN minus SSS-only with issuer x signal-month clustered inference."""
    d = frame.loc[frame["excess_return"].notna() & frame["signal_type"].isin(["SSN", "SSS_only"])].copy()
    test = d["signal_type"].eq("SSN")
    result = {
        "N_SSN": int(test.sum()), "N_SSS_only": int((~test).sum()),
        "SSN_mean": float(d.loc[test, "excess_return"].mean()),
        "SSS_only_mean": float(d.loc[~test, "excess_return"].mean()),
    }
    result["difference"] = result["SSN_mean"] - result["SSS_only_mean"]
    if min(result["N_SSN"], result["N_SSS_only"]) == 0:
        result.update(clustered_se=np.nan, t=np.nan, p_two_sided=np.nan)
        return result
    design = sm.add_constant(test.astype(float).to_numpy())
    fit = sm.OLS(d["excess_return"].to_numpy(float), design).fit()
    groups = np.column_stack([
        pd.factorize(d["issuer_cik"], sort=False)[0],
        pd.factorize(d["signal_month"].dt.to_period("M").astype(str), sort=False)[0],
    ])
    robust = fit.get_robustcov_results(cov_type="cluster", groups=groups, use_correction=True)
    se = float(robust.bse[1])
    degrees = min(d["issuer_cik"].nunique(), d["signal_month"].dt.to_period("M").nunique()) - 1
    t_value = result["difference"] / se if se > 0 else np.nan
    p_value = 2 * stats.t.sf(abs(t_value), degrees) if degrees > 0 and np.isfinite(t_value) else np.nan
    result.update(clustered_se=se, t=t_value, p_two_sided=float(p_value), df=int(degrees))
    return result


def evaluate_ssn(
    firm_months: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    horizons: Sequence[int] = S4_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate SSN versus SSS-only returns by horizon and study period."""
    events = firm_months.rename(columns={"permno": "security_id"})
    outcomes = event_returns(events, prices[["date", "security_id", "close", "delisted"]], benchmark, horizons)
    rows: list[dict] = []
    for period, low, high in S4_PERIODS:
        period_frame = outcomes.loc[outcomes["signal_month"].dt.year.between(low, high)]
        for horizon in horizons:
            current = period_frame.loc[period_frame["horizon"].eq(horizon)]
            contrast = contrast_stats(current)
            for arm in ("SSN", "SSS_only"):
                arm_stats = summary_stats(current.loc[current["signal_type"].eq(arm)])
                contrast[f"{arm}_t_vs_zero"] = arm_stats["t"]
                contrast[f"{arm}_p_vs_zero"] = arm_stats["p_two_sided"]
            rows.append({"period": period, "horizon": int(horizon), **contrast})
    return pd.DataFrame(rows), outcomes


def counts_table(owner_audit: dict[str, int], firm_audit: dict[str, int], outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return owner, firm-month, and priced-outcome attrition counts."""
    rows = [
        {"stage": "owners", "arm": "all", "horizon": pd.NA, "count": owner_audit["owners"]},
        {
            "stage": "owner_issuer_pairs",
            "arm": "all",
            "horizon": pd.NA,
            "count": owner_audit["owner_issuer_pairs"],
        },
        {
            "stage": "routine_seller_months",
            "arm": "all",
            "horizon": pd.NA,
            "count": firm_audit["routine_seller_months"],
        },
        {
            "stage": "eligible_firm_months",
            "arm": "SSN",
            "horizon": pd.NA,
            "count": firm_audit["ssn_firm_months_eligible"],
        },
        {
            "stage": "eligible_firm_months",
            "arm": "SSS_only",
            "horizon": pd.NA,
            "count": firm_audit["sss_only_firm_months_eligible"],
        },
    ]
    for (arm, horizon), frame in outcomes.groupby(["signal_type", "horizon"], sort=True):
        rows.append(
            {
                "stage": "priced",
                "arm": arm,
                "horizon": int(horizon),
                "count": int(frame["excess_return"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def run_ssn(
    universe: pd.DataFrame,
    levels: pd.DataFrame,
    benchmark: pd.Series,
    config: Config = DEFAULT_CONFIG,
    sales: Optional[pd.DataFrame] = None,
) -> dict:
    """Build, evaluate, and return all seller-silence pipeline artifacts."""
    if sales is None:
        sales, owner_audit = load_owner_sales(universe, config)
    else:
        owner_audit = {
            "owners": int(sales["owner_cik"].nunique()),
            "owner_issuer_pairs": int(sales[["owner_cik", "issuer_cik"]].drop_duplicates().shape[0]),
        }
    owner_months = build_owner_months(sales, 2009, 2025)
    firm_months, firm_audit = build_firm_months(owner_months, universe)
    results, outcomes = evaluate_ssn(firm_months, levels, benchmark)
    return {
        "sales": sales, "owner_months": owner_months, "firm_months": firm_months,
        "results": results, "outcomes": outcomes,
        "counts": counts_table(owner_audit, firm_audit, outcomes),
        "owner_audit": owner_audit, "firm_audit": firm_audit,
    }
