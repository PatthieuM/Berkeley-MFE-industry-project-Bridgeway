"""Run flag-based data sanity checks for S1 and S4 tables and market paths.

Flags are retained and summarized; the official samples are never silently
altered. The returned summary, robustness, and filing-lag tables support reporting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from shared.config import Config, DEFAULT_CONFIG
from shared.engine.evaluate import event_returns, summary_stats
from signals.seller_silence.ssn import contrast_stats, run_ssn


TX_FLAGS = (
    "direction_inconsistent", "transaction_after_filing", "late_filing_gt_30",
    "price_outside_daily_range", "likely_price_scale_error",
    "duplicate_transaction_across_filings",
    "large_trade_vs_market_cap", "large_trade_vs_average_dollar_volume",
)


def _dates(values) -> pd.Series:
    """Parse values to dates normalized to midnight; bad values become NaT."""
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def load_transactions(universe: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load P/S rows with owner and fields needed for sanity checks."""
    target = set(pd.to_numeric(universe["issuer_cik"], errors="coerce").dropna().astype(int))
    start, end = pd.Timestamp(config.breakpoint_history_start), pd.Timestamp(config.transaction_date_end)
    frames: list[pd.DataFrame] = []
    for sub_path in sorted(config.normalized_sec_dir.glob("year=*/quarter=*/submissions.parquet")):
        year = int(sub_path.parent.parent.name.split("=", 1)[1])
        if year < start.year or year > end.year:
            continue
        sub = pd.read_parquet(sub_path, columns=[
            "accession", "filing_date", "form", "issuer_cik", "is_amendment"
        ])
        sub["filing_date"] = _dates(sub["filing_date"])
        sub["issuer_cik"] = pd.to_numeric(sub["issuer_cik"], errors="coerce").astype("Int64")
        sub = sub.loc[
            sub["form"].eq("4") & ~sub["is_amendment"].fillna(False)
            & sub["issuer_cik"].isin(target) & sub["filing_date"].between(start, end)
        ]
        if sub.empty:
            continue
        accessions = set(sub["accession"])
        tx = pd.read_parquet(sub_path.parent / "transactions.parquet", columns=[
            "accession", "transaction_key", "transaction_date", "transaction_code",
            "shares", "price", "acquired_disposed",
        ])
        tx["transaction_date"] = _dates(tx["transaction_date"])
        tx = tx.loc[
            tx["accession"].isin(accessions) & tx["transaction_code"].isin(["P", "S"])
            & tx["transaction_date"].between(start, end)
        ].drop_duplicates("transaction_key", keep="last")
        if tx.empty:
            continue
        owners = pd.read_parquet(sub_path.parent / "owners.parquet", columns=["accession", "owner_cik"])
        owners["owner_cik"] = pd.to_numeric(owners["owner_cik"], errors="coerce").astype("Int64")
        owners = owners.loc[owners["accession"].isin(set(tx["accession"])) & owners["owner_cik"].notna()]
        if owners.empty:
            continue
        frames.append(tx.merge(sub, on="accession", how="inner", validate="many_to_one")
                      .merge(owners, on="accession", how="inner", validate="many_to_many"))
    if not frames:
        raise RuntimeError("No transactions available for sanity checks")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["transaction_key", "owner_cik"], keep="last")
    result["owner_cik"] = result["owner_cik"].astype("int64")
    result["issuer_cik"] = result["issuer_cik"].astype("int64")
    return result.reset_index(drop=True)


def _map_transaction_permno(rows: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Attach a PERMNO to each transaction row by CIK and universe interval.

    A row is mapped only when the intervals containing its transaction date
    give exactly one PERMNO (point-in-time CIK intervals preferred over
    current-CIK fallbacks); otherwise ``permno`` is left missing.
    """
    base = rows.reset_index(drop=True).copy()
    base["_tx_row"] = np.arange(len(base), dtype=np.int64)
    candidates = base[["_tx_row", "issuer_cik", "transaction_date"]].merge(universe, on="issuer_cik", how="left")
    candidates = candidates.loc[
        candidates["permno"].notna()
        & candidates["transaction_date"].between(candidates["effective_start"], candidates["effective_end"])
    ]
    selected = []
    for row_id, group in candidates.groupby("_tx_row", sort=False):
        current = group.loc[group["cik_is_point_in_time"]] if group["cik_is_point_in_time"].any() else group
        if current["permno"].dropna().nunique() == 1:
            selected.append({"_tx_row": int(row_id), "permno": int(current["permno"].dropna().iloc[0])})
    return base.merge(pd.DataFrame(selected, columns=["_tx_row", "permno"]), on="_tx_row", how="left").drop(
        columns="_tx_row"
    )


def load_daily_ranges(config: Config = DEFAULT_CONFIG, permnos: Optional[set[int]] = None) -> pd.DataFrame:
    """Load daily price ranges plus lagged market-cap and liquidity measures."""
    parts = []
    for path in sorted(config.universe_price_dir.glob("daily_*.parquet")):
        frame = pd.read_parquet(path, columns=[
            "permno", "dlycaldt", "dlylow", "dlyhigh", "dlyclose", "dlyvol", "dlycap",
        ])
        if permnos is not None:
            frame = frame.loc[frame["permno"].isin(permnos)]
        if not frame.empty:
            parts.append(frame)
    daily = pd.concat(parts, ignore_index=True).rename(columns={"dlycaldt": "transaction_date"})
    daily["transaction_date"] = _dates(daily["transaction_date"])
    daily = daily.sort_values(["permno", "transaction_date"])
    daily["market_cap_lag"] = (
        pd.to_numeric(daily["dlycap"], errors="coerce").abs().mul(1000)
        .groupby(daily["permno"]).shift(1)
    )
    dollar_volume = (
        pd.to_numeric(daily["dlyclose"], errors="coerce").abs()
        * pd.to_numeric(daily["dlyvol"], errors="coerce").abs()
    )
    daily["average_dollar_volume_20_lag"] = dollar_volume.groupby(daily["permno"]).transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )
    return daily.drop_duplicates(["permno", "transaction_date"], keep="last")


def add_large_trade_flags(rows: pd.DataFrame) -> pd.DataFrame:
    """Flag reported trade value against lagged cap and trailing 20-session ADV."""
    x = rows.copy()
    price = pd.to_numeric(x["price"], errors="coerce").abs()
    x["reported_trade_value"] = pd.to_numeric(x["shares"], errors="coerce").abs() * price
    x["large_trade_vs_market_cap"] = (
        x["reported_trade_value"].notna() & x["market_cap_lag"].notna()
        & x["reported_trade_value"].gt(.01 * x["market_cap_lag"])
    )
    x["large_trade_vs_average_dollar_volume"] = (
        x["reported_trade_value"].notna() & x["average_dollar_volume_20_lag"].notna()
        & x["reported_trade_value"].gt(x["average_dollar_volume_20_lag"])
    )
    x["large_trade"] = x[["large_trade_vs_market_cap", "large_trade_vs_average_dollar_volume"]].any(axis=1)
    return x


def flag_transactions(rows: pd.DataFrame, universe: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Attach every transaction-level flag without dropping observations."""
    x = rows.copy()
    x["filing_lag_days"] = (x["filing_date"] - x["transaction_date"]).dt.days
    x["direction_inconsistent"] = ~(
        (x["transaction_code"].eq("P") & x["acquired_disposed"].eq("A"))
        | (x["transaction_code"].eq("S") & x["acquired_disposed"].eq("D"))
    )
    x["transaction_after_filing"] = x["filing_lag_days"].lt(0) | x["filing_lag_days"].isna()
    x["late_filing_gt_30"] = x["filing_lag_days"].gt(30)
    duplicate_keys = ["owner_cik", "issuer_cik", "transaction_date", "shares", "price"]
    accession_count = x.groupby(duplicate_keys, dropna=False)["accession"].transform("nunique")
    x["duplicate_transaction_across_filings"] = accession_count.gt(1)
    x = _map_transaction_permno(x, universe)
    ranges = load_daily_ranges(config, set(x["permno"].dropna().astype(int)))
    x = x.merge(ranges, on=["permno", "transaction_date"], how="left", validate="many_to_one")
    price = pd.to_numeric(x["price"], errors="coerce")
    low = pd.to_numeric(x["dlylow"], errors="coerce").abs()
    high = pd.to_numeric(x["dlyhigh"], errors="coerce").abs()
    comparable = price.notna() & price.gt(0) & low.notna() & high.notna() & low.gt(0) & high.gt(0)
    x["price_range_comparable"] = comparable
    x["price_outside_daily_range"] = comparable & ((price < .99 * low) | (price > 1.01 * high))
    x["likely_price_scale_error"] = comparable & ((price >= 10 * high) | (price <= low / 10))
    x = add_large_trade_flags(x)
    x["any_transaction_flag"] = x[list(TX_FLAGS)].any(axis=1)
    return x


def _event_market_flags(outcomes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Flag extreme returns, missing paths and delistings for each event-horizon."""
    result = outcomes[
        ["event_id", "horizon", "security_id", "entry_date", "end_date", "partial_path", "delisted"]
    ].copy()
    groups = {
        int(permno): frame.set_index("date")["dlyret"]
        for permno, frame in daily.groupby("permno", sort=False)
    }
    extreme = []
    for row in result.itertuples(index=False):
        series = groups.get(int(row.security_id)) if pd.notna(row.security_id) else None
        if series is None or pd.isna(row.entry_date) or pd.isna(row.end_date):
            extreme.append(False)
        else:
            window = pd.to_numeric(
                series.loc[(series.index >= row.entry_date) & (series.index <= row.end_date)], errors="coerce"
            )
            extreme.append(bool(window.abs().gt(2).any()))
    result["extreme_daily_return_gt_200pct"] = extreme
    result["missing_sessions_in_window"] = result["partial_path"].fillna(True)
    result["delisting_in_window"] = result["delisted"].fillna(False)
    result["any_market_flag"] = result[[
        "extreme_daily_return_gt_200pct", "missing_sessions_in_window", "delisting_in_window"
    ]].any(axis=1)
    return result


def _summary_rows(signal: str, transactions: pd.DataFrame, market: pd.DataFrame) -> list[dict]:
    """Count how many events each sanity check flags for one signal.

    Returns one row per check with the number of events checked, the number
    flagged and the percentage. Transaction checks use ``transactions``; path
    checks (extreme returns, missing sessions, delistings) use ``market``.
    """
    rows: list[dict] = []
    labels = {
        "direction_inconsistent": "direction_consistency",
        "transaction_after_filing": "transaction_date_after_filing_date",
        "late_filing_gt_30": "filing_lag_over_30_days",
        "price_outside_daily_range": "reported_price_outside_CRSP_low_high_pm1pct",
        "likely_price_scale_error": "likely_reported_price_scale_error_10x",
        "duplicate_transaction_across_filings": "identical_owner_date_shares_price_multiple_filings",
        "large_trade_vs_market_cap": "reported_value_over_1pct_lagged_market_cap",
        "large_trade_vs_average_dollar_volume": "reported_value_over_trailing_20_session_average_dollar_volume",
    }
    for flag, label in labels.items():
        checked = len(transactions)
        flagged = int(transactions[flag].sum())
        rows.append({"signal": signal, "check": label, "events_checked": checked,
                     "flagged": flagged, "percent_flagged": 100 * flagged / checked if checked else np.nan})
    for flag in ["extreme_daily_return_gt_200pct", "missing_sessions_in_window", "delisting_in_window"]:
        checked = len(market)
        flagged = int(market[flag].sum())
        rows.append({"signal": signal, "check": flag, "events_checked": checked,
                     "flagged": flagged, "percent_flagged": 100 * flagged / checked if checked else np.nan})
    rows.append({
        "signal": signal,
        "check": "ownership_continuity_UNAVAILABLE_normalized_archive_keeps_only_P_S_rows",
        "events_checked": 0, "flagged": pd.NA, "percent_flagged": np.nan,
    })
    return rows


def _robust_s1(analysis: dict, flagged_event_ids: set[str]) -> list[dict]:
    """Re-estimate S1 at 5 sessions with sanity-flagged events removed.

    Returns, for each period, the official and the robust N, mean, t and p
    side by side.
    """
    official = analysis["results_table"]
    events = analysis["signals"]["S1"]
    clean = events.loc[~events["event_id"].isin(flagged_event_ids)].rename(columns={"permno": "security_id"})
    outcomes = event_returns(clean, analysis["levels"], analysis["benchmark"], [5])
    rows = []
    for period, low, high in (("2009-2025", 2009, 2025), ("2009-2019", 2009, 2019), ("2020-2025", 2020, 2025)):
        off = official.loc[
            (official["signal"].eq("S1")) & official["horizon"].eq(5) & official["period"].eq(period)
        ].iloc[0]
        robust = summary_stats(outcomes.loc[outcomes["filing_date"].dt.year.between(low, high)])
        rows.append({"signal": "S1", "period": period, "horizon": 5, "estimand": "mean_excess",
                     "official_N": int(off.N), "official_effect": off["mean"], "official_t": off.t,
                     "official_p": off.p_two_sided, "robust_N": robust["N"],
                     "robust_effect": robust["mean"], "robust_t": robust["t"],
                     "robust_p": robust["p_two_sided"]})
    return rows


def _robust_s4(official_bundle: dict, robust_bundle: dict, robust_market: pd.DataFrame) -> list[dict]:
    """Re-estimate S4 at 63 and 126 sessions with market-flagged outcomes set to missing.

    Returns, for each period and horizon, the official and the robust
    SSN-minus-SSS-only contrast side by side.
    """
    rows = []
    official = official_bundle["results"]
    outcomes = robust_bundle["outcomes"].merge(
        robust_market[["event_id", "horizon", "any_market_flag"]], on=["event_id", "horizon"], how="left"
    )
    outcomes.loc[outcomes["any_market_flag"].fillna(False), "excess_return"] = np.nan
    for period, low, high in (("2009-2025", 2009, 2025), ("2009-2019", 2009, 2019), ("2020-2025", 2020, 2025)):
        for horizon in (63, 126):
            off = official.loc[official["period"].eq(period) & official["horizon"].eq(horizon)].iloc[0]
            current = outcomes.loc[
                outcomes["horizon"].eq(horizon) & outcomes["signal_month"].dt.year.between(low, high)
            ]
            robust = contrast_stats(current)
            rows.append({"signal": "S4", "period": period, "horizon": horizon,
                         "estimand": "SSN_minus_SSS_only", "official_N": int(off.N_SSN + off.N_SSS_only),
                         "official_effect": off.difference, "official_t": off.t, "official_p": off.p_two_sided,
                         "robust_N": int(robust["N_SSN"] + robust["N_SSS_only"]),
                         "robust_effect": robust["difference"], "robust_t": robust["t"],
                         "robust_p": robust["p_two_sided"]})
    return rows


def run_sanity(analysis: dict, ssn_bundle: dict, config: Config = DEFAULT_CONFIG) -> dict:
    """Return S1/S4 flag summaries, exclusion robustness, and filing-lag statistics."""
    raw = load_transactions(analysis["universe"], config)
    flagged = flag_transactions(raw, analysis["universe"], config)

    s1_events = analysis["signals"]["S1"][["event_id", "issuer_cik", "filing_date", "permno"]]
    s1_tx = flagged.loc[flagged["transaction_code"].eq("P")].merge(
        s1_events, on=["issuer_cik", "filing_date"], how="inner", suffixes=("", "_event")
    )
    s1_out = analysis["outcomes"]["S1"].loc[analysis["outcomes"]["S1"]["horizon"].eq(5)]
    s1_market = _event_market_flags(s1_out, analysis["daily"])
    s1_tx_bad = set(s1_tx.loc[s1_tx["any_transaction_flag"], "event_id"])
    s1_market_bad = set(s1_market.loc[s1_market["any_market_flag"], "event_id"])

    routine_pairs = ssn_bundle["owner_months"][["owner_cik", "issuer_cik"]].drop_duplicates()
    s4_tx = flagged.loc[flagged["transaction_code"].eq("S")].merge(
        routine_pairs, on=["owner_cik", "issuer_cik"], how="inner"
    )
    s4_primary = ssn_bundle["outcomes"].loc[ssn_bundle["outcomes"]["horizon"].isin([63, 126])]
    s4_market = _event_market_flags(s4_primary, analysis["daily"])

    summary = pd.DataFrame(
        _summary_rows("S1", s1_tx, s1_market) + _summary_rows("S4", s4_tx, s4_market)
    )

    bad_sale_keys = s4_tx.loc[s4_tx["any_transaction_flag"], ["transaction_key", "owner_cik"]].drop_duplicates()
    robust_sales = ssn_bundle["sales"].merge(
        bad_sale_keys.assign(_bad=True), on=["transaction_key", "owner_cik"], how="left"
    )
    robust_sales = robust_sales.loc[~robust_sales["_bad"].fillna(False)].drop(columns="_bad")
    robust_ssn = run_ssn(analysis["universe"], analysis["levels"], analysis["benchmark"], config, robust_sales)
    robust_s4_market = _event_market_flags(
        robust_ssn["outcomes"].loc[robust_ssn["outcomes"]["horizon"].isin([63, 126])], analysis["daily"]
    )
    robustness = pd.DataFrame(
        _robust_s1(analysis, s1_tx_bad | s1_market_bad)
        + _robust_s4(ssn_bundle, robust_ssn, robust_s4_market)
    )
    lag = flagged["filing_lag_days"].describe(percentiles=[.5, .9, .95, .99]).rename("days").reset_index()
    return {"summary": summary, "robustness": robustness, "filing_lag_distribution": lag}
