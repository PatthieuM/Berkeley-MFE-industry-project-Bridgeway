#!/usr/bin/env python3
"""Run offline unit checks and local-data regression checks for the signal pipeline.

Tests construct in-memory tables or read configured SEC, universe, and market files;
the runner prints pass/fail results and returns a process status code.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from shared.config import DEFAULT_CONFIG
from shared.events.events import load_insider_events, load_prices, load_universe, load_xbi, price_levels
from shared.engine.evaluate import event_returns, summary_stats
from shared.engine.continuity import audit_continuity
from shared.engine.sanity import add_large_trade_flags
from shared.sec.acquire_normalize import normalize_all_archive, synthetic_zip_bytes
from shared.sec.form144 import Filing144, parse_form144_xml, parse_master_index
from signals.smallcap_purchases.signals import build_signals
from signals.smallcap_purchases.conditioning import _role_bucket, classify_seller_decisions
from signals.seller_silence.ssn import availability_date, build_owner_months


def check(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with context when a test condition is false."""
    if not condition:
        raise AssertionError(message)


def test_timing() -> None:
    """Verify next-session entry and rejection of incomplete price paths."""
    calendar = pd.to_datetime(["2021-07-01", "2021-07-02", "2021-07-06", "2021-07-07", "2021-07-08"])
    benchmark = pd.Series([100, 101, 102, 103, 104], index=calendar, dtype=float)
    prices = pd.DataFrame({
        "date": np.tile(calendar, 2),
        "security_id": np.repeat([1, 2], len(calendar)),
        "close": [10, 11, np.nan, 12, 13, 20, 21, 22, 23, 24],
        "delisted": False,
    })
    events = pd.DataFrame({
        "event_id": ["friday-missing", "holiday"],
        "security_id": [1, 2],
        "filing_date": pd.to_datetime(["2021-07-02", "2021-07-05"]),
        "issuer_cik": [1, 2],
    })
    result = event_returns(events, prices, benchmark, [1]).set_index("event_id")
    expected_entry = pd.Timestamp("2021-07-06")
    check(
        result.loc["friday-missing", "entry_date"] == expected_entry,
        "Friday filing did not enter Tuesday after the holiday",
    )
    check(result.loc["holiday", "entry_date"] == expected_entry, "Holiday filing did not enter on the next session")
    check(pd.isna(result.loc["friday-missing", "stock_return"]), "Missing entry session was shifted or zero-filled")
    check(not pd.isna(result.loc["holiday", "stock_return"]), "Complete holiday path was incorrectly rejected")


def test_benchmark_alignment() -> None:
    """Verify stock and benchmark returns use identical window endpoints."""
    calendar = pd.date_range("2021-01-01", periods=4, freq="D")
    benchmark = pd.Series([100.0, 101.0, 103.02, 106.1106], index=calendar)
    prices = pd.DataFrame({
        "date": calendar,
        "security_id": 7,
        "close": [100.0, 102.0, 105.06, 109.2624],
        "delisted": False,
    })
    events = pd.DataFrame(
        {"event_id": ["aligned"], "security_id": [7], "issuer_cik": [7], "filing_date": [calendar[0]]}
    )
    row = event_returns(events, prices, benchmark, [2]).iloc[0]
    check(
        row["entry_date"] == calendar[1] and row["end_date"] == calendar[3],
        "Stock and benchmark window endpoints differ",
    )
    check(abs(row["stock_return"] - (109.2624 / 102.0 - 1)) < 1e-12, "Stock compounding is wrong")
    check(abs(row["benchmark_return"] - (106.1106 / 101.0 - 1)) < 1e-12, "Benchmark compounding is wrong")


def test_clustered_se_sanity() -> None:
    """Check clustered inference under null and injected-signal simulations."""
    rng = np.random.default_rng(240924)
    issuers = np.repeat(np.arange(40), 24)
    month_index = np.tile(np.arange(24), 40)
    dates = pd.to_datetime("2018-01-01") + pd.to_timedelta(month_index * 31, unit="D")
    issuer_effect = np.repeat(rng.normal(0, 0.01, 40), 24)
    month_effect = np.tile(rng.normal(0, 0.01, 24), 40)
    noise = rng.normal(0, 0.03, len(issuers))
    zero = issuer_effect + month_effect + noise
    zero -= zero.mean()
    base = pd.DataFrame({"issuer_cik": issuers, "filing_date": dates, "excess_return": zero})
    null = summary_stats(base)
    check(null["ci_low"] <= 0 <= null["ci_high"], "Zero-signal clustered CI does not cover zero")
    injected = base.copy()
    injected["excess_return"] += 0.02
    detected = summary_stats(injected)
    check(detected["ci_low"] > 0 and detected["p_two_sided"] < 0.05, "Injected signal was not detected")


def test_ssn_point_in_time() -> None:
    """Verify late-filed sales cannot alter seller state before availability."""
    sales = pd.DataFrame({
        "owner_cik": [10, 10, 10], "issuer_cik": [20, 20, 20],
        "transaction_date": pd.to_datetime(["2019-05-15", "2020-05-14", "2021-05-20"]),
        "filing_date": pd.to_datetime(["2019-05-17", "2020-05-18", "2021-06-10"]),
    })
    row = build_owner_months(sales, 2021, 2021).iloc[0]
    check(row["owner_state"] == "SSN", "A sale filed after availability incorrectly flipped SSN")
    check(bool(row["late_current_sale"]), "Late current-year sale was not exposed by the audit flag")


def test_ssn_routine_definition() -> None:
    """Verify routine sellers require the same month in both prior years."""
    sales = pd.DataFrame({
        "owner_cik": [1, 1, 2, 2], "issuer_cik": [9, 9, 9, 9],
        "transaction_date": pd.to_datetime(["2019-03-04", "2020-03-05", "2019-03-04", "2020-04-05"]),
        "filing_date": pd.to_datetime(["2019-03-06", "2020-03-09", "2019-03-06", "2020-04-07"]),
    })
    result = build_owner_months(sales, 2021, 2021)
    check(len(result) == 1, "Routine status did not require the same calendar month in both prior years")
    check(int(result.iloc[0]["owner_cik"]) == 1 and int(result.iloc[0]["signal_month"].month) == 3,
          "Routine seller-month was classified incorrectly")


def test_ssn_availability_date() -> None:
    """Verify federal-business-day handling for seller-silence availability."""
    check(availability_date(2021, 5) == pd.Timestamp("2021-06-02"),
          "Memorial Day was not respected in the second-federal-business-day rule")
    check(availability_date(2021, 7) == pd.Timestamp("2021-08-03"),
          "Weekend month-end availability date is incorrect")


def test_sec_all_offline() -> None:
    """Verify offline normalization retains all expected SEC table content."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "2021q3_form345.zip"
        archive.write_bytes(synthetic_zip_bytes())
        manifest = normalize_all_archive(archive, root / "out", "2021q3")
        check(manifest["forms"] == {"4": 1}, "All-form manifest form counts are wrong")
        check(manifest["tables"]["deriv_transactions"] == 1, "Derivative transactions were not retained")
        tx = pd.read_parquet(root / "out/nonderiv_transactions.parquet")
        check(set(tx["trans_code"]) == {"P"}, "All-table normalizer changed transaction codes")


def test_form144_offline() -> None:
    """Verify offline Form 144 discovery and XML field extraction."""
    index = "CIK|Company Name|Form Type|Date Filed|Filename\n1234|Example Bio|144/A|2024-05-01|edgar/data/1234/0000001234-24-000001.txt\n"
    filings = parse_master_index(index, {1234})
    check(len(filings) == 1 and filings[0].form == "144/A", "Form 144/A master-index discovery failed")
    xml = b"""<?xml version='1.0'?><edgarSubmission><formData><issuerInfo>
      <issuerCik>0000001234</issuerCik><issuerName>Example Bio</issuerName>
      <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Jane Doe</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>
      <relationshipsToIssuer><relationshipToIssuer>Officer</relationshipToIssuer></relationshipsToIssuer>
      </issuerInfo><securitiesInformation><securitiesClassTitle>Common Stock</securitiesClassTitle>
      <noOfUnitsSold>1200</noOfUnitsSold><aggregateMarketValue>60000</aggregateMarketValue>
      <approxSaleDate>05/15/2024</approxSaleDate><brokerOrMarketmakerDetails><name>Broker LLC</name></brokerOrMarketmakerDetails>
      </securitiesInformation><is10b5OnePlan>true</is10b5OnePlan><dateOfAdoptionOfTradingPlan>01/02/2024</dateOfAdoptionOfTradingPlan>
      </formData></edgarSubmission>"""
    parsed = parse_form144_xml(xml, filings[0]).iloc[0]
    check(parsed["issuer_cik"] == "0000001234" and parsed["shares_to_be_sold"] == 1200, "Form 144 XML values failed")
    check(
        "is10b5OnePlan=true" in parsed["plan_10b5_1"] and parsed["broker"] == "Broker LLC",
        "Form 144 plan/broker failed",
    )


def test_continuity_offline() -> None:
    """Verify ownership-continuity status mapping on a two-event sequence."""
    rows = pd.DataFrame([
        {"table": "non_derivative", "row_kind": "transaction", "owner_cik": "1", "issuer_cik": "9",
         "security_title": "Common Stock", "direct_or_indirect": "D", "form": "4", "accession": "a",
         "filing_date": "2020-01-02", "transaction_date": "2020-01-01", "transaction_code": "P",
         "acquired_disposed": "A", "shares": "100", "shares_owned_following": "100", "permno": 11},
        {"table": "non_derivative", "row_kind": "transaction", "owner_cik": "1", "issuer_cik": "9",
         "security_title": "Common Stock", "direct_or_indirect": "D", "form": "4", "accession": "b",
         "filing_date": "2020-02-02", "transaction_date": "2020-02-01", "transaction_code": "S",
         "acquired_disposed": "D", "shares": "40", "shares_owned_following": "60", "permno": 11},
    ])
    result = audit_continuity(rows)
    check(result["status"].tolist() == ["first_observation", "consistent"], "Continuity status mapping failed")


def test_large_trade_offline() -> None:
    """Verify large-trade thresholds against cap and dollar volume."""
    rows = pd.DataFrame({
        "shares": [1000, 10], "price": [20, 20], "market_cap_lag": [1_000_000, 1_000_000],
        "average_dollar_volume_20_lag": [15_000, 15_000],
    })
    flagged = add_large_trade_flags(rows)
    check(
        bool(flagged.iloc[0]["large_trade"]) and not bool(flagged.iloc[1]["large_trade"]),
        "Large-trade threshold failed",
    )


def test_conditioning_offline() -> None:
    """Verify role buckets and issuer-day seller classification precedence."""
    mixed = pd.DataFrame({"role": ["Director", "10% owner"], "officer_title": ["", ""]})
    executive = pd.DataFrame({"role": ["Other officer"], "officer_title": ["Chief Financial Officer"]})
    check(_role_bucket(mixed) is None, "Mixed director/owner day was labelled as an only-role group")
    check(_role_bucket(executive) == "senior_executive", "C-suite title precedence failed")
    rows = []
    for owner, months in ((1, [3, 3, 3, 6]), (2, [1, 2, 3, 6]), (3, [1, 2, None, 6])):
        for year, month in zip((2018, 2019, 2020, 2021), months):
            if month is not None:
                rows.append({"owner_cik": owner, "issuer_cik": 9, "transaction_date": f"{year}-{month:02d}-05",
                             "filing_date": f"{year}-{month:02d}-07"})
    classified = classify_seller_decisions(pd.DataFrame(rows))
    current = classified.loc[classified["filing_date"].eq(pd.Timestamp("2021-06-07"))]
    # Issuer-day precedence is opportunistic when any seller is opportunistic.
    check(current.iloc[0]["seller_class"] == "opportunistic", "CMP issuer-day precedence failed")


REAL_FIXTURES = {}


def build_real_fixture(source: str):
    """Build and cache the real-data signal fixture for a price source."""
    if source in REAL_FIXTURES:
        return REAL_FIXTURES[source]
    config = DEFAULT_CONFIG.with_price_source(source)
    universe = load_universe(config)
    events = load_insider_events(config, universe)
    requested_permnos = (
        set(universe["permno"].dropna().astype(int))
        if source == "universe_full"
        else set(events["permno"].dropna().astype(int))
    )
    daily = load_prices(source, config, requested_permnos)
    benchmark = load_xbi(config)
    signals, breakpoints, featured = build_signals(events, daily, pd.DatetimeIndex(benchmark.index), config)
    start, end = pd.Timestamp(config.study_start), pd.Timestamp(config.study_end)
    signals = {name: frame.loc[frame["filing_date"].between(start, end)].copy() for name, frame in signals.items()}
    fixture = config, universe, events, daily, benchmark, signals, breakpoints, featured
    REAL_FIXTURES[source] = fixture
    return fixture


def test_no_lookahead() -> None:
    """Verify real-data size breakpoints use prior years only."""
    _, _, _, _, _, _, breakpoints, _ = build_real_fixture("v6_cache")
    check(
        breakpoints["training_max_year"].lt(breakpoints["year"]).all(),
        "A real-data breakpoint uses current/future events",
    )
    check(int(breakpoints["training_min_year"].min()) == 2006, "Breakpoint history does not begin in 2006")


def test_v6_regression() -> None:
    """Verify cached-data S1 counts and estimates match frozen benchmarks."""
    _, _, _, daily, benchmark, signals, _, _ = build_real_fixture("v6_cache")
    levels = price_levels(daily)
    event_input = signals["S1"].rename(columns={"permno": "security_id"})
    outcomes = event_returns(event_input, levels, benchmark, [5])
    full = summary_stats(outcomes)
    early = summary_stats(outcomes.loc[outcomes["filing_date"].dt.year.between(2009, 2019)])
    late = summary_stats(outcomes.loc[outcomes["filing_date"].dt.year.between(2020, 2025)])
    check(full["N"] == 2684, f"v6 N mismatch: {full['N']} != 2684")
    check(abs(100 * full["mean"] - 1.509) <= 0.001, f"v6 mean mismatch: {100*full['mean']:.6f} pp")
    check(abs(full["t"] - 3.91) <= 0.01, f"v6 t mismatch: {full['t']:.6f}")
    check(abs(100 * early["mean"] - 1.614) <= 0.001, f"v6 early mean mismatch: {100*early['mean']:.6f} pp")
    check(abs(100 * late["mean"] - 1.405) <= 0.001, f"v6 late mean mismatch: {100*late['mean']:.6f} pp")


def test_universe_full_coverage() -> None:
    """Verify full-universe price coverage and valid five-session paths."""
    config, universe, _, daily, benchmark, signals, _, _ = build_real_fixture("universe_full")
    universe_permnos = set(universe["permno"].dropna().astype(int))
    priced_permnos = set(daily["permno"].dropna().astype(int))
    share_prices = len(universe_permnos & priced_permnos) / len(universe_permnos)
    combined = pd.concat([signals["S2"], signals["S3"]], ignore_index=True).drop_duplicates("event_id")
    outcomes = event_returns(combined.rename(columns={"permno": "security_id"}), price_levels(daily), benchmark, [5])
    share_paths = outcomes["excess_return"].notna().mean()
    check(share_prices == 1.0, f"Universe price coverage is {share_prices:.2%}, expected 100%")
    check(0 < share_paths <= 1, f"Invalid complete-path share: {share_paths}")
    print(
        f"    coverage: PERMNOs {len(universe_permnos & priced_permnos)}/{len(universe_permnos)}; complete H5 paths {share_paths:.2%}"
    )


TESTS = [
    ("sec_all_offline", test_sec_all_offline),
    ("form144_offline", test_form144_offline),
    ("continuity_offline", test_continuity_offline),
    ("large_trade_offline", test_large_trade_offline),
    ("conditioning_offline", test_conditioning_offline),
    ("timing", test_timing),
    ("ssn_point_in_time", test_ssn_point_in_time),
    ("ssn_routine_definition", test_ssn_routine_definition),
    ("ssn_availability_date", test_ssn_availability_date),
    ("no_lookahead", test_no_lookahead),
    ("benchmark_alignment", test_benchmark_alignment),
    ("clustered_se_sanity", test_clustered_se_sanity),
    ("v6_regression", test_v6_regression),
    ("universe_full_coverage", test_universe_full_coverage),
]


def main() -> int:
    """Run every registered test and return a nonzero status on failure."""
    failed = []
    for name, function in TESTS:
        try:
            function()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"\n{len(TESTS)-len(failed)}/{len(TESTS)} tests passed")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
