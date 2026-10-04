"""Offline regressions for opt-in continuity v2 and the corrected price adapter."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from shared.sec.form4_parser import quality_checks as compatibility
from shared.sec.form4_parser.src import quality_checks as legacy
from shared.sec.form4_parser.src import quality_checks_v2 as qc


def position(day, balance, **changes):
    row = {
        "table": "non_derivative", "row_kind": "transaction", "owner_cik": "1",
        "issuer_cik": "9", "issuer_ticker": "T", "owner_name": "Jane",
        "security_title": "Common Stock", "direct_or_indirect": "D", "form": "4",
        "accession": day, "owner_row_id": day, "transaction_date": day,
        "edgar_filing_date": day, "sequence_in_filing": "1", "shares": "0",
        "acquired_disposed": "A", "shares_owned_following": str(balance), "parse_status": "ok",
    }
    return {**row, **changes}


def sale(**changes):
    return {"issuer_ticker": "T", "transaction_date": "2026-01-05", "transaction_code": "S",
            "shares": "10", "price_per_share": "50", "parse_status": "ok", **changes}


class ContinuityV2Tests(unittest.TestCase):
    def test_legacy_implementation_and_default_export_remain_frozen(self):
        digest = hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()
        self.assertEqual(digest, "f43ebce69b33aa938293bc66bc360168a99f07bc9d52fb40faa58d5a0af40810")
        self.assertIs(compatibility.check_ownership_continuity, legacy.check_ownership_continuity)
        self.assertIs(compatibility.check_ownership_continuity_v2, qc.check_ownership_continuity)

    def test_mixed_cik_and_ticker_split_rows(self):
        rows = pd.DataFrame([position("2026-01-01", 100), position("2026-06-01", 400)])
        variants = [
            [{"issuer_cik": 9, "split_factor": 4}, {"ticker": "OTHER", "split_factor": 2}],
            [{"issuer_cik": 88, "split_factor": 2}, {"ticker": "T", "split_factor": 4}],
            [{"issuer_ticker": "OTHER", "split_factor": 2}, {"ticker": "T", "split_factor": 4}],
        ]
        for adjustments in variants:
            with self.subTest(adjustments=adjustments):
                splits = pd.DataFrame([{**item, "effective_date": "2026-05-01"} for item in adjustments])
                result = qc.check_ownership_continuity(rows, splits)
                self.assertEqual(result.iloc[1]["continuity_status"], "PASS")
                self.assertEqual(result.iloc[1]["split_adjustment_factor"], 4)

    def test_identical_numeric_identifiers_share_a_position_bucket(self):
        rows = pd.DataFrame([position("2026-01-01", 100, owner_cik="0001", issuer_cik="0009"),
                             position("2026-01-02", 999, owner_cik=1.0, issuer_cik=9.0)])
        result = qc.check_ownership_continuity(rows)
        self.assertEqual(result.iloc[1]["continuity_status"], "FLAG_REVIEW_UNADJUSTED")
        self.assertEqual(result.iloc[1]["position_difference"], 899)

    def test_zero_ciks_fall_back_to_owner_names(self):
        rows = pd.DataFrame([position("2026-01-01", 100, owner_cik="0000000000", owner_name="Jane"),
                             position("2026-01-02", 999, owner_cik=0.0, owner_name="John")])
        self.assertEqual(qc.check_ownership_continuity(rows).continuity_status.tolist(),
                         ["BASELINE_NO_PRIOR", "BASELINE_NO_PRIOR"])

    def test_derivative_price_spellings_match_but_instruments_remain_distinct(self):
        rows = pd.DataFrame([
            position("2026-01-01", 100, table="derivative", conversion_or_exercise_price="1"),
            position("2026-01-02", 999, table="derivative", conversion_or_exercise_price="1.0"),
            position("2026-01-03", 50, table="derivative", conversion_or_exercise_price="2"),
        ])
        self.assertEqual(qc.check_ownership_continuity(rows).continuity_status.tolist(),
                         ["BASELINE_NO_PRIOR", "FLAG_REVIEW_UNADJUSTED", "BASELINE_NO_PRIOR"])

    def test_permno_split_scope_takes_precedence_over_issuer(self):
        rows = pd.DataFrame([position("2026-01-01", 100, permno=123),
                             position("2026-06-01", 100, permno=123)])
        splits = pd.DataFrame([{"permno": 456, "issuer_cik": 9, "effective_date": "2026-05-01", "split_factor": 4}])
        result = qc.check_ownership_continuity(rows, splits)
        self.assertEqual(result.iloc[1]["continuity_status"], "PASS")
        self.assertEqual(result.iloc[1]["split_adjustment_factor"], 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            qc.check_ownership_continuity(rows, splits.assign(permno="invalid"))

    def test_invalid_tolerance_is_rejected(self):
        for tolerance in (-1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "tolerance"):
                qc.check_ownership_continuity(pd.DataFrame(), tolerance=tolerance)

    def test_tolerance_is_absolute_shares(self):
        rows = pd.DataFrame([position("2026-01-01", "100"),
                             position("2026-01-02", "100.0000005")])
        self.assertEqual(qc.check_ownership_continuity(rows).iloc[1]["continuity_status"], "PASS")
        self.assertEqual(qc.check_ownership_continuity(rows, tolerance=0).iloc[1]["continuity_status"],
                         "FLAG_REVIEW_UNADJUSTED")

    def test_parse_review_and_joint_owners_do_not_update_positions(self):
        rows = pd.DataFrame([
            position("2026-01-01", 100),
            position("2026-01-02", 999, parse_status="needs_review"),
            position("2026-01-03", 999, owner_attribution_status="joint_owner_unallocated"),
            position("2026-01-04", 100),
        ])
        result = qc.check_ownership_continuity(rows)
        self.assertEqual(result.continuity_status.tolist(),
                         ["BASELINE_NO_PRIOR", "NOT_TESTABLE_PARSE_STATUS", "NOT_TESTABLE_OWNER_ATTRIBUTION", "PASS"])
        self.assertEqual(result.position_state_updated.tolist(), [True, False, False, True])
        self.assertEqual(len(result), len(rows))

    def test_review_amendment_cannot_replace_original(self):
        rows = pd.DataFrame([
            position("2026-01-01", 100, owner_row_id="original"),
            position("2026-01-01", 999, owner_row_id="correction", form="4/A",
                     edgar_filing_date="2026-01-02", amends_owner_row_id="original", parse_status="needs_review"),
        ])
        result = qc.resolve_amendments(rows)
        self.assertEqual(result.record_active.tolist(), [True, False])
        self.assertEqual(result.iloc[1]["amendment_status"], "unreviewed_parse")

    def test_explicit_amendments_respect_availability_and_preserve_prior_results(self):
        rows = pd.DataFrame([
            position("2026-01-01", 100, owner_row_id="original", acceptance_datetime="2026-01-02T12:00:00-05:00"),
            position("2026-01-01", 120, owner_row_id="correction", form="4/A", amends_owner_row_id="original",
                     acceptance_datetime="2026-01-03T12:00:00-05:00"),
            position("2026-01-04", 130, shares="10", acceptance_datetime="2026-01-05T12:00:00-05:00"),
        ])
        before = qc.check_ownership_continuity(rows, as_of="2026-01-02T18:00:00Z")
        after = qc.check_ownership_continuity(rows)
        self.assertEqual(before.continuity_status.tolist(),
                         ["BASELINE_NO_PRIOR", "NOT_AVAILABLE_AS_OF", "NOT_AVAILABLE_AS_OF"])
        self.assertEqual(after.continuity_status.tolist(), ["BASELINE_NO_PRIOR", "AMENDMENT_BASELINE_RESET", "PASS"])


class PriceV2Tests(unittest.TestCase):
    def test_numeric_permnos_and_shared_identifier_selection(self):
        transactions = pd.DataFrame([sale(permno=12345.0)])
        market = pd.DataFrame([{"permno": 12345, "ticker": "T", "date": "2026-01-05", "low": 49, "high": 51}])
        for tx, prices in ((transactions, market), (transactions, market.drop(columns="permno")),
                           (transactions.drop(columns="permno"), market)):
            with self.subTest(columns=list(prices)):
                self.assertEqual(qc.check_transaction_prices(tx, prices).iloc[0]["price_check_status"], "PASS_EXACT_DAY_RANGE")

    def test_timezone_dates_keep_the_supplied_calendar_day(self):
        transactions = pd.DataFrame([sale(transaction_date="2026-01-05T23:30:00-05:00")])
        market = pd.DataFrame([{"ticker": "T", "date": "2026-01-05T00:00:00Z", "low": 49, "high": 51}])
        result = qc.check_transaction_prices(transactions, market).iloc[0]
        self.assertEqual(result["transaction_date"], "2026-01-05")
        self.assertEqual(result["price_check_status"], "PASS_EXACT_DAY_RANGE")

    def test_nonfinite_close_does_not_invalidate_usable_bounds_or_emit_infinite_distance(self):
        for close in (float("inf"), -float("inf"), float("nan")):
            with self.subTest(close=close):
                market = pd.DataFrame([{"ticker": "T", "date": "2026-01-05", "low": 49, "high": 51, "close": close}])
                result = qc.check_transaction_prices(pd.DataFrame([sale()]), market).iloc[0]
                self.assertEqual(result["price_check_status"], "PASS_EXACT_DAY_RANGE")
                self.assertTrue(pd.isna(result["exact_close"]))
                self.assertTrue(pd.isna(result["distance_from_exact_close"]))

    def test_missing_incomplete_and_invalid_bounds_are_not_price_failures(self):
        for low, high in ((None, None), (49, None), (51, 49), (float("inf"), float("inf")), (0, 0)):
            with self.subTest(low=low, high=high):
                market = pd.DataFrame([{"ticker": "T", "date": "2026-01-05", "low": low, "high": high}])
                result = qc.check_transaction_prices(pd.DataFrame([sale()]), market)
                self.assertEqual(result.iloc[0]["price_check_status"], "NO_MARKET_DATA")
        partial = pd.DataFrame([{"ticker": "T", "date": "2026-01-05", "low": 49, "high": None},
                                {"ticker": "T", "date": "2026-01-06", "low": None, "high": 51}])
        self.assertEqual(qc.check_transaction_prices(pd.DataFrame([sale()]), partial).iloc[0]["price_check_status"], "NO_MARKET_DATA")

    def test_failure_names_an_available_window_and_reports_boundary_distance(self):
        market = pd.DataFrame([{"ticker": "T", "date": "2026-01-02", "low": 49, "high": 51}])
        result = qc.check_transaction_prices(pd.DataFrame([sale(price_per_share="100")]), market).iloc[0]
        self.assertEqual(result["price_check_status"], "FLAG_OUTSIDE_ADJACENT_RANGE")
        self.assertEqual(result["matched_window"], "adjacent_trading_days")
        self.assertEqual(result["distance_from_nearest_boundary"], 49)

    def test_review_rows_are_retained_without_price_judgments(self):
        market = pd.DataFrame([{"ticker": "T", "date": "2026-01-05", "low": 49, "high": 51}])
        for status in ("needs_review", "error", "unsupported_pdf", "empty"):
            with self.subTest(status=status):
                result = qc.check_transaction_prices(pd.DataFrame([sale(parse_status=status)]), market)
                self.assertEqual(result.iloc[0]["price_check_status"], "NOT_TESTABLE_PARSE_STATUS")
                self.assertEqual(len(result), 1)

    def test_independent_reported_value_uses_one_cent_tolerance(self):
        transactions = pd.DataFrame([sale(reported_transaction_value=value)
                                     for value in ("500.01", "499.99", "500.02", "")])
        result = qc.check_transaction_prices(transactions)
        self.assertEqual(result.transaction_value_status.tolist(),
                         ["PASS_REPORTED_VALUE", "PASS_REPORTED_VALUE", "FLAG_REPORTED_VALUE_DIFFERENCE",
                          "NO_INDEPENDENT_REPORTED_VALUE"])
        self.assertEqual(result.derived_transaction_value.tolist(), [500] * 4)

    def test_duplicate_numeric_security_dates_are_rejected_after_normalization(self):
        market = pd.DataFrame([{"permno": value, "date": "2026-01-05", "low": 49, "high": 51}
                               for value in ("12345", "12345.0")])
        with self.assertRaisesRegex(ValueError, "duplicate security/date"):
            qc.check_transaction_prices(pd.DataFrame([sale(permno=12345)]), market)

    def test_callback_span_misses_and_ticker_case(self):
        calls = []
        def lookup(ticker, day):
            calls.append((ticker, day))
            return {"low": 49, "high": 51} if day == pd.Timestamp("2026-01-15") else None
        transactions = pd.DataFrame([sale(), sale()])
        result = compatibility.check_transaction_prices(transactions, price_lookup=lookup, max_adjacent_calendar_days=14)
        self.assertEqual(result.price_status.tolist(), ["PASS_ADJACENT_DAY_RANGE"] * 2)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(calls), 29)
        self.assertEqual({ticker for ticker, _ in calls}, {"T"})

    def test_callback_keeps_calendar_week_when_adjacent_span_is_zero(self):
        def lookup(ticker, day):
            return {"low": 49, "high": 51} if day == pd.Timestamp("2026-01-09") else {}
        result = compatibility.check_transaction_prices(
            pd.DataFrame([sale(issuer_ticker=None, ticker="T", permno=123)]),
            price_lookup=lookup, max_adjacent_calendar_days=0)
        self.assertEqual(result.iloc[0]["price_status"], "PASS_WEEKLY_RANGE")

    def test_callback_rejects_missing_explicit_security_column_before_lookup(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "MISSING.*not present"):
            compatibility.check_transaction_prices(
                pd.DataFrame([sale()]), transaction_security_col="MISSING",
                price_lookup=lambda ticker, day: calls.append((ticker, day)))
        self.assertEqual(calls, [])


class VersionedCliTests(unittest.TestCase):
    def test_cli_writes_separate_versioned_results_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.csv"
            pd.DataFrame([position("2026-01-01", 100), position("2026-01-02", 100)]).to_csv(source, index=False)
            output = root / "v2"
            command = [sys.executable, "-B", "-m", "shared.sec.form4_parser.src.quality_checks_v2",
                       "--transactions", str(source), "--output-dir", str(output)]
            completed = subprocess.run(command, text=True, capture_output=True, check=True)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["quality_checks_version"], 2)
            self.assertEqual(summary["ownership_continuity_statuses"], {"BASELINE_NO_PRIOR": 1, "PASS": 1})
            original = (output / "ownership_continuity.csv").read_bytes()
            rejected = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("must not already exist", rejected.stderr)
            self.assertEqual((output / "ownership_continuity.csv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
